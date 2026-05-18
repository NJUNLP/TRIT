
class RayStSSoloLanguageFilterTrainer0114Comet(RayPPOTrainer):
    """Self-improving via Self-translation Trainer"""
    # 第一阶段把大于0的部分全部纳入训练
    def fit(self):
        """
        Self-Translation训练流程，包含三个阶段：
        1. 英文推理与过滤
        2. 翻译成目标语言
        3. 多语言推理与翻译验证
        """
        from omegaconf import OmegaConf
        from verl.utils.tracking import Tracking

        logger = Tracking(
            project_name=self.config.trainer.project_name,
            experiment_name=self.config.trainer.experiment_name,
            default_backend=self.config.trainer.logger,
            config=OmegaConf.to_container(self.config, resolve=True),
        )

        self.global_steps = 0

        # load checkpoint before doing anything
        self._load_checkpoint()
        
        # 获取StS相关配置
        self.target_language = self.config.data.get("target_language", "ZH")
        self.translation_acc_lower = self.config.data.get("translation_source_acc_lower", 0.5)
        self.translation_acc_upper = self.config.data.get("translation_source_acc_upper", 1.0)
        self.qt_training_ratio = self.config.data.get("qt_training_ratio", 1.0)
        
        # LLM Judge配置（替代Comet评分）
        self.llm_judge_api_base_url = os.getenv("API_BASE_URL", "")
        self.llm_judge_api_key = os.getenv("API_KEY", "")
        self.llm_judge_model_name = os.getenv("MODEL_NAME", "")
        self.llm_judge_max_concurrency = int(os.getenv("LLM_JUDGE_CONCURRENCY", "16"))
        self.llm_judge_timeout = int(os.getenv("LLM_JUDGE_TIMEOUT", "60"))

        # 读取LLM Judge prompt模板
        with open("/mnt/dolphinfs/ssd_pool/docker/user/hadoop-hldy-nlp/FMG/liujunxiao03/NJUWorkspace/TRIT/MeiTuan/TRIT/prompts/llm_judge_translation.txt", "r", encoding="utf-8") as f:
            self.llm_judge_prompt = f.read().strip()

        # 读取翻译prompt模板

        with open("/mnt/dolphinfs/ssd_pool/docker/user/hadoop-hldy-nlp/FMG/liujunxiao03/NJUWorkspace/TRIT/MeiTuan/TRIT/prompts/translation_template.txt", "r", encoding="utf-8") as f:
            self.translation_prompt = f.read().strip()

        # perform validation before training
        if self.val_reward_fn is not None and self.config.trainer.get("val_before_train", True) and self.global_steps == 0:
            val_metrics = self._validate()
            assert val_metrics, f"{val_metrics=}"
            pprint(f"Initial validation metrics: {val_metrics}")
            logger.log(data=val_metrics, step=self.global_steps)
            if self.config.trainer.get("val_only", False):
                return

        # add tqdm
        progress_bar = tqdm(total=self.total_training_steps, initial=self.global_steps, desc="StS Training Progress")

        # we start from step 1
        self.global_steps += 1
        last_val_metrics = None
        self.max_steps_duration = 0
        n_samples = self.config.actor_rollout_ref.rollout.n

        for epoch in range(self.config.trainer.total_epochs):
            for batch_dict in self.train_dataloader:
                do_profile = self.global_steps in self.config.trainer.profile_steps if self.config.trainer.profile_steps is not None else False
                if do_profile:
                    self.actor_rollout_wg.start_profile()
                    if self.use_reference_policy:
                        self.ref_policy_wg.start_profile()
                    if self.use_critic:
                        self.critic_wg.start_profile()
                    if self.use_rm:
                        self.rm_wg.start_profile()

                metrics = {}
                timing_raw = {}
                batch: DataProto = DataProto.from_single_dict(batch_dict)

                # pop those keys for generation
                batch_keys_to_pop = ["input_ids", "attention_mask", "position_ids"]
                non_tensor_batch_keys_to_pop = ["raw_prompt_ids"]
                if "multi_modal_data" in batch.non_tensor_batch:
                    non_tensor_batch_keys_to_pop.append("multi_modal_data")
                if "raw_prompt" in batch.non_tensor_batch:
                    non_tensor_batch_keys_to_pop.append("raw_prompt")
                if "tools_kwargs" in batch.non_tensor_batch:
                    non_tensor_batch_keys_to_pop.append("tools_kwargs")
                if "interaction_kwargs" in batch.non_tensor_batch:
                    non_tensor_batch_keys_to_pop.append("interaction_kwargs")
                gen_batch = batch.pop(
                    batch_keys=batch_keys_to_pop,
                    non_tensor_batch_keys=non_tensor_batch_keys_to_pop,
                )

                is_last_step = self.global_steps >= self.total_training_steps

                with marked_timer("step", timing_raw):
                    ##### 阶段一：英文推理与过滤 #####
                    print(f">>> 阶段一：英文推理与过滤")
                    with marked_timer("stage1_english_reasoning", timing_raw, color="red"):
                        if not self.async_rollout_mode:
                            gen_batch_output = self.actor_rollout_wg.generate_sequences(gen_batch)
                        else:
                            self.async_rollout_manager.wake_up()
                            gen_batch_output = self.async_rollout_manager.generate_sequences(gen_batch)
                            self.async_rollout_manager.sleep()
                        timing_raw.update(gen_batch_output.meta_info["timing"])
                        gen_batch_output.meta_info.pop("timing", None)

                    batch.non_tensor_batch["uid"] = np.array([str(uuid.uuid4()) for _ in range(len(batch.batch))], dtype=object)
                    # repeat to align with repeated responses in rollout
                    sample_languages = []
                    if "extra_info" in batch.non_tensor_batch:
                        for extra_info in batch.non_tensor_batch["extra_info"]:
                            # 从extra_info中提取lang字段
                            if isinstance(extra_info, dict) and "lang" in extra_info:
                                sample_lang = extra_info["lang"]
                            else:
                                assert False, "No Lang in Extra Info Stage1"
                                # 如果没有lang字段，使用默认的target_language
                                # sample_lang = self.target_language
                            sample_languages.append(sample_lang)
                        # 转换为numpy array
                        batch.non_tensor_batch["sample_lang"] = np.array(sample_languages, dtype=object)
                    else:
                        assert False, "No Extra Info Stage1"
                        # 如果没有extra_info，全部使用默认的target_language
                        # batch.non_tensor_batch["sample_lang"] = np.array([self.target_language] * len(batch.batch), dtype=object)

                    batch = batch.repeat(repeat_times=self.config.actor_rollout_ref.rollout.n, interleave=True)
                    batch = batch.union(gen_batch_output)
                    batch.batch["response_mask"] = compute_response_mask(batch)
                    
                    # 打印阶段一样例
                    print(f"\n{'='*80}")
                    print(f"阶段一样例（英文推理）")
                    print(f"{'='*80}")
                    stage1_sample_idx = 0  # 打印第一个样例
                    stage1_prompt = self.tokenizer.decode(batch.batch['prompts'][stage1_sample_idx], skip_special_tokens=True)
                    stage1_response = self.tokenizer.decode(batch.batch['responses'][stage1_sample_idx], skip_special_tokens=True)
                    print(f"[Prompt]:\n{stage1_prompt}")
                    print(f"\n[Response]:\n{stage1_response}")
                    print(f"{'='*80}\n")

                    # Balance the number of valid tokens across DP ranks
                    if self.config.trainer.balance_batch:
                        self._balance_batch(batch, metrics=metrics)

                    # compute global_valid tokens
                    batch.meta_info["global_token_num"] = torch.sum(batch.batch["attention_mask"], dim=-1).tolist()
                    stage1_use_en_think = os.getenv("STAGE1_USE_EN_THINK")
                    if stage1_use_en_think and stage1_use_en_think.lower() in ["true", "1"]:
                        # print("--------------阶段一使用英文think---------------------------")
                        os.environ["USE_ALL_EN_THINK"] = "1"
                    # 计算英文回答的奖励
                    with marked_timer("stage1_reward", timing_raw):
                        if self.use_rm:
                            reward_tensor = self.rm_wg.compute_rm_score(batch)
                            batch = batch.union(reward_tensor)

                        if self.config.reward_model.launch_reward_fn_async:
                            future_reward = compute_reward_async.remote(batch, self.config, self.tokenizer)
                            reward_tensor, stage1_reward_extra_infos_dict = ray.get(future_reward)
                        else:
                            reward_tensor, stage1_reward_extra_infos_dict = compute_reward(batch, self.reward_fn)
                        batch.batch['reward_tensor'] = reward_tensor  # 直接使用reward_tensor，不需要[1]索引
                    
                    metrics['sts/stage1_english_reward'] = torch.mean(torch.sum(batch.batch['reward_tensor'], dim=-1)).item()
                    if stage1_use_en_think and stage1_use_en_think.lower() in ["true", "1"]:
                        os.environ["USE_ALL_EN_THINK"] = "0"
                    # 记录阶段一的额外奖励信息
                    if stage1_reward_extra_infos_dict:
                        for key_info in ['accuracy_reward', 'format_reward', 'language_reward', 'language_probability', 'language_weight']:
                            if key_info in stage1_reward_extra_infos_dict and len(stage1_reward_extra_infos_dict[key_info]) > 0:
                                metrics[f"reward/stage1_{key_info}_mean"] = np.mean(stage1_reward_extra_infos_dict[key_info])
                                metrics[f"reward/stage1_{key_info}_std"] = np.std(stage1_reward_extra_infos_dict[key_info])
                                metrics[f"reward/stage1_{key_info}_max"] = np.max(stage1_reward_extra_infos_dict[key_info])
                                metrics[f"reward/stage1_{key_info}_min"] = np.min(stage1_reward_extra_infos_dict[key_info])

                    sts_metrics = {}
                    
                    # 计算每个样本的准确率
                    sample_accs = {}
                    for idx in range(0, len(batch.non_tensor_batch["uid"])):
                        if batch.non_tensor_batch["uid"][idx] not in sample_accs:
                            sample_accs[batch.non_tensor_batch["uid"][idx]] = []
                        sample_accs[batch.non_tensor_batch["uid"][idx]].append(batch.batch['acc'][idx].item())
                    sample_accs = {k: np.mean(v).item() for k, v in sample_accs.items()}

                    # 阶段一过滤：准确率过滤
                    stage1_filtered_indices = []     # 所有准确率>=下限的数据
                    stage1_training_indices = []     # 用于参数更新的数据（准确率在[lower, upper)区间）
                    stage2_input_indices = []        # 进入阶段二的数据（准确率>=下限的问题，每个问题只选一次）
                    use_all_data = os.getenv("USE_ALL_DATA", None)
                    use_all_stage1_data = True if use_all_data and use_all_data.lower() in ["true", "1"] else False

                    for k in sample_accs.keys():
                        cur_uid_indices = np.where(batch.non_tensor_batch["uid"] == k)[0].tolist()
                        
                        # 准确率>=下限的数据都可以进入后续阶段
                        if sample_accs[k] >= self.translation_acc_lower or use_all_stage1_data:
                            stage1_filtered_indices.extend(cur_uid_indices)
                            
                            # 进入翻译阶段：准确率>=下限的问题进入翻译
                            stage2_input_indices.append(cur_uid_indices[0])
                            
                            # 用于参数更新：准确率在[lower, upper)区间内的数据
                            if 0 < sample_accs[k] < self.translation_acc_upper:
                                stage1_training_indices.extend(cur_uid_indices)
                        elif sample_accs[k] > 0:
                            stage1_training_indices.extend(cur_uid_indices)

                    if len(stage2_input_indices) == 0:
                        print("阶段一过滤后没有合适的数据进入阶段二，跳过此iteration")
                        continue
                    
                    sts_metrics["StS-Valid-Ratio/Stage1-all-filtered"] = np.round(len(stage1_filtered_indices) * 100 / len(batch), 2).item()
                    sts_metrics["StS-Valid-Ratio/Stage1-training"] = np.round(len(stage1_training_indices) * 100 / len(batch), 2).item()
                    sts_metrics["StS-Valid-Ratio/Stage2-input"] = np.round(len(stage2_input_indices) * 100 / len(batch), 2).item()

                    ##### 阶段二：翻译成目标语言 #####
                    print(f">>> 阶段二：翻译成目标语言 ({self.target_language})")
                    
                    # 准备翻译输入
                    stage2_batch = batch.select_idxs(stage2_input_indices)
                    stage2_batch.non_tensor_batch["uid"] = np.array([str(uuid.uuid4()) for _ in range(len(stage2_batch.batch))], dtype=object)
                    stage2_languages = stage2_batch.non_tensor_batch["sample_lang"].tolist()
                    
                    # 获取英文问题文本（优先使用数据集中的en_question字段）
                    if "en_question" in stage2_batch.non_tensor_batch:
                        english_questions = stage2_batch.non_tensor_batch["en_question"].tolist()
                    else:
                        # 如果没有en_question字段，则从聚天模板中提取并保存为en_question
                        english_questions_raw = self.tokenizer.batch_decode(stage2_batch.batch['prompts'], skip_special_tokens=True)
                        # 清理提取的英文问题
                        cleaned_english_questions = []
                        for eq in english_questions_raw:
                            if "<|im_start|>user\n" in eq and "<|im_end|>" in eq:
                                clean_question = eq.split("<|im_start|>user\n")[1].split("<|im_end|>")[0].strip()
                            else:
                                clean_question = eq.strip()
                            cleaned_english_questions.append(clean_question)
                        english_questions = cleaned_english_questions
                        # 保存为en_question，以便后续统一使用
                        stage2_batch.non_tensor_batch["en_question"] = np.array(cleaned_english_questions, dtype=object)
                    
                    # 构建翻译输入
                    translation_inputs = []
                    for eq, lang in zip(english_questions, stage2_languages):
                        # 如果是从数据集获取的en_question，直接使用
                        if "en_question" in stage2_batch.non_tensor_batch:
                            clean_question = eq
                        else:
                            # 如果是从聚天模板解码的，需要提取原始问题
                            if "<|im_start|>user\n" in eq and "<|im_end|>" in eq:
                                clean_question = eq.split("<|im_start|>user\n")[1].split("<|im_end|>")[0].strip()
                            else:
                                clean_question = eq.strip()
                        
                        translation_input = self.translation_prompt.format(
                            language=LANGUAGE_MAP[lang.upper()],  # 使用样本特定的语言
                            question=clean_question
                        )
                        translation_input = self.tokenizer.apply_chat_template(
                            [{"content": translation_input, "role": "user"}], 
                            tokenize=False, 
                            add_generation_prompt=True
                        )
                        translation_input += f"<think>\nOkay, I will translate the English question into {LANGUAGE_MAP[lang.upper()]}."
                        translation_inputs.append(translation_input)
                    
                    # 构建翻译生成batch
                    translation_model_inputs = self.tokenizer(
                        translation_inputs, 
                        return_tensors="pt", 
                        add_special_tokens=False, 
                        padding=True, 
                        truncation=True, 
                        max_length=self.config.data.max_prompt_length
                    )
                    
                    trans_input_ids = translation_model_inputs.pop("input_ids")
                    trans_attention_mask = translation_model_inputs.pop("attention_mask")
                    trans_input_ids, trans_attention_mask = verl_F.postprocess_data(
                        input_ids=trans_input_ids,
                        attention_mask=trans_attention_mask,
                        max_length=self.config.data.max_prompt_length,
                        pad_token_id=self.tokenizer.pad_token_id,
                        left_pad=True,
                        truncation=self.config.data.get("truncation", "error"),
                    )
                    trans_position_ids = compute_position_id_with_mask(trans_attention_mask)
                    trans_raw_prompt_ids = np.array(self.tokenizer.batch_encode_plus(translation_inputs, add_special_tokens=False)['input_ids'], dtype=object)
                    trans_tools_kwargs = np.array([{} for _ in range(len(trans_position_ids))], dtype=object)
                    
                    trans_gen_dict = {
                        "input_ids": trans_input_ids,
                        "attention_mask": trans_attention_mask,
                        "position_ids": trans_position_ids,
                        "raw_prompt_ids": trans_raw_prompt_ids,
                        "tools_kwargs": trans_tools_kwargs,
                    }
                    
                    # 生成翻译结果
                    with marked_timer("stage2_translation", timing_raw, color="blue"):
                        trans_gen_batch = DataProto.from_single_dict(trans_gen_dict)
                        # 按照SvS的模式，显式进行repeat
                        trans_gen_batch = trans_gen_batch.repeat(repeat_times=self.config.actor_rollout_ref.rollout.translation_sample_n, interleave=True)
                        
                        trans_gen_batch.meta_info = {
                            "translation": True, 
                            "temperature": self.config.actor_rollout_ref.rollout.get("translation_temperature", 0.7), 
                            "top_p": self.config.actor_rollout_ref.rollout.get("translation_top_p", 0.9), 
                            "n": 1
                        }
                        
                        # 打印阶段二样例（翻译输入）
                        print(f"\n{'='*80}")
                        print(f"阶段二样例（翻译）- 输入")
                        print(f"{'='*80}")
                        stage2_sample_idx = 0  # 打印第一个样例
                        stage2_prompt = self.tokenizer.decode(trans_gen_batch.batch['input_ids'][stage2_sample_idx], skip_special_tokens=True)
                        print(f"[Prompt]:\n{stage2_prompt}")
                        print(f"{'='*80}\n")
                        
                        trans_gen_batch_padded, pad_size = pad_dataproto_to_divisor(trans_gen_batch, self.actor_rollout_wg.world_size)
                        if not self.async_rollout_mode:
                            trans_output_padded = self.actor_rollout_wg.generate_sequences(trans_gen_batch_padded)
                        else:
                            self.async_rollout_manager.wake_up()
                            trans_output_padded = self.async_rollout_manager.generate_sequences(trans_gen_batch_padded)
                            self.async_rollout_manager.sleep()
                        trans_output = unpad_dataproto(trans_output_padded, pad_size=pad_size)
                        
                        # 打印阶段二样例（翻译输出）
                        print(f"\n{'='*80}")
                        print(f"阶段二样例（翻译）- 输出")
                        print(f"{'='*80}")
                        stage2_response = self.tokenizer.decode(trans_output.batch['responses'][stage2_sample_idx], skip_special_tokens=True)
                        print(f"[Response]:\n{stage2_response}")
                        print(f"{'='*80}\n")
                    
                    # 更新stage2_batch
                    # 按照SvS的模式，对stage2_batch也进行相同的repeat操作
                    print(f">>> Debug: 在repeat前 stage2_batch batch size: {len(stage2_batch.batch)}")
                    print(f">>> Debug: trans_output batch size: {len(trans_output.batch)}")
                    stage2_batch = stage2_batch.repeat(repeat_times=self.config.actor_rollout_ref.rollout.translation_sample_n, interleave=True)
                    
                    print(f">>> Debug: 在repeat后 stage2_batch batch size: {len(stage2_batch.batch)}")
                    stage2_batch_keys_to_pop = ['prompts', 'responses', 'input_ids', 'attention_mask', 'position_ids']
                    stage2_non_tensor_batch_keys_to_pop = ["tools_kwargs"]
                    stage2_batch.pop(batch_keys=stage2_batch_keys_to_pop, non_tensor_batch_keys=stage2_non_tensor_batch_keys_to_pop)
                    stage2_batch = stage2_batch.union(trans_output)
                    stage2_batch.batch["response_mask"] = compute_response_mask(stage2_batch)
                    stage2_languages = stage2_batch.non_tensor_batch["sample_lang"].tolist()
                    # 提取翻译结果
                    valid_translation_indices = []
                    translated_questions = []
                    responses_strs = self.tokenizer.batch_decode(stage2_batch.batch['responses'], skip_special_tokens=True)
                    
                    for idx, response in enumerate(responses_strs):
                        try:
                            # 提取翻译结果
                            sample_lang = stage2_languages[idx]
                            extracted = extract_last_translation(response, sample_lang)
                            if extracted and len(extracted) > 10:  # 基本长度检查
                                translated_questions.append(extracted)
                                valid_translation_indices.append(idx)
                            else:
                                translated_questions.append(None)
                        except:
                            translated_questions.append(None)
                    
                    # 过滤有效翻译
                    valid_indices = [i for i in valid_translation_indices if translated_questions[i] is not None]
                    if len(valid_indices) == 0:
                        print("阶段二翻译后没有有效结果，跳过此iteration")
                        continue
                    
                    sts_metrics["StS-Valid-Ratio/Stage2-extracted"] = np.round(len(valid_indices) * 100 / len(stage2_batch), 2).item()
                    
                    # 使用外部LLM Judge为翻译打分（二值评估：0 or 1）
                    print(f">>> 使用LLM Judge为翻译打分")
                    with marked_timer("stage2_llm_judge_scoring", timing_raw, color="yellow"):
                        import requests
                        from concurrent.futures import ThreadPoolExecutor, as_completed
                        import re as _re

                        llm_judge_scores = {}
                        judge_tasks = []  # (position_in_valid_indices, idx, src_text, mt_text, language)

                        # 准备LLM Judge评分请求数据
                        translation_sample_n = self.config.actor_rollout_ref.rollout.translation_sample_n
                        original_question_count = len(stage2_batch) // translation_sample_n

                        print(f">>> LLM Judge评分准备: translation_sample_n={translation_sample_n}, original_question_count={original_question_count}, stage2_batch长度={len(stage2_batch)}, valid_indices数量={len(valid_indices)}")
                        if "en_question" in stage2_batch.non_tensor_batch:
                            en_q_len = len(stage2_batch.non_tensor_batch["en_question"])
                            print(f">>> en_question存在，长度={en_q_len}")
                            if en_q_len > 0:
                                en_q_samples = []
                                for i in range(min(3, en_q_len)):
                                    en_q_str = str(stage2_batch.non_tensor_batch['en_question'][i])
                                    if len(en_q_str) > 50:
                                        en_q_samples.append(en_q_str[:50] + "...")
                                    else:
                                        en_q_samples.append(en_q_str)
                                print(f">>> en_question前3个示例: {en_q_samples}")
                        else:
                            print(f">>> en_question不存在，将从prompts中提取")

                        for pos, idx in enumerate(valid_indices):
                            if translation_sample_n > 1:
                                orig_q_idx = idx // translation_sample_n
                            else:
                                orig_q_idx = idx

                            if "en_question" in stage2_batch.non_tensor_batch:
                                if idx < len(stage2_batch.non_tensor_batch["en_question"]):
                                    src_text = stage2_batch.non_tensor_batch["en_question"][idx]
                                else:
                                    print(f">>> 警告: idx={idx}超出en_question长度={len(stage2_batch.non_tensor_batch['en_question'])}")
                                    src_text = ""
                            else:
                                prompt_text = self.tokenizer.decode(stage2_batch.batch['prompts'][idx], skip_special_tokens=True)
                                if "<|im_start|>user\n" in prompt_text and "<|im_end|>" in prompt_text:
                                    src_text = prompt_text.split("<|im_start|>user\n")[1].split("<|im_end|>")[0].strip()
                                else:
                                    src_text = prompt_text.strip()

                            mt_text = translated_questions[idx]
                            sample_lang = stage2_languages[idx] if idx < len(stage2_languages) else self.target_language

                            if pos < min(5, len(valid_indices)):
                                src_preview = src_text[:50] + "..." if len(str(src_text)) > 50 else src_text
                                mt_preview = mt_text[:50] + "..." if len(str(mt_text)) > 50 else mt_text
                                print(f">>> LLM Judge样本[{idx}]: orig_q_idx={orig_q_idx}, src={src_preview}, mt={mt_preview}")

                            judge_tasks.append((pos, idx, src_text, mt_text, sample_lang))

                        def _call_llm_judge(task_item):
                            """调用外部LLM对单个翻译进行二值评估，返回 (position, idx, score)"""
                            pos, idx, src_text, mt_text, lang = task_item
                            try:
                                judge_prompt_filled = self.llm_judge_prompt.format(
                                    question=src_text,
                                    translation=mt_text,
                                    language=LANGUAGE_MAP.get(lang.upper(), lang)
                                )

                                headers = {
                                    "Content-Type": "application/json",
                                    "Authorization": f"Bearer {self.llm_judge_api_key}"
                                }
                                payload = {
                                    "model": self.llm_judge_model_name,
                                    "messages": [
                                        {"role": "user", "content": judge_prompt_filled}
                                    ],
                                    "temperature": 0.0,
                                    "max_tokens": 512
                                }

                                api_url = self.llm_judge_api_base_url.rstrip("/")
                                if not api_url.endswith("/chat/completions"):
                                    api_url = api_url + "/chat/completions"

                                resp = requests.post(
                                    api_url,
                                    json=payload,
                                    headers=headers,
                                    timeout=self.llm_judge_timeout
                                )

                                if resp.status_code == 200:
                                    result = resp.json()
                                    content = result["choices"][0]["message"]["content"]
                                    # 解析结果：从 \boxed{Correct} 或 \boxed{Incorrect} 中提取
                                    if _re.search(r'\\boxed\{?\s*Correct\s*\}?', content):
                                        return (pos, idx, 1)
                                    elif _re.search(r'\\boxed\{?\s*Incorrect\s*\}?', content):
                                        return (pos, idx, 0)
                                    else:
                                        # 无法解析，保守给0
                                        print(f">>> LLM Judge结果无法解析 idx={idx}: {content[:100]}")
                                        return (pos, idx, 0)
                                else:
                                    print(f">>> LLM Judge调用失败 idx={idx}: HTTP {resp.status_code}")
                                    return (pos, idx, 0)
                            except Exception as e:
                                print(f">>> LLM Judge调用异常 idx={idx}: {e}")
                                return (pos, idx, 0)

                        # 使用ThreadPoolExecutor并发调用LLM Judge
                        results = [None] * len(judge_tasks)
                        with ThreadPoolExecutor(max_workers=self.llm_judge_max_concurrency) as executor:
                            future_to_task = {executor.submit(_call_llm_judge, task): task for task in judge_tasks}
                            for future in as_completed(future_to_task):
                                pos, idx, score = future.result()
                                results[pos] = (idx, score)

                        # 按原始顺序收集结果
                        scores_list = []
                        for pos, (idx, score) in enumerate(results):
                            llm_judge_scores[idx] = score
                            scores_list.append(score)

                        correct_count = sum(1 for s in scores_list if s == 1)
                        incorrect_count = sum(1 for s in scores_list if s == 0)
                        accuracy = correct_count / len(scores_list) if len(scores_list) > 0 else 0.0

                        print(f">>> LLM Judge评分完成: {len(scores_list)}个翻译已评分")
                        print(f">>> LLM Judge结果: Correct={correct_count}, Incorrect={incorrect_count}, Accuracy={accuracy:.4f}")

                        # 记录LLM Judge分数统计
                        metrics['sts/stage2_llm_judge_accuracy'] = accuracy
                        metrics['sts/stage2_llm_judge_correct_count'] = correct_count
                        metrics['sts/stage2_llm_judge_incorrect_count'] = incorrect_count
                        metrics['sts/stage2_llm_judge_total'] = len(scores_list)
                    
                    # 计算翻译阶段的奖励（稀疏奖励，只在最后一个token，二值：0 or 1）
                    translation_reward_tensor = torch.zeros_like(stage2_batch.batch["responses"], dtype=torch.float32)
                    valid_response_length = stage2_batch.batch["attention_mask"][:, stage2_batch.batch["prompts"].shape[-1]:].sum(dim=-1)

                    # 使用LLM Judge二值分数作为翻译奖励（0 or 1）
                    for idx in valid_indices:
                        reward_value = float(llm_judge_scores.get(idx, 0))
                        translation_reward_tensor[idx, valid_response_length[idx].item() - 1] = reward_value
                    
                    # 对于无效翻译（不在valid_indices中的），奖励保持为0
                    stage2_batch.batch['reward_tensor'] = translation_reward_tensor
                    stage2_batch.batch['acc'] = torch.sum(translation_reward_tensor, dim=1)
                    
                    ##### 阶段三：多语言推理与翻译验证 #####
                    print(f">>> 阶段三：多语言推理与翻译验证")
                    
                    # 为有效翻译构建多语言推理输入
                    stage3_batch = stage2_batch.select_idxs(valid_indices)
                    if DEBUGGING_STS:
                        print(f">>> Debug: 阶段三初始 stage3_batch size: {len(stage3_batch.batch)}")
                        
                        # ===== Debug信息：验证select_idxs后的uid继承 =====
                        print(f"\n========== 阶段三batch构建 ==========")
                        print(f"从stage2选取的valid_indices: {valid_indices[:5]}... (前5个)")
                        print(f"stage3_batch初始大小: {len(stage3_batch)}")
                        print(f"即将为stage3_batch生成{len(stage3_batch)}个新uid")
                        print(f"=========================================\n")
                    
                    stage3_batch.non_tensor_batch["uid"] = np.array([str(uuid.uuid4()) for _ in range(len(stage3_batch.batch))], dtype=object)
                    stage3_languages = stage3_batch.non_tensor_batch["sample_lang"].tolist()
                    multilingual_questions = [translated_questions[i] for i in valid_indices]
                    multilingual_inputs = []
                    for mq, lang in zip(multilingual_questions, stage3_languages):
                        # 使用当前样本的目标语言
                        reasoning_input = mq + LANGUAGE_REASONING_MAP[lang.upper()]  # 使用样本特定的语言
                        
                        reasoning_input = self.tokenizer.apply_chat_template(
                            [{"content": reasoning_input, "role": "user"}], 
                            tokenize=False, 
                            add_generation_prompt=True
                        )
                        reasoning_input += LANGUAGE_START_PREFIX_MAP[lang.upper()]  # 使用样本特定的语言
                        multilingual_inputs.append(reasoning_input)
                    
                    # 构建多语言推理batch
                    multilingual_model_inputs = self.tokenizer(
                        multilingual_inputs, 
                        return_tensors="pt", 
                        add_special_tokens=False, 
                        padding=True, 
                        truncation=True,
                        max_length=self.config.data.max_prompt_length
                    )
                    
                    multi_input_ids = multilingual_model_inputs.pop("input_ids")
                    multi_attention_mask = multilingual_model_inputs.pop("attention_mask")
                    multi_input_ids, multi_attention_mask = verl_F.postprocess_data(
                        input_ids=multi_input_ids,
                        attention_mask=multi_attention_mask,
                        max_length=self.config.data.max_prompt_length,
                        pad_token_id=self.tokenizer.pad_token_id,
                        left_pad=True,
                        truncation=self.config.data.get("truncation", "error"),
                    )
                    multi_position_ids = compute_position_id_with_mask(multi_attention_mask)
                    multi_raw_prompt_ids = np.array(self.tokenizer.batch_encode_plus(multilingual_inputs, add_special_tokens=False)['input_ids'], dtype=object)
                    multi_tools_kwargs = np.array([{} for _ in range(len(multi_position_ids))], dtype=object)
                    
                    multi_gen_dict = {
                        "input_ids": multi_input_ids,
                        "attention_mask": multi_attention_mask,
                        "position_ids": multi_position_ids,
                        "raw_prompt_ids": multi_raw_prompt_ids,
                        "tools_kwargs": multi_tools_kwargs,
                    }
                    
                    # 生成多语言推理结果
                    with marked_timer("stage3_multilingual_reasoning", timing_raw, color="green"):
                        multi_gen_batch = DataProto.from_single_dict(multi_gen_dict)
                        print(f">>> Debug: 阶段三多语言推理前 batch size: {len(multi_gen_batch.batch)}")
                        
                        # 打印阶段三样例（多语言推理输入）
                        print(f"\n{'='*80}")
                        print(f"阶段三样例（多语言推理）- 输入")
                        print(f"{'='*80}")
                        stage3_sample_idx = 0  # 打印第一个样例
                        stage3_prompt = self.tokenizer.decode(multi_gen_batch.batch['input_ids'][stage3_sample_idx], skip_special_tokens=True)
                        print(f"[Prompt]:\n{stage3_prompt}")
                        print(f"{'='*80}\n")
                        
                        # ===== Debug信息：验证repeat操作 =====
                        print(f"\n========== Stage3 Repeat操作验证 ==========")
                        before_repeat_size = len(multi_gen_batch.batch)
                        print(f"repeat前大小: {before_repeat_size}")
                        print(f"repeat参数: repeat_times={self.config.actor_rollout_ref.rollout.n}, interleave=True")
                        
                        multi_gen_batch = multi_gen_batch.repeat(repeat_times=self.config.actor_rollout_ref.rollout.n, interleave=True)
                        after_repeat_size = len(multi_gen_batch.batch)
                        print(f"repeat后大小: {after_repeat_size}")
                        print(f"预期大小: {before_repeat_size * self.config.actor_rollout_ref.rollout.n}")
                        print(f"大小匹配: {after_repeat_size == before_repeat_size * self.config.actor_rollout_ref.rollout.n}")
                        print(f"=========================================\n")
                        # print(f">>> Debug: 阶段三多语言推理repeat后 batch size: {len(multi_gen_batch.batch)}")
                        multi_gen_batch.meta_info = {"multilingual_reasoning": True, "n": 1}
                        
                        multi_gen_batch_padded, pad_size = pad_dataproto_to_divisor(multi_gen_batch, self.actor_rollout_wg.world_size)
                        if not self.async_rollout_mode:
                            multi_output_padded = self.actor_rollout_wg.generate_sequences(multi_gen_batch_padded)
                        else:
                            self.async_rollout_manager.wake_up()
                            multi_output_padded = self.async_rollout_manager.generate_sequences(multi_gen_batch_padded)
                            self.async_rollout_manager.sleep()
                        multi_output = unpad_dataproto(multi_output_padded, pad_size=pad_size)
                        
                        # 打印阶段三样例（多语言推理输出）
                        print(f"\n{'='*80}")
                        print(f"阶段三样例（多语言推理）- 输出")
                        print(f"{'='*80}")
                        stage3_response = self.tokenizer.decode(multi_output.batch['responses'][stage3_sample_idx], skip_special_tokens=True)
                        print(f"[Response]:\n{stage3_response}")
                        print(f"{'='*80}\n")
                        
                        if DEBUGGING_STS:
                            print(f">>> Debug: 阶段三multi_output batch size: {len(multi_output.batch)}")
                    
                    # 更新stage3_batch（需要repeat以匹配multi_output的大小）
                    if DEBUGGING_STS:
                        print(f">>> Debug: stage3_batch union前 batch size: {len(stage3_batch.batch)}")
                        
                        # ===== Debug信息：验证stage3_batch的repeat和uid分布 =====
                        print(f"\n========== Stage3_batch Repeat和UID分布 ==========")
                        print(f"stage3_batch在repeat前的uid数量: {len(set(stage3_batch.non_tensor_batch['uid']))}")
                        original_uids = stage3_batch.non_tensor_batch['uid'].copy()
                        print(f"前3个原始uid: {[uid[:8] + '...' for uid in original_uids[:3]]}")
                    
                    stage3_batch = stage3_batch.repeat(repeat_times=self.config.actor_rollout_ref.rollout.n, interleave=True)
                    
                    if DEBUGGING_STS:
                        print(f">>> Debug: stage3_batch repeat后 batch size: {len(stage3_batch.batch)}")
                        print(f"stage3_batch在repeat后的总大小: {len(stage3_batch)}")
                        print(f"stage3_batch在repeat后的唯一uid数量: {len(set(stage3_batch.non_tensor_batch['uid']))}")
                        
                        # 验证interleave=True的效果
                        print(f"\n验证interleave=True的效果（前{min(15, len(stage3_batch))}个样本的uid）:")
                        for i in range(min(15, len(stage3_batch))):
                            uid_short = stage3_batch.non_tensor_batch['uid'][i][:8]
                            original_idx = i % len(original_uids) if self.config.actor_rollout_ref.rollout.n > 1 else i
                            expected_uid = original_uids[original_idx][:8]
                            match = "✓" if uid_short == expected_uid else "✗"
                            print(f"  索引{i:2d}: uid={uid_short}... (预期={expected_uid}...) {match}")
                        print(f"=========================================\n")
                    stage3_batch_keys_to_pop = ['prompts', 'responses', 'input_ids', 'attention_mask', 'position_ids']
                    stage3_non_tensor_batch_keys_to_pop = ["tools_kwargs"]
                    meta_info_keys_to_pop = ["timing"]
                    stage3_batch.pop(batch_keys=stage3_batch_keys_to_pop, 
                        non_tensor_batch_keys=stage3_non_tensor_batch_keys_to_pop, 
                        meta_info_keys=meta_info_keys_to_pop)
                    # stage3_batch.pop(batch_keys=stage3_batch_keys_to_pop, non_tensor_batch_keys=stage3_non_tensor_batch_keys_to_pop)
                    stage3_batch = stage3_batch.union(multi_output)
                    stage3_batch.batch["response_mask"] = compute_response_mask(stage3_batch)
                    if DEBUGGING_STS:
                        print(f">>> Debug: stage3_batch union后 batch size: {len(stage3_batch.batch)}")
                    
                    # 计算多语言推理的奖励
                    with marked_timer("stage3_reward", timing_raw):
                        if self.use_rm:
                            reward_tensor = self.rm_wg.compute_rm_score(stage3_batch)
                            stage3_batch = stage3_batch.union(reward_tensor)
                        
                        if self.config.reward_model.launch_reward_fn_async:
                            future_reward = compute_reward_async.remote(stage3_batch, self.config, self.tokenizer)
                            reward_tensor, stage3_reward_extra_infos_dict = ray.get(future_reward)
                        else:
                            reward_tensor, stage3_reward_extra_infos_dict = compute_reward(stage3_batch, self.reward_fn)
                        stage3_batch.batch['multilingual_reward_tensor'] = reward_tensor  # 直接使用reward_tensor，不需要[1]索引
                    
                    # 记录阶段三的额外奖励信息
                    if stage3_reward_extra_infos_dict:
                        for key_info in ['accuracy_reward', 'format_reward', 'language_reward', 'language_probability', 'language_weight']:
                            if key_info in stage3_reward_extra_infos_dict and len(stage3_reward_extra_infos_dict[key_info]) > 0:
                                metrics[f"reward/stage3_{key_info}_mean"] = np.mean(stage3_reward_extra_infos_dict[key_info])
                                metrics[f"reward/stage3_{key_info}_std"] = np.std(stage3_reward_extra_infos_dict[key_info])
                                metrics[f"reward/stage3_{key_info}_max"] = np.max(stage3_reward_extra_infos_dict[key_info])
                                metrics[f"reward/stage3_{key_info}_min"] = np.min(stage3_reward_extra_infos_dict[key_info])
                    
                    # 计算多语言推理准确率
                    stage3_sample_accs = {}
                    for idx in range(0, len(stage3_batch.non_tensor_batch["uid"])):
                        if stage3_batch.non_tensor_batch["uid"][idx] not in stage3_sample_accs:
                            stage3_sample_accs[stage3_batch.non_tensor_batch["uid"][idx]] = []
                        stage3_sample_accs[stage3_batch.non_tensor_batch["uid"][idx]].append(stage3_batch.batch['acc'][idx].item())
                    
                    if DEBUGGING_STS:
                        # ===== Debug信息：验证数据结构 =====
                        print(f"\n========== 阶段三数据结构分析 ==========")
                        print(f"valid_indices长度: {len(valid_indices)}")
                        print(f"stage3_batch总长度: {len(stage3_batch)}")
                        print(f"stage3中唯一uid数量: {len(stage3_sample_accs)}")
                        print(f"每个uid的样本数: {[len(v) for v in list(stage3_sample_accs.values())[:3]]}... (前3个)")
                        print(f"rollout.n配置: {self.config.actor_rollout_ref.rollout.n}")
                        
                        # 展示uid分布（前10个样本）
                        print(f"\n前10个stage3样本的uid:")
                        for idx in range(min(10, len(stage3_batch))):
                            print(f"  索引{idx}: uid={stage3_batch.non_tensor_batch['uid'][idx][:8]}..., acc={stage3_batch.batch['acc'][idx].item()}")
                        print(f"=========================================\n")
                    
                    # 注意：翻译奖励已经在第二阶段通过LLM Judge二值评估确定了，不再使用第三阶段的推理准确率来更新翻译奖励
                    # 第三阶段仍然执行，用于多语言推理训练
                    stage3_final_indices = []  # 只保留多语言推理准确率>0的数据
                    
                    # 构建stage2索引到stage3 uid的映射
                    # stage3_batch在repeat前为每个成功翻译生成了唯一的uid
                    # repeat后，每个uid对应n个推理结果
                    stage3_uids_list = list(stage3_sample_accs.keys())  # 所有唯一的uid
                    
                    if DEBUGGING_STS:
                        print(f"\n========== 阶段三数据处理（不再用于更新翻译奖励）==========")
                        print(f"  - 翻译奖励已在第二阶段通过LLM Judge二值评估确定")
                        print(f"  - 第三阶段仅用于多语言推理训练")
                        print(f"  - valid_indices数量: {len(valid_indices)}")
                        print(f"  - stage3唯一uid数量: {len(stage3_uids_list)}")
                        print(f"=========================================\n")
                    
                    # 为每个成功翻译的stage2索引分配对应的stage3 uid
                    # valid_indices[i] 对应 stage3_uids_list[i]
                    for i, stage2_idx in enumerate(valid_indices):
                        if i >= len(stage3_uids_list):
                            if DEBUGGING_STS:
                                print(f"警告: valid_indices索引{i}超出stage3_uids_list范围")
                            break
                            
                        stage3_uid = stage3_uids_list[i]
                        multilingual_acc = np.mean(stage3_sample_accs[stage3_uid])
                        
                        # 阶段三只保留多语言推理准确率>0的数据（所有n个推理结果都保留）
                        stage3_indices_to_add = np.where(stage3_batch.non_tensor_batch["uid"] == stage3_uid)[0].tolist()
                        if multilingual_acc > 0 and multilingual_acc < 1:
                            stage3_final_indices.extend(stage3_indices_to_add)
                        
                        if DEBUGGING_STS and i < 3:  # 前3次详细打印
                            same_uid_indices = np.where(stage3_batch.non_tensor_batch["uid"] == stage3_uid)[0].tolist()
                            print(f"  循环{i}: stage2_idx={stage2_idx}")
                            print(f"    -> stage3_uid={stage3_uid[:8]}..., 该uid的平均准确率: {multilingual_acc:.4f}")
                            print(f"    -> 翻译奖励已由LLM Judge确定，不再更新")
                    
                    if DEBUGGING_STS:
                        print(f"=========================================\n")
                    stage2_filtered_indices = []

                    # 获取stage2_batch中每个样本对应的原始stage2问题的uid
                    # 由于stage2_batch是通过repeat得到的，我们需要重建原始uid的映射
                    translation_sample_n = self.config.actor_rollout_ref.rollout.translation_sample_n

                    # 为stage2_batch中的每个样本分配原始问题id
                    # stage2_batch = stage2_batch.repeat(repeat_times=translation_sample_n, interleave=True)
                    # 所以索引 i 对应的原始问题索引是 i % (len(stage2_batch) // translation_sample_n)
                    original_question_count = len(stage2_batch) // translation_sample_n

                    # 按原始问题分组检查奖励
                    for orig_q_idx in range(original_question_count):
                        # 获取该原始问题的所有翻译尝试的索引
                        translation_indices = []
                        if translation_sample_n > 1:
                            # interleave=True的情况：索引分布为 0, 1, 2, ..., orig_q_idx, orig_q_idx+original_question_count, ...
                            for sample_idx in range(translation_sample_n):
                                idx = orig_q_idx + sample_idx * original_question_count
                                translation_indices.append(idx)
                        else:
                            translation_indices = [orig_q_idx]
                        
                        # 获取这些翻译的奖励值（每个翻译的总奖励）
                        rewards = []
                        for idx in translation_indices:
                            reward_sum = stage2_batch.batch['reward_tensor'][idx].sum().item()
                            rewards.append(reward_sum)
                        
                        # 检查是否所有翻译的奖励都相同
                        # 使用LLM Judge二值分数（0 or 1）
                        unique_rewards = set(rewards)
                        
                        # 如果奖励有区分度（既有0又有1），则保留这些翻译用于训练
                        if len(unique_rewards) > 1:
                            stage2_filtered_indices.extend(translation_indices)
                            if DEBUGGING_STS and orig_q_idx < 3:
                                print(f"原始问题{orig_q_idx}: 奖励{rewards}，有区分度，保留用于训练")
                        else:
                            if DEBUGGING_STS and orig_q_idx < 3:
                                print(f"原始问题{orig_q_idx}: 奖励{rewards}，无区分度（全{'0' if 0.0 in unique_rewards else '1'}），过滤掉")

                    # 更新stage2_final_indices为过滤后的索引
                    stage2_final_indices = stage2_filtered_indices
                    
                    metrics['sts/stage2_translation_reward'] = torch.mean(torch.sum(stage2_batch.batch['reward_tensor'], dim=-1)).item()
                    metrics['sts/stage3_multilingual_reward'] = torch.mean(torch.sum(stage3_batch.batch['multilingual_reward_tensor'], dim=-1)).item()
                    
                    sts_metrics["StS-Valid-Ratio/Stage2-final"] = np.round(len(stage2_final_indices) * 100 / len(stage2_batch), 2).item()
                    sts_metrics["StS-Valid-Ratio/Stage3-final"] = np.round(len(stage3_final_indices) * 100 / len(stage3_batch), 2).item()
                    
                    if DEBUGGING_STS:
                        # ===== Debug信息：验证最终结果 =====
                        print(f"\n========== 翻译奖励更新结果（使用LLM Judge二值评估）==========")
                        print(f"stage2_final_indices数量: {len(stage2_final_indices)} （收集所有翻译尝试）")
                        print(f"其中成功翻译: {len(valid_indices)} 个 ({len(valid_indices)*100/len(stage2_batch):.2f}%)")
                        print(f"失败翻译: {len(stage2_final_indices) - len(valid_indices)} 个 ({(len(stage2_final_indices) - len(valid_indices))*100/len(stage2_batch):.2f}%) [奖励保持0]")
                        print(f"stage3_final_indices数量: {len(stage3_final_indices)} (占stage3_batch的{len(stage3_final_indices)*100/len(stage3_batch):.2f}%)")
                        print(f"stage2_batch中奖励=1数量: {(stage2_batch.batch['reward_tensor'].sum(dim=1) == 1).sum().item()} [LLM Judge判定Correct]")
                        print(f"stage2_batch中奖励=0数量: {(stage2_batch.batch['reward_tensor'].sum(dim=1) == 0).sum().item()} [LLM Judge判定Incorrect或翻译失败]")
                        print(f"=========================================\n")
                    
                    ##### 组合最终训练数据 #####
                    # 阶段一：排除全对的英文回答
                    stage1_final_batch = batch.select_idxs(stage1_training_indices)
                    
                    # 阶段二：所有翻译尝试（包括失败的翻译），按照qt_training_ratio采样
                    if self.qt_training_ratio < 1.0:
                        stage2_final_indices = random.sample(stage2_final_indices, int(len(stage2_final_indices) * self.qt_training_ratio))
                    stage2_final_batch = stage2_batch.select_idxs(stage2_final_indices)
                    
                    # 阶段三：只保留准确率非0的多语言推理数据
                    stage3_final_batch = stage3_batch.select_idxs(stage3_final_indices)
                    # 获取环境变量，是否拒绝将一阶段数据加入训练
                    reject_stage1_data = os.getenv("REJECT_STAGE1_DATA")
                    if reject_stage1_data and reject_stage1_data.lower() in ["true", "1"]:
                        stage1_final_batch = []
                    # 获取环境变量，是否拒绝将二阶段数据加入训练
                    reject_stage2_data = os.getenv("REJECT_STAGE2_DATA")
                    if reject_stage2_data and reject_stage2_data.lower() in ["true", "1"]:
                        stage2_final_batch = []
                    # 获取环境变量，是否拒绝将三阶段数据加入训练
                    reject_stage3_data = os.getenv("REJECT_STAGE3_DATA")
                    if reject_stage3_data and reject_stage3_data.lower() in ["true", "1"]:
                        stage3_final_batch = []

                    # 合并所有阶段的数据
                    final_batches = []
                    if len(stage1_final_batch) > 0:
                        final_batches.append(stage1_final_batch)
                    if len(stage2_final_batch) > 0:
                        final_batches.append(stage2_final_batch)
                    if len(stage3_final_batch) > 0:
                        stage3_final_batch.batch['reward_tensor'] = stage3_final_batch.batch.pop('multilingual_reward_tensor')
                        final_batches.append(stage3_final_batch)
                    
                    if len(final_batches) == 0:
                        print("最终没有可用于训练的数据，跳过此iteration")
                        continue
                    
                    batch = DataProto.concat(final_batches)
                    
                    print(f">>> 最终训练数据组成: 阶段一: {len(stage1_final_batch)} | 阶段二: {len(stage2_final_batch)} | 阶段三: {len(stage3_final_batch)} | 总计: {len(batch)}")
                    print_sts_metrics = {k.replace('StS-Valid-Ratio/', ''): v for k, v in sts_metrics.items()}
                    print(f">>> StS数据有效比例: {print_sts_metrics}")
                    
                    # 添加数据统计
                    sts_metrics['StS-Valid-Ratio/Num-Stage1'] = len(stage1_final_batch)
                    sts_metrics['StS-Valid-Ratio/Num-Stage2'] = len(stage2_final_batch) 
                    sts_metrics['StS-Valid-Ratio/Num-Stage3'] = len(stage3_final_batch)
                    sts_metrics['StS-Valid-Ratio/Num-Total'] = len(batch)
                    metrics.update(sts_metrics)
                    
                    # 裁剪数据以保证能被world_size整除
                    ori_batch_length = len(batch)
                    remain_batch_index = [i for i in range(ori_batch_length - ori_batch_length % self.actor_rollout_wg.world_size)]
                    if len(remain_batch_index) == 0:
                        print(f"数据不足以被{self.actor_rollout_wg.world_size}整除，跳过此iteration")
                        continue
                    batch = batch.select_idxs(remain_batch_index)
                    print(f">>> 为保证能被world_size整除，数据从{ori_batch_length}裁剪到{len(batch)}")
                    
                    # 重新计算old_log_probs
                    with marked_timer("old_log_prob", timing_raw, color="blue"):
                        old_log_prob = self.actor_rollout_wg.compute_log_prob(batch)
                        entropys = old_log_prob.batch["entropys"]
                        response_masks = batch.batch["response_mask"]
                        loss_agg_mode = self.config.actor_rollout_ref.actor.loss_agg_mode
                        entropy_agg = agg_loss(loss_mat=entropys, loss_mask=response_masks, loss_agg_mode=loss_agg_mode)
                        old_log_prob_metrics = {"actor/entropy": entropy_agg.detach().item()}
                        metrics.update(old_log_prob_metrics)
                        old_log_prob.batch.pop("entropys")
                        batch = batch.union(old_log_prob)

                    if self.use_reference_policy:
                        # compute reference log_prob
                        with marked_timer("ref", timing_raw, color="olive"):
                            if not self.ref_in_actor:
                                ref_log_prob = self.ref_policy_wg.compute_ref_log_prob(batch)
                            else:
                                ref_log_prob = self.actor_rollout_wg.compute_ref_log_prob(batch)
                            batch = batch.union(ref_log_prob)

                    # compute values
                    if self.use_critic:
                        with marked_timer("values", timing_raw, color="cyan"):
                            values = self.critic_wg.compute_values(batch)
                            batch = batch.union(values)

                    with marked_timer("adv", timing_raw, color="brown"):
                        # 设置token_level_scores
                        if "reward_tensor" in batch.batch.keys():
                            reward_tensor = batch.batch['reward_tensor']
                            reward_extra_infos_dict = {}
                            batch.batch["token_level_scores"] = reward_tensor
                        else:
                            batch.batch["token_level_scores"] = reward_tensor

                        if reward_extra_infos_dict:
                            batch.non_tensor_batch.update({k: np.array(v) for k, v in reward_extra_infos_dict.items()})

                        # compute rewards. apply_kl_penalty if available
                        if self.config.algorithm.use_kl_in_reward:
                            batch, kl_metrics = apply_kl_penalty(batch, kl_ctrl=self.kl_ctrl_in_reward, kl_penalty=self.config.algorithm.kl_penalty)
                            metrics.update(kl_metrics)
                        else:
                            batch.batch["token_level_rewards"] = batch.batch["token_level_scores"]

                        # compute advantages
                        norm_adv_by_std_in_grpo = self.config.algorithm.get("norm_adv_by_std_in_grpo", True)

                        batch = compute_advantage(
                            batch,
                            adv_estimator=self.config.algorithm.adv_estimator,
                            gamma=self.config.algorithm.gamma,
                            lam=self.config.algorithm.lam,
                            num_repeat=self.config.actor_rollout_ref.rollout.n,
                            norm_adv_by_std_in_grpo=norm_adv_by_std_in_grpo,
                            multi_turn=self.config.actor_rollout_ref.rollout.multi_turn.enable,
                            config=self.config.algorithm,
                        )

                    # update critic
                    if self.use_critic:
                        with marked_timer("update_critic", timing_raw, color="pink"):
                            critic_output = self.critic_wg.update_critic(batch)
                        critic_output_metrics = reduce_metrics(critic_output.meta_info["metrics"])
                        metrics.update(critic_output_metrics)

                    # implement critic warmup
                    if self.config.trainer.critic_warmup <= self.global_steps:
                        # update actor
                        with marked_timer("update_actor", timing_raw, color="red"):
                            batch.meta_info["multi_turn"] = self.config.actor_rollout_ref.rollout.multi_turn.enable
                            actor_output = self.actor_rollout_wg.update_actor(batch)
                        actor_output_metrics = reduce_metrics(actor_output.meta_info["metrics"])
                        metrics.update(actor_output_metrics)

                    # Log rollout generations if enabled
                    rollout_data_dir = self.config.trainer.get("rollout_data_dir", None)
                    if rollout_data_dir:
                        with marked_timer("dump_rollout_generations", timing_raw, color="green"):
                            inputs = self.tokenizer.batch_decode(batch.batch["prompts"], skip_special_tokens=True)
                            outputs = self.tokenizer.batch_decode(batch.batch["responses"], skip_special_tokens=True)
                            scores = batch.batch["token_level_scores"].sum(-1).cpu().tolist()
                            self._dump_generations(
                                inputs=inputs,
                                outputs=outputs,
                                scores=scores,
                                reward_extra_infos_dict=reward_extra_infos_dict,
                                dump_path=rollout_data_dir,
                            )

                    # validate
                    if self.val_reward_fn is not None and self.config.trainer.test_freq > 0 and (is_last_step or self.global_steps % self.config.trainer.test_freq == 0):
                        with marked_timer("testing", timing_raw, color="green"):
                            val_metrics: dict = self._validate()
                            if is_last_step:
                                last_val_metrics = val_metrics
                        metrics.update(val_metrics)

                    esi_close_to_expiration = should_save_ckpt_esi(max_steps_duration=self.max_steps_duration, redundant_time=self.config.trainer.esi_redundant_time)
                    if self.config.trainer.save_freq > 0 and (is_last_step or self.global_steps % self.config.trainer.save_freq == 0 or esi_close_to_expiration):
                        if esi_close_to_expiration:
                            print("Force saving checkpoint: ESI instance expiration approaching.")
                        with marked_timer("save_checkpoint", timing_raw, color="green"):
                            self._save_checkpoint()

                steps_duration = timing_raw["step"]
                self.max_steps_duration = max(self.max_steps_duration, steps_duration)
                # training metrics
                metrics.update(
                    {
                        "training/global_step": self.global_steps,
                        "training/epoch": epoch,
                    }
                )
                # collect metrics
                metrics.update(compute_data_metrics(batch=batch, use_critic=self.use_critic))
                metrics.update(compute_timing_metrics(batch=batch, timing_raw=timing_raw))
                # TODO: implement actual tflpo and theoretical tflpo
                n_gpus = self.resource_pool_manager.get_n_gpus()
                metrics.update(compute_throughout_metrics(batch=batch, timing_raw=timing_raw, n_gpus=n_gpus))

                # TODO: make a canonical logger that supports various backend
                logger.log(data=metrics, step=self.global_steps)

                progress_bar.update(1)
                self.global_steps += 1

                if do_profile:
                    self.actor_rollout_wg.stop_profile()
                    if self.use_reference_policy:
                        self.ref_policy_wg.stop_profile()
                    if self.use_critic:
                        self.critic_wg.stop_profile()
                    if self.use_rm:
                        self.rm_wg.stop_profile()

                if is_last_step:
                    pprint(f"Final validation metrics: {last_val_metrics}")
                    progress_bar.close()
                    return

