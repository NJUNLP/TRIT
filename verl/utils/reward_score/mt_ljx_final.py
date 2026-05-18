"""
COMET 客户端 - 统一版本
融合高性能特性和业务逻辑
支持翻译评分和通用COMET调用
"""

import re
import asyncio
import aiohttp
from typing import List, Optional, Dict, Any
import logging
from langdetect import detect_langs, DetectorFactory, LangDetectException

# ==================== 配置日志 ====================
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ==================== 语言配置 ====================
LANGUAGE_MAP = {
    "ZH": "Chinese",
    "EN": "English",
    "JA": "Japanese",
    "KO": "Korean",
    "FR": "French",
    "DE": "German",
    "ES": "Spanish",
    "IT": "Italian",
    "PT": "Portuguese",
    "RU": "Russian",
    "AR": "Arabic",
    "BN": "Bengali",
    "TH": "Thai"
}

LANGUAGE_REASONING_MAP = {
    "EN": "\nPlease reason step by step, and put your final answer within \\boxed{}.",
    "DE": "\nBitte begründen Sie Schritt für Schritt, und fassen Sie Ihre endgültige Antwort in \\boxed{}.",
    "FR": "\nVeuillez raisonner étape par étape, et placez votre réponse finale dans \\boxed{}.",
    "BN": "\nদয়া করে ধাপে ধাপে যুক্তি দিন এবং আপনার চূড়ান্ত উত্তরটি \\boxed{} এর মধ্যে লিখুন।",
    "ZH": "\n请逐步推理，并将最终答案放在 \\boxed{} 中。",
    "JA": "\n段階的に推理し、最終的な答えを\\boxed{}の中に入れてください。",
    "KO": "\n단계별로 논리적으로 설명해 주시고, 최종 답변을 \\boxed{} 안에 넣어 주세요.",
    "TH": "\nโปรดให้เหตุผลทีละขั้นตอน และใส่คำตอบสุดท้ายไว้ใน \\boxed{}.",
    "PT": "\nPor favor, raciocine passo a passo e coloque sua resposta final dentro de \\boxed{}."
}

LANGUAGE_START_PREFIX_MAP = {
    "ZH": "<think>\n好的",
    "FR": "<think>\nD'accord",
    "BN": "<think>\nঠিক আছে",
    "EN": "<think>\nOkay",
    "JA": "<think>\nさて",
    "KO": "<think>\n네",
    "PT": "<think>\nCerto",
    "TH": "<think>\nโอเค"
}

LANGUAGE_START_PREFIX_DISTILL_MAP = {
    "ZH": "好的",
    "FR": "D'accord",
    "BN": "ঠিক আছে",
    "EN": "Okay",
    "JA": "<think>\nさて",
}

LANGUAGE_CODE_MAP = {
    "ZH": ["zh-cn", "zh"],
    "EN": ["en"],
    "JA": ["ja"],
    "FR": ["fr"],
    "DE": ["de"],
    "ES": ["es"],
    "RU": ["ru"],
    "BN": ["bn"],
    "SW": ["sw"],
    "TE": ["te"],
    "TH": ["th"],
    "PT": ["pt"],
    "KO": ["ko"]
}

DetectorFactory.seed = 0

# ==================== 客户端配置 ====================
class CometClientConfig:
    """COMET 客户端配置"""
    # Nginx 地址
    NGINX_HOST = "10.238.18.19"
    NGINX_PORT = 9000
    NGINX_URL = f"http://{NGINX_HOST}:{NGINX_PORT}"
    
    # 并发控制
    MAX_CONCURRENT = 64  # 最大并发数
    TIMEOUT = 120  # 单个请求超时（秒）
    
    # 连接池
    POOL_SIZE = 100  # 连接池大小
    POOL_PER_HOST = 50  # 每个主机的连接数
    
    # 重试策略
    MAX_RETRIES = 3
    RETRY_DELAY = 1  # 重试延迟（秒）
    
    # 批处理
    AUTO_BATCH_SIZE = 32  # 自动批处理大小

config = CometClientConfig()

# ==================== 全局资源管理 ====================
class GlobalResources:
    """全局资源单例"""
    _instance = None
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance.session = None
            cls._instance.semaphore = None
            cls._instance.stats = {
                'total_requests': 0,
                'successful_requests': 0,
                'failed_requests': 0,
                'total_retries': 0
            }
        return cls._instance
    
    async def get_session(self):
        """获取或创建 aiohttp Session"""
        if self.session is None or self.session.closed:
            timeout = aiohttp.ClientTimeout(total=config.TIMEOUT)
            connector = aiohttp.TCPConnector(
                limit=config.POOL_SIZE,
                limit_per_host=config.POOL_PER_HOST,
                ttl_dns_cache=300,
                enable_cleanup_closed=True
            )
            self.session = aiohttp.ClientSession(
                timeout=timeout,
                connector=connector
            )
        return self.session
    
    def get_semaphore(self):
        """获取信号量"""
        if self.semaphore is None:
            self.semaphore = asyncio.Semaphore(config.MAX_CONCURRENT)
        return self.semaphore
    
    async def cleanup(self):
        """清理资源"""
        if self.session and not self.session.closed:
            await self.session.close()
            self.session = None

resources = GlobalResources()

# ==================== 文本处理工具 ====================
def clean_latex_code(text: str) -> str:
    """清理文本中的LaTeX代码"""
    text = re.sub(r'\$\$.*?\$\$', ' ', text, flags=re.DOTALL)
    text = re.sub(r'\\\[.*?\\\]', ' ', text, flags=re.DOTALL)
    text = re.sub(r'\\\(.*?\\\)', ' ', text, flags=re.DOTALL)
    text = re.sub(r'\$[^$\n]{1,100}\$', ' ', text)
    text = re.sub(r'\\[a-zA-Z]+\{[^}]*\}', ' ', text)
    text = re.sub(r'\\[a-zA-Z]+\b', ' ', text)
    text = re.sub(r'\s+', ' ', text)
    return text.strip()

def extract_last_translation(text: str, language_code: str) -> Optional[str]:
    """从模型返回的文本中提取最后一个 <TRANSLATION> 标签内的内容"""
    if "<TRANSLATION>" not in text:
        return None

    text_segment = "<TRANSLATION>" + text.split("<TRANSLATION>")[-1]
    matches = re.findall(r'<TRANSLATION>(.*?)</TRANSLATION>', text_segment, re.DOTALL)

    if not matches:
        return None

    translation = matches[-1].strip()
    translation_remove_latex_code = clean_latex_code(translation)
    
    try:
        detected_langs = detect_langs(translation_remove_latex_code)
        top_lang = detected_langs[0].lang.lower()
    except LangDetectException:
        return None
    
    target_langs = [l.lower() for l in LANGUAGE_CODE_MAP.get(language_code.upper(), [])]

    if top_lang in target_langs:
        if len(translation.split("\n\n")) > 2:
            return None
        return translation
    else:
        return None

# ==================== 核心 COMET 调用函数 ====================
async def call_comet_single(
    src: str,
    mt: str,
    ref: str,
    max_retries: Optional[int] = None
) -> float:
    """
    调用 COMET 服务（单个请求）
    
    Args:
        src: 源文本
        mt: 机器翻译
        ref: 参考翻译
        max_retries: 最大重试次数
    
    Returns:
        COMET 分数 (0.0-1.0)
    """
    if max_retries is None:
        max_retries = config.MAX_RETRIES
    
    semaphore = resources.get_semaphore()
    session = await resources.get_session()
    
    url = f"{config.NGINX_URL}/comet_score"
    payload = {"src": src, "mt": mt, "ref": ref}
    
    async with semaphore:
        for attempt in range(max_retries):
            try:
                resources.stats['total_requests'] += 1
                
                async with session.post(url, json=payload) as resp:
                    resp.raise_for_status()
                    result = await resp.json()
                    score = result.get("score", 0.0)
                    
                    resources.stats['successful_requests'] += 1
                    return score
                    
            except asyncio.TimeoutError:
                logger.warning(f"COMET timeout (attempt {attempt + 1}/{max_retries})")
                resources.stats['total_retries'] += 1
                if attempt < max_retries - 1:
                    await asyncio.sleep(config.RETRY_DELAY * (attempt + 1))
                    
            except aiohttp.ClientError as e:
                logger.warning(f"COMET client error: {e} (attempt {attempt + 1}/{max_retries})")
                resources.stats['total_retries'] += 1
                if attempt < max_retries - 1:
                    await asyncio.sleep(config.RETRY_DELAY)
                    
            except Exception as e:
                logger.error(f"COMET unexpected error: {e}")
                break
        
        resources.stats['failed_requests'] += 1
        logger.error("All COMET retries failed")
        return 0.0

async def call_comet_batch(
    requests: List[Dict[str, str]],
    max_retries: Optional[int] = None
) -> List[float]:
    """
    调用 COMET 服务（批量请求）
    
    Args:
        requests: 请求列表，每个元素为 {"src": str, "mt": str, "ref": str}
        max_retries: 最大重试次数
    
    Returns:
        COMET 分数列表
    """
    if not requests:
        return []
    
    if max_retries is None:
        max_retries = config.MAX_RETRIES
    
    semaphore = resources.get_semaphore()
    session = await resources.get_session()
    
    url = f"{config.NGINX_URL}/comet_score_batch"
    
    async with semaphore:
        for attempt in range(max_retries):
            try:
                resources.stats['total_requests'] += len(requests)
                
                async with session.post(url, json=requests) as resp:
                    resp.raise_for_status()
                    result = await resp.json()
                    scores = result.get("scores", [0.0] * len(requests))
                    
                    resources.stats['successful_requests'] += len(requests)
                    return scores
                    
            except asyncio.TimeoutError:
                logger.warning(f"Batch timeout (attempt {attempt + 1}/{max_retries})")
                resources.stats['total_retries'] += 1
                if attempt < max_retries - 1:
                    await asyncio.sleep(config.RETRY_DELAY * (attempt + 1))
                    
            except aiohttp.ClientError as e:
                logger.warning(f"Batch client error: {e}")
                resources.stats['total_retries'] += 1
                if attempt < max_retries - 1:
                    await asyncio.sleep(config.RETRY_DELAY)
                    
            except Exception as e:
                logger.error(f"Batch unexpected error: {e}")
                break
        
        resources.stats['failed_requests'] += len(requests)
        return [0.0] * len(requests)

async def call_comet_concurrent(
    requests: List[Dict[str, str]],
    batch_size: Optional[int] = None
) -> List[float]:
    """
    并发批量调用 COMET
    自动分批并并发执行
    
    Args:
        requests: 请求列表
        batch_size: 批大小
    
    Returns:
        COMET 分数列表
    """
    if not requests:
        return []
    
    if batch_size is None:
        batch_size = config.AUTO_BATCH_SIZE
    
    # 分批
    batches = [
        requests[i:i + batch_size]
        for i in range(0, len(requests), batch_size)
    ]
    
    # 并发执行
    tasks = [call_comet_batch(batch) for batch in batches]
    results = await asyncio.gather(*tasks)
    
    # 合并结果
    scores = []
    for batch_scores in results:
        scores.extend(batch_scores)
    
    return scores

# ==================== 同步包装器 ====================
def get_or_create_event_loop():
    """
    获取或创建事件循环
    解决在多线程环境（如 Ray）中的事件循环问题
    """
    try:
        loop = asyncio.get_event_loop()
        if loop.is_closed():
            raise RuntimeError("Event loop is closed")
        return loop
    except RuntimeError:
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        return loop

def call_comet_sync(src: str, mt: str, ref: str) -> float:
    """
    同步调用 COMET（单个请求）
    
    Args:
        src: 源文本
        mt: 机器翻译
        ref: 参考翻译
    
    Returns:
        COMET 分数
    """
    loop = get_or_create_event_loop()
    return loop.run_until_complete(call_comet_single(src, mt, ref))

def call_comet_batch_sync(requests: List[Dict[str, str]]) -> List[float]:
    """
    同步调用 COMET（批量请求）
    
    Args:
        requests: 请求列表
    
    Returns:
        COMET 分数列表
    """
    loop = get_or_create_event_loop()
    return loop.run_until_complete(call_comet_concurrent(requests))

# ==================== 业务逻辑函数 ====================
def compute_score(
    solution_str: str,
    ground_truth: str,
    method: str = "strict",
    format_score: float = 0.1,
    score: float = 1.0
) -> Dict[str, Any]:
    """
    评分函数 - 用于翻译任务
    
    Args:
        solution_str: 模型生成的解答文本
        ground_truth: 标准答案，格式为 "language_code<translation>target_translation<translation>origin_question"
        method: 提取方法（暂未使用）
        format_score: 格式分数
        score: 基础分数
    
    Returns:
        dict: 包含各项评分指标的字典
            - score: 总分
            - acc: 准确率
            - format_correctness: 格式正确性
            - accuracy_reward: 准确性奖励
            - language_reward: 语言奖励
    """
    
    # 检查 </think> 是否正确
    if "</think>" not in solution_str or solution_str.count("</think>") > 1:
        return {
            'score': -1,
            'acc': 0,
            'format_correctness': 0,
            'accuracy_reward': 0,
            'language_reward': 0
        }

    # 解析 ground_truth
    try:
        parts = ground_truth.split("<translation>")
        if len(parts) != 3:
            raise ValueError(f"Invalid format, expected 3 parts, got {len(parts)}")
        
        language_code = parts[0]
        target_translation = parts[1]
        origin_question = parts[2]
        
    except Exception as e:
        logger.error(f"Invalid ground_truth: {e}")
        return {
            'score': -1,
            'acc': 0,
            'format_correctness': 0,
            'accuracy_reward': 0,
            'language_reward': 0
        }

    # 提取翻译
    translation = extract_last_translation(solution_str, language_code)
    if translation is None:
        return {
            'score': -1,
            'acc': 0,
            'format_correctness': 1,
            'accuracy_reward': 0,
            'language_reward': 0
        }

    # 调用 COMET 服务（通过 Nginx 负载均衡）
    comet_score = call_comet_sync(origin_question, translation, target_translation)

    return {
        'score': comet_score,
        'acc': comet_score,
        'format_correctness': 1,
        'accuracy_reward': comet_score,
        'language_reward': 1
    }

# ==================== 健康检查 ====================
async def check_health() -> Dict[str, Any]:
    """检查服务健康状态"""
    session = await resources.get_session()
    url = f"{config.NGINX_URL}/health"
    
    try:
        async with session.get(url, timeout=aiohttp.ClientTimeout(total=5)) as resp:
            resp.raise_for_status()
            return await resp.json()
    except Exception as e:
        logger.error(f"Health check failed: {e}")
        return {"status": "error", "message": str(e)}

def check_health_sync() -> Dict[str, Any]:
    """同步健康检查"""
    loop = get_or_create_event_loop()
    return loop.run_until_complete(check_health())

# ==================== 统计信息 ====================
def get_client_stats() -> Dict[str, int]:
    """获取客户端统计信息"""
    return resources.stats.copy()

def reset_client_stats():
    """重置客户端统计信息"""
    resources.stats = {
        'total_requests': 0,
        'successful_requests': 0,
        'failed_requests': 0,
        'total_retries': 0
    }

# ==================== 资源清理 ====================
async def cleanup_resources():
    """清理全局资源"""
    await resources.cleanup()

# 注册清理函数
import atexit
atexit.register(lambda: asyncio.run(cleanup_resources()))

# ==================== 使用示例 ====================
if __name__ == "__main__":
    # 示例 1: 单个同步调用
    print("示例 1: 单个同步调用")
    score = call_comet_sync(
        src="Hello world",
        mt="你好世界",
        ref="你好，世界"
    )
    print(f"Score: {score}\n")
    
    # 示例 2: 批量同步调用
    print("示例 2: 批量同步调用")
    requests = [
        {"src": "Hello", "mt": "你好", "ref": "您好"},
        {"src": "Goodbye", "mt": "再见", "ref": "再见"}
    ]
    scores = call_comet_batch_sync(requests)
    print(f"Scores: {scores}\n")
    
    # 示例 3: 异步调用
    async def async_example():
        print("示例 3: 异步调用")
        score = await call_comet_single(
            src="How are you?",
            mt="你好吗？",
            ref="你好吗？"
        )
        print(f"Async Score: {score}\n")
        
        # 健康检查
        health = await check_health()
        print(f"Health: {health}\n")
        
        # 统计信息
        stats = get_client_stats()
        print(f"Stats: {stats}")
    
    asyncio.run(async_example())