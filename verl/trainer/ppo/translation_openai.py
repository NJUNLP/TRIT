import asyncio
import time
from typing import List, Optional, Dict, Any
import logging
# 引入 OpenAI SDK 的异步客户端及相关异常
from openai import AsyncOpenAI, APIError, RateLimitError, APITimeoutError, APIConnectionError, APIStatusError
from contextlib import AsyncExitStack

logger = logging.getLogger(__name__)

LANGUAGE_MAP = {
    "ZH": "Chinese",
    "ES": "Spanish",
    "FR": "French",
    "DE": "German",
    "JA": "Japanese",
    "KO": "Korean",
    "RU": "Russian",
    "AR": "Arabic"
}

class AsyncTranslationAPI:
    """异步翻译API封装类（基于 AsyncOpenAI SDK）"""
    
    def __init__(self, 
                 api_base_url: str, 
                 api_key: str, 
                 model_name: str,
                 max_concurrency: int = 10,
                 timeout: int = 300):
        """
        初始化异步翻译API
        """
        self.api_base_url = api_base_url.rstrip('/')
        self.api_key = api_key
        self.model_name = model_name
        self.max_concurrency = max_concurrency
        self.timeout = float(timeout) # 转换为浮点数以匹配 SDK 参数
        self.semaphore = asyncio.Semaphore(max_concurrency)
        
        # 实例化 AsyncOpenAI 客户端
        self.client = AsyncOpenAI(
            api_key=self.api_key,
            base_url=self.api_base_url,
            max_retries=0,        # 由我们自己的 translate_single 函数处理重试
            timeout=self.timeout  # 设置全局超时
        )
        
    async def __aenter__(self):
        """异步上下文管理器入口（保留结构，但客户端已在 __init__ 中创建）"""
        # AsyncOpenAI 客户端通常不需要在 __aenter__ 中做特殊处理，但保留结构
        return self
        
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """异步上下文管理器退出：关闭客户端会话"""
        if self.client:
            await self.client.close()
    
    async def translate_single(self, 
                               text: str, 
                               item_id: int,
                               max_retries: int = 3) -> Dict[str, Any]:
        """
        异步翻译单个文本（带重试机制，使用 AsyncOpenAI）
        """
        async with self.semaphore:
            translation_prompt = text
            
            for attempt in range(max_retries + 1):
                try:
                    # 使用 AsyncOpenAI 的 chat.completions.create
                    response = await self.client.chat.completions.create(
                        model=self.model_name,
                        messages=[
                            {"role": "user", "content": translation_prompt}
                        ],
                        temperature=0.6,
                        max_tokens=2048,
                        n=1,
                        stream=False,
                        extra_body={
                            "cache": {
                                "no-cache": True
                            }
                        }
                        # AsyncOpenAI 自动处理标准 Header、JSON 格式和连接
                    )

                    translation = response.choices[0].message.content.strip()
                    
                    if translation:
                        logger.debug(f"ID {item_id} 翻译成功 (尝试 {attempt + 1} 次)")
                        return {
                            'id': item_id,
                            'original_text': text,
                            'translation': translation,
                            'status': 'success',
                            'attempts': attempt + 1,
                            'timestamp': time.time()
                        }
                    else:
                        raise ValueError("翻译结果为空") # 结果为空，作为可重试异常处理

                except RateLimitError as e:
                    logger.warning(f"ID {item_id} 触发限流 (尝试 {attempt + 1})")
                    if attempt == max_retries: break
                    await asyncio.sleep(5 + 2 ** attempt) # 限流时等待更久
                    
                except APITimeoutError:
                    logger.warning(f"ID {item_id} 请求超时 (尝试 {attempt + 1})")
                    if attempt == max_retries: break
                    await asyncio.sleep(2 ** attempt)
                
                except (APIStatusError, APIError, APIConnectionError, ValueError) as e:
                    # 捕获所有 API 状态、连接和自定义的 ValueError (结果为空)
                    error_msg = f"API Error: {str(e)}"
                    if hasattr(e, 'status_code'):
                         error_msg = f"HTTP {e.status_code}: {str(e)}"
                         
                    logger.warning(
                        f"API请求失败 (ID: {item_id}, 尝试 {attempt + 1}/{max_retries + 1}) - {error_msg}"
                    )
                    
                    if attempt == max_retries: 
                        return {
                            'id': item_id,
                            'original_text': text,
                            'translation': None,
                            'status': 'failed',
                            'error': error_msg,
                            'attempts': attempt + 1,
                            'timestamp': time.time()
                        }
                    
                    # 指数退避重试
                    await asyncio.sleep(2 ** attempt)
                            
                except Exception as e:
                    logger.error(f"未知错误 (ID: {item_id}, 尝试 {attempt + 1}/{max_retries + 1}) - Error: {str(e)}")
                    if attempt == max_retries:
                        return {
                            'id': item_id,
                            'original_text': text,
                            'translation': None,
                            'status': 'failed',
                            'error': str(e),
                            'attempts': attempt + 1,
                            'timestamp': time.time()
                        }
                    await asyncio.sleep(2 ** attempt)

    
    async def translate_batch(self, 
                             texts: List[str]) -> List[Optional[str]]:
        """
        异步批量翻译文本
        """
        logger.info(f"开始翻译 {len(texts)} 个文本，并发度: {self.max_concurrency}")
        
        tasks = [
            self.translate_single(text, idx)
            for idx, text in enumerate(texts)
        ]
        
        # 并发执行所有任务
        # return_exceptions=True 确保不会因单个失败而中断所有任务
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # 处理结果
        translations = [None] * len(texts)
        success_count = 0
        
        for result in results:
            if isinstance(result, Exception):
                # 任务直接抛出未捕获的异常
                logger.error(f"翻译任务异常: {type(result).__name__}: {result}")
                continue
            
            # 💥 关键修复：检查 result 是否为 None
            if result is None:
                # 理论上 translate_single 应该返回一个字典，如果返回 None，说明有逻辑错误
                logger.error(f"翻译任务返回了 NoneType 结果，跳过处理。")
                continue
                
            idx = result.get('id')
            if idx is None:
                # 如果返回了字典但没有 'id' 键，也跳过
                logger.error(f"翻译任务返回结果中缺少 'id' 键: {result}")
                continue
                
            if result['status'] == 'success':
                translations[idx] = result['translation']
                success_count += 1
            else:
                logger.warning(f"ID {idx} 翻译失败: {result.get('error', 'Unknown error')}")
        
        success_rate = success_count / len(texts) * 100 if texts else 0
        logger.info(f"翻译完成: 成功 {success_count}/{len(texts)} ({success_rate:.1f}%)")
        
        return translations


def run_async_translation(texts: List[str], 
                         api_base_url: str,
                         api_key: str,
                         model_name: str,
                         max_concurrency: int = 10) -> List[Optional[str]]:
    """
    同步包装器：在同步代码中调用异步翻译
    """
    async def _async_wrapper():
        async with AsyncTranslationAPI(
            api_base_url=api_base_url,
            api_key=api_key,
            model_name=model_name,
            max_concurrency=max_concurrency
        ) as translator:
            return await translator.translate_batch(texts=texts)
    
    # 在新的事件循环中运行（兼容各种运行环境）
    try:
        loop = asyncio.get_event_loop()
        if loop.is_running():
            # 如果已有运行的事件循环，创建新线程运行
            import concurrent.futures
            with concurrent.futures.ThreadPoolExecutor() as executor:
                # 使用 asyncio.run 在单独的线程中安全地启动新的事件循环
                future = executor.submit(lambda: asyncio.run(_async_wrapper()))
                return future.result()
        else:
            return loop.run_until_complete(_async_wrapper())
    except RuntimeError:
        # 没有事件循环，直接创建新的
        return asyncio.run(_async_wrapper())