
import asyncio
import aiohttp
import time
from typing import List, Optional, Dict, Any
import logging

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
    """异步翻译API封装类（基于你的异步设计）"""
    
    def __init__(self, 
                 api_base_url: str, 
                 api_key: str, 
                 model_name: str,
                 max_concurrency: int = 10,
                 timeout: int = 300):
        """
        初始化异步翻译API
        
        Args:
            api_base_url: API基础URL
            api_key: API密钥
            model_name: 使用的模型名称
            max_concurrency: 最大并发数
            timeout: 请求超时时间（秒）
        """
        self.api_base_url = api_base_url.rstrip('/')
        self.api_key = api_key
        self.model_name = model_name
        self.max_concurrency = max_concurrency
        self.timeout = timeout
        self.semaphore = asyncio.Semaphore(max_concurrency)
        self.session = None
        
    async def __aenter__(self):
        """异步上下文管理器入口"""
        connector = aiohttp.TCPConnector(limit=self.max_concurrency * 2)
        timeout = aiohttp.ClientTimeout(total=self.timeout)
        self.session = aiohttp.ClientSession(
            connector=connector,
            timeout=timeout,
            headers={
                'Authorization': f'Bearer {self.api_key}',
                'Content-Type': 'application/json'
            }
        )
        return self
        
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """异步上下文管理器退出"""
        if self.session:
            await self.session.close()
    
    async def translate_single(self, 
                               text: str, 
                               item_id: int,
                               max_retries: int = 3) -> Dict[str, Any]:
        """
        异步翻译单个文本（带重试机制）
        
        Args:
            text: 完整的prompt文本（不需要额外模板处理）
            item_id: 项目ID（用于日志）
            max_retries: 最大重试次数
            
        Returns:
            翻译结果字典
        """
        async with self.semaphore:
            # 直接使用传入的文本作为完整prompt
            translation_prompt = text
            
            for attempt in range(max_retries + 1):
                try:
                    request_data = {
                        "model": self.model_name,
                        "messages": [
                            {"role": "user", "content": translation_prompt}
                        ],
                        "temperature": 0.6,  # 翻译任务使用较低温度
                        "max_tokens": 2048,
                        "n": 1,
                        "stream": False
                    }
                    
                    async with self.session.post(
                        f"{self.api_base_url}/chat/completions",
                        json=request_data
                    ) as response:
                        if response.status == 200:
                            result = await response.json(content_type=None)
                            translation = result['choices'][0]['message']['content'].strip()
                            
                            # 直接返回原始翻译结果，不进行后处理
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
                                logger.warning(f"ID {item_id} 翻译结果为空 (尝试 {attempt + 1})")
                                if attempt == max_retries:
                                    return {
                                        'id': item_id,
                                        'original_text': text,
                                        'translation': None,
                                        'status': 'failed',
                                        'error': '翻译结果为空',
                                        'raw_output': translation,
                                        'attempts': attempt + 1,
                                        'timestamp': time.time()
                                    }
                        else:
                            error_text = await response.text()
                            logger.warning(
                                f"API请求失败 (ID: {item_id}, 尝试 {attempt + 1}/{max_retries + 1}) - "
                                f"Status: {response.status}, Error: {error_text}"
                            )
                            
                            if attempt == max_retries:
                                return {
                                    'id': item_id,
                                    'original_text': text,
                                    'translation': None,
                                    'status': 'failed',
                                    'error': f'HTTP {response.status}: {error_text}',
                                    'attempts': attempt + 1,
                                    'timestamp': time.time()
                                }
                        
                        # 指数退避重试
                        if attempt < max_retries:
                            await asyncio.sleep(2 ** attempt)
                            
                except asyncio.TimeoutError:
                    logger.warning(f"请求超时 (ID: {item_id}, 尝试 {attempt + 1}/{max_retries + 1})")
                    if attempt == max_retries:
                        return {
                            'id': item_id,
                            'original_text': text,
                            'translation': None,
                            'status': 'failed',
                            'error': '请求超时',
                            'attempts': attempt + 1,
                            'timestamp': time.time()
                        }
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
        
        Args:
            texts: 完整的prompt文本列表（不需要额外模板处理）
            
        Returns:
            翻译结果列表，失败的位置为None
        """
        logger.info(f"开始翻译 {len(texts)} 个文本，并发度: {self.max_concurrency}")
        
        # 创建任务列表
        tasks = [
            self.translate_single(text, idx)
            for idx, text in enumerate(texts)
        ]
        
        # 并发执行所有任务
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # 处理结果
        translations = [None] * len(texts)
        success_count = 0
        
        for result in results:
            if isinstance(result, Exception):
                logger.error(f"翻译任务异常: {result}")
                continue
                
            idx = result['id']
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
    
    Args:
        texts: 完整的prompt文本列表（不需要额外模板处理）
        api_base_url: API基础URL
        api_key: API密钥
        model_name: 模型名称
        max_concurrency: 最大并发数
        
    Returns:
        翻译结果列表
    """
    async def _async_wrapper():
        async with AsyncTranslationAPI(
            api_base_url=api_base_url,
            api_key=api_key,
            model_name=model_name,
            max_concurrency=max_concurrency
        ) as translator:
            return await translator.translate_batch(texts=texts)
    
    # 在新的事件循环中运行（避免与Ray的事件循环冲突）
    try:
        loop = asyncio.get_event_loop()
        if loop.is_running():
            # 如果已有运行的事件循环，创建新线程运行
            import concurrent.futures
            with concurrent.futures.ThreadPoolExecutor() as executor:
                # ✅ 用lambda包装，防止“未await”警告
                future = executor.submit(lambda: asyncio.run(_async_wrapper()))
                return future.result()
        else:
            return loop.run_until_complete(_async_wrapper())
    except RuntimeError:
        # 没有事件循环，直接创建新的
        return asyncio.run(_async_wrapper())
