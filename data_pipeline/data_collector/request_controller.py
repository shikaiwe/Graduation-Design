"""
请求频率控制器
用于控制API请求频率，避免因请求过于频繁而被拒绝
"""

import time
import logging
import random
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Callable, Any, Optional, List, Dict
from dataclasses import dataclass
from enum import Enum
from queue import Queue, Empty

logger = logging.getLogger(__name__)

class RequestErrorType(Enum):
    """请求错误类型枚举"""
    CONNECTION_ERROR = "connection_error"
    RATE_LIMIT_ERROR = "rate_limit_error"
    SERVER_ERROR = "server_error"
    UNKNOWN_ERROR = "unknown_error"

@dataclass
class RequestConfig:
    """请求配置类"""
    base_delay: float = 1.0  # 基础延迟（秒）
    max_delay: float = 60.0  # 最大延迟（秒）
    max_retries: int = 5     # 最大重试次数
    backoff_factor: float = 2.0  # 退避因子
    jitter: float = 0.1      # 随机抖动因子
    batch_size: int = 50    # 批次大小
    requests_per_minute: int = 30  # 每分钟最大请求数

class RateLimiter:
    """请求频率限制器"""
    
    def __init__(self, requests_per_minute: int = 30):
        """
        初始化频率限制器
        
        Args:
            requests_per_minute: 每分钟最大请求数
        """
        self.requests_per_minute = requests_per_minute
        self.min_interval = 60.0 / requests_per_minute  # 最小请求间隔（秒）
        self.last_request_time = 0
        self.error_count = 0
        self.success_count = 0
        
    def wait_if_needed(self):
        """如果需要等待，则进行等待"""
        current_time = time.time()
        time_since_last = current_time - self.last_request_time
        
        # 动态调整等待时间，基于错误率
        dynamic_interval = self.min_interval * (1 + self.error_count / max(1, self.success_count))
        
        if time_since_last < dynamic_interval:
            sleep_time = dynamic_interval - time_since_last
            logger.debug(f"频率控制: 等待 {sleep_time:.2f} 秒")
            time.sleep(sleep_time)
        
        self.last_request_time = time.time()
    
    def record_success(self):
        """记录成功请求"""
        self.success_count += 1
        # 成功时逐渐减少错误计数
        if self.error_count > 0:
            self.error_count = max(0, self.error_count - 1)
    
    def record_error(self):
        """记录错误请求"""
        self.error_count += 1
        # 错误时增加等待时间
        self.min_interval = min(self.min_interval * 1.5, 10.0)  # 最大间隔10秒
    
    def get_status(self) -> dict:
        """获取状态信息"""
        return {
            'requests_per_minute': self.requests_per_minute,
            'min_interval': self.min_interval,
            'error_count': self.error_count,
            'success_count': self.success_count,
            'success_rate': self.success_count / max(1, self.success_count + self.error_count)
        }

class RetryStrategy:
    """重试策略类"""
    
    def __init__(self, config: RequestConfig):
        """
        初始化重试策略
        
        Args:
            config: 请求配置
        """
        self.config = config
        self.retry_count = 0
    
    def should_retry(self, error_type: RequestErrorType) -> bool:
        """
        判断是否应该重试
        
        Args:
            error_type: 错误类型
            
        Returns:
            bool: 是否应该重试
        """
        if self.retry_count >= self.config.max_retries:
            return False
        
        # 连接错误和限流错误应该重试
        if error_type in [RequestErrorType.CONNECTION_ERROR, RequestErrorType.RATE_LIMIT_ERROR]:
            return True
        
        # 服务器错误在特定情况下重试
        if error_type == RequestErrorType.SERVER_ERROR and self.retry_count < 2:
            return True
            
        return False
    
    def get_retry_delay(self) -> float:
        """
        获取重试延迟时间
        
        Returns:
            float: 延迟时间（秒）
        """
        # 指数退避算法
        delay = self.config.base_delay * (self.config.backoff_factor ** self.retry_count)
        
        # 添加随机抖动
        jitter = random.uniform(-self.config.jitter, self.config.jitter) * delay
        delay_with_jitter = delay + jitter
        
        # 限制最大延迟
        delay_with_jitter = min(delay_with_jitter, self.config.max_delay)
        
        self.retry_count += 1
        return delay_with_jitter
    
    def reset(self):
        """重置重试计数"""
        self.retry_count = 0

class RequestController:
    """智能请求控制器"""
    
    def __init__(self, config: Optional[RequestConfig] = None):
        """
        初始化请求控制器
        
        Args:
            config: 请求配置，如果为None则使用默认配置
        """
        self.config = config or RequestConfig()
        self.rate_limiter = RateLimiter(self.config.requests_per_minute)
        self.retry_strategy = RetryStrategy(self.config)
        self.total_requests = 0
        self.failed_requests = 0
        
        # 多线程相关属性
        self._lock = threading.Lock()  # 线程锁，用于保护共享资源
        self._active_threads = 0       # 当前活跃线程数
        self._max_workers = min(10, self.config.batch_size // 2)  # 最大工作线程数
    
    def classify_error(self, error: Exception) -> RequestErrorType:
        """
        分类错误类型
        
        Args:
            error: 异常对象
            
        Returns:
            RequestErrorType: 错误类型
        """
        error_str = str(error).lower()
        
        if 'connection' in error_str or 'remote' in error_str:
            return RequestErrorType.CONNECTION_ERROR
        elif 'rate' in error_str or 'limit' in error_str or 'too many' in error_str:
            return RequestErrorType.RATE_LIMIT_ERROR
        elif 'server' in error_str or '500' in error_str or '503' in error_str:
            return RequestErrorType.SERVER_ERROR
        else:
            return RequestErrorType.UNKNOWN_ERROR
    
    def execute_with_retry(self, func: Callable, *args, **kwargs) -> Any:
        """
        带重试机制的请求执行
        
        Args:
            func: 要执行的函数
            *args: 函数参数
            **kwargs: 函数关键字参数
            
        Returns:
            Any: 函数执行结果
            
        Raises:
            Exception: 如果所有重试都失败，则抛出异常
        """
        self.retry_strategy.reset()
        
        while True:
            try:
                # 频率控制
                self.rate_limiter.wait_if_needed()
                
                # 执行请求（使用线程超时机制，兼容Windows）
                import threading
                
                class TimeoutException(Exception):
                    pass
                
                def timeout_handler():
                    raise TimeoutException("请求超时")
                
                result = None
                exception = None
                
                def worker():
                    nonlocal result, exception
                    try:
                        result = func(*args, **kwargs)
                    except Exception as e:
                        exception = e
                
                # 创建并启动线程
                thread = threading.Thread(target=worker)
                thread.daemon = True
                thread.start()
                
                # 等待线程完成，最多30秒
                thread.join(timeout=30)
                
                # 检查线程是否还在运行
                if thread.is_alive():
                    # 线程超时
                    logger.warning("请求超时，将进行重试")
                    raise TimeoutException("请求超时")
                
                # 检查是否有异常
                if exception is not None:
                    raise exception
                
                # 记录成功
                self.rate_limiter.record_success()
                self.total_requests += 1
                
                return result
                
            except TimeoutException as e:
                # 超时错误，记录并重试
                self.total_requests += 1
                self.failed_requests += 1
                
                if self.retry_strategy.should_retry(RequestErrorType.CONNECTION_ERROR):
                    delay = self.retry_strategy.get_retry_delay()
                    logger.warning(f"请求超时，第 {self.retry_strategy.retry_count} 次重试，等待 {delay:.2f} 秒")
                    self.rate_limiter.record_error()
                    time.sleep(delay)
                else:
                    logger.error(f"请求最终超时，已重试 {self.retry_strategy.retry_count} 次")
                    raise
                    
            except Exception as e:
                self.total_requests += 1
                self.failed_requests += 1
                
                # 分类错误类型
                error_type = self.classify_error(e)
                
                # 判断是否应该重试
                if self.retry_strategy.should_retry(error_type):
                    delay = self.retry_strategy.get_retry_delay()
                    logger.warning(f"请求失败 ({error_type.value})，第 {self.retry_strategy.retry_count} 次重试，等待 {delay:.2f} 秒: {e}")
                    
                    # 记录错误
                    self.rate_limiter.record_error()
                    
                    # 等待重试
                    time.sleep(delay)
                else:
                    # 记录最终错误
                    self.rate_limiter.record_error()
                    logger.error(f"请求最终失败，已重试 {self.retry_strategy.retry_count} 次: {e}")
                    raise
    
    def _thread_safe_execute(self, task_func: Callable, task: Any) -> Any:
        """
        线程安全的执行函数
        
        Args:
            task_func: 任务处理函数
            task: 任务参数
            
        Returns:
            Any: 执行结果，失败时返回None
        """
        try:
            # 使用线程锁保护频率控制器的状态更新
            with self._lock:
                self._active_threads += 1
            
            # 执行任务
            result = self.execute_with_retry(task_func, task)
            
            return result
            
        except Exception as e:
            logger.error(f"线程任务执行失败: {e}")
            return None
            
        finally:
            # 确保线程计数正确减少
            with self._lock:
                self._active_threads -= 1
    
    def batch_execute(self, tasks: list, task_func: Callable, progress_desc: str = "处理任务", 
                     use_multithreading: bool = True, max_workers: int = None) -> list:
        """
        批量执行任务（支持多线程）
        
        Args:
            tasks: 任务列表
            task_func: 任务处理函数
            progress_desc: 进度描述
            use_multithreading: 是否使用多线程
            max_workers: 最大工作线程数，如果为None则自动计算
            
        Returns:
            list: 成功的结果列表
        """
        from tqdm import tqdm
        
        if not tasks:
            logger.warning("任务列表为空")
            return []
        
        # 计算合适的线程数
        if max_workers is None:
            max_workers = min(len(tasks), self._max_workers)
        else:
            max_workers = min(max_workers, len(tasks), self._max_workers)
        
        # 如果禁用多线程或最大工作线程数小于等于1，使用单线程
        if not use_multithreading or max_workers <= 1:
            logger.info(f"使用单线程模式处理 {len(tasks)} 个任务")
            return self._sequential_batch_execute(tasks, task_func, progress_desc)
        
        # 如果任务数量少，使用较小的线程数但仍然是多线程
        if len(tasks) <= 5:
            max_workers = min(2, len(tasks))  # 对于少量任务，使用1-2个线程
            logger.info(f"任务数量较少 ({len(tasks)})，使用 {max_workers} 个线程的多线程模式")
        
        logger.info(f"使用多线程模式处理 {len(tasks)} 个任务，最大并发数: {max_workers}")
        
        results = []
        successful_tasks = 0
        failed_tasks = 0
        
        # 使用线程池执行任务
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            # 提交所有任务
            future_to_task = {executor.submit(self._thread_safe_execute, task_func, task): task 
                            for task in tasks}
            
            # 使用进度条显示进度
            with tqdm(total=len(tasks), desc=progress_desc) as pbar:
                # 按完成顺序处理结果
                for future in as_completed(future_to_task):
                    task = future_to_task[future]
                    
                    try:
                        result = future.result()
                        if result is not None:
                            results.append(result)
                            successful_tasks += 1
                    except Exception as e:
                        failed_tasks += 1
                        logger.error(f"任务处理失败: {e}")
                    
                    pbar.update(1)
        
        # 输出统计信息
        success_rate = successful_tasks / len(tasks) if tasks else 0
        logger.info(f"批量处理完成: 成功 {successful_tasks}/{len(tasks)}，失败 {failed_tasks}，成功率 {success_rate:.2%}")
        
        return results
    
    def _sequential_batch_execute(self, tasks: list, task_func: Callable, progress_desc: str = "处理任务") -> list:
        """
        顺序批量执行任务（单线程模式）
        
        Args:
            tasks: 任务列表
            task_func: 任务处理函数
            progress_desc: 进度描述
            
        Returns:
            list: 成功的结果列表
        """
        from tqdm import tqdm
        
        results = []
        successful_tasks = 0
        failed_tasks = 0
        
        for i, task in enumerate(tqdm(tasks, desc=progress_desc)):
            try:
                result = self.execute_with_retry(task_func, task)
                if result is not None:
                    results.append(result)
                    successful_tasks += 1
                
                # 批次间额外延迟，避免连续请求
                if (i + 1) % 10 == 0:
                    time.sleep(0.5)
                    
            except Exception as e:
                failed_tasks += 1
                logger.error(f"任务 {i+1} 处理失败: {e}")
        
        # 输出统计信息
        success_rate = successful_tasks / len(tasks) if tasks else 0
        logger.info(f"批量处理完成: 成功 {successful_tasks}/{len(tasks)}，失败 {failed_tasks}，成功率 {success_rate:.2%}")
        
        return results
    
    def get_statistics(self) -> dict:
        """
        获取统计信息
        
        Returns:
            dict: 统计信息
        """
        rate_limiter_stats = self.rate_limiter.get_status()
        
        return {
            'total_requests': self.total_requests,
            'failed_requests': self.failed_requests,
            'success_rate': (self.total_requests - self.failed_requests) / max(1, self.total_requests),
            'rate_limiter': rate_limiter_stats
        }

# 默认控制器实例
default_controller = RequestController()

def execute_with_retry(func: Callable, *args, **kwargs) -> Any:
    """
    使用默认控制器的便捷函数
    
    Args:
        func: 要执行的函数
        *args: 函数参数
        **kwargs: 函数关键字参数
        
    Returns:
        Any: 函数执行结果
    """
    return default_controller.execute_with_retry(func, *args, **kwargs)