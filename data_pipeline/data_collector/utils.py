"""
工具函数模块
包含数据采集过程中的辅助功能
"""

import os
import pandas as pd
from datetime import datetime, timedelta
from typing import List, Dict, Optional
import logging


def create_output_directory(output_dir: str) -> bool:
    """
    创建输出目录
    
    Args:
        output_dir (str): 输出目录路径
        
    Returns:
        bool: 创建成功返回True，否则返回False
    """
    try:
        os.makedirs(output_dir, exist_ok=True)
        return True
    except Exception as e:
        logging.error(f"创建目录 {output_dir} 失败: {e}")
        return False


def validate_date_range(start_date: str, end_date: str) -> bool:
    """
    验证日期范围是否有效
    
    Args:
        start_date (str): 开始日期，格式 'YYYY-MM-DD'
        end_date (str): 结束日期，格式 'YYYY-MM-DD'
        
    Returns:
        bool: 日期范围有效返回True，否则返回False
    """
    try:
        start = datetime.strptime(start_date, '%Y-%m-%d')
        end = datetime.strptime(end_date, '%Y-%m-%d')
        
        if start > end:
            logging.error(f"开始日期 {start_date} 不能晚于结束日期 {end_date}")
            return False
        
        # 检查是否在合理范围内（比如不早于2000年）
        if start.year < 2000:
            logging.warning(f"开始日期 {start_date} 较早，数据可能不完整")
        
        return True
        
    except ValueError as e:
        logging.error(f"日期格式无效: {e}")
        return False


def get_processed_stocks(output_dir: str) -> set:
    """
    获取已处理的股票列表
    
    Args:
        output_dir (str): 输出目录路径
        
    Returns:
        set: 已处理的股票代码集合
    """
    processed_stocks = set()
    
    if not os.path.exists(output_dir):
        return processed_stocks
    
    try:
        # 遍历输出目录中的所有CSV文件
        for filename in os.listdir(output_dir):
            if filename.endswith('.csv') and 'hs300_daily_prices_batch' in filename:
                filepath = os.path.join(output_dir, filename)
                
                try:
                    # 读取文件获取股票代码
                    df = pd.read_csv(filepath)
                    if 'stock_code' in df.columns:
                        stocks = set(df['stock_code'].unique())
                        processed_stocks.update(stocks)
                except Exception as e:
                    logging.warning(f"读取文件 {filename} 失败: {e}")
        
        logging.info(f"找到 {len(processed_stocks)} 只已处理的股票")
        
    except Exception as e:
        logging.error(f"获取已处理股票列表失败: {e}")
    
    return processed_stocks


def format_stock_code(stock_code: str) -> str:
    """
    格式化股票代码
    
    Args:
        stock_code (str): 原始股票代码
        
    Returns:
        str: 格式化后的股票代码（6位数字）
    """
    # 去除可能的空格和特殊字符
    code = str(stock_code).strip()
    
    # 补齐到6位
    if len(code) < 6:
        code = code.zfill(6)
    
    return code


def calculate_progress(current: int, total: int) -> Dict[str, float]:
    """
    计算进度信息
    
    Args:
        current (int): 当前进度
        total (int): 总数量
        
    Returns:
        Dict[str, float]: 包含进度信息的字典
    """
    if total == 0:
        return {
            'percentage': 0.0,
            'completed': 0,
            'remaining': 0
        }
    
    percentage = (current / total) * 100
    remaining = total - current
    
    return {
        'percentage': round(percentage, 2),
        'completed': current,
        'remaining': remaining
    }


def format_timedelta(delta: timedelta) -> str:
    """
    格式化时间差为可读字符串
    
    Args:
        delta (timedelta): 时间差
        
    Returns:
        str: 格式化后的时间字符串
    """
    total_seconds = int(delta.total_seconds())
    
    if total_seconds < 60:
        return f"{total_seconds}秒"
    elif total_seconds < 3600:
        minutes = total_seconds // 60
        seconds = total_seconds % 60
        return f"{minutes}分{seconds}秒"
    else:
        hours = total_seconds // 3600
        minutes = (total_seconds % 3600) // 60
        return f"{hours}小时{minutes}分"


def estimate_remaining_time(start_time: datetime, completed: int, total: int) -> Optional[str]:
    """
    估算剩余时间
    
    Args:
        start_time (datetime): 开始时间
        completed (int): 已完成数量
        total (int): 总数量
        
    Returns:
        Optional[str]: 估算的剩余时间字符串，无法估算时返回None
    """
    if completed == 0:
        return None
    
    current_time = datetime.now()
    elapsed = current_time - start_time
    
    # 计算每个项目的平均时间
    time_per_item = elapsed.total_seconds() / completed
    
    # 估算剩余时间
    remaining_items = total - completed
    remaining_seconds = time_per_item * remaining_items
    
    if remaining_seconds > 0:
        remaining_delta = timedelta(seconds=remaining_seconds)
        return format_timedelta(remaining_delta)
    
    return None


def cleanup_old_files(directory: str, days: int = 30):
    """
    清理指定天数前的旧文件
    
    Args:
        directory (str): 目录路径
        days (int): 保留天数，默认30天
    """
    if not os.path.exists(directory):
        return
    
    cutoff_time = datetime.now() - timedelta(days=days)
    
    try:
        for filename in os.listdir(directory):
            filepath = os.path.join(directory, filename)
            
            if os.path.isfile(filepath):
                file_time = datetime.fromtimestamp(os.path.getmtime(filepath))
                
                if file_time < cutoff_time:
                    os.remove(filepath)
                    logging.info(f"清理旧文件: {filename}")
    
    except Exception as e:
        logging.error(f"清理文件失败: {e}")


def validate_csv_file(filepath: str) -> bool:
    """
    验证CSV文件是否有效
    
    Args:
        filepath (str): 文件路径
        
    Returns:
        bool: 文件有效返回True，否则返回False
    """
    try:
        # 检查文件是否存在
        if not os.path.exists(filepath):
            return False
        
        # 尝试读取文件
        df = pd.read_csv(filepath, nrows=1)  # 只读取第一行
        
        # 检查是否有数据
        if df.empty:
            return False
        
        # 检查必要的列是否存在
        required_columns = ['date', 'open', 'close', 'high', 'low', 'volume']
        missing_columns = [col for col in required_columns if col not in df.columns]
        
        if missing_columns:
            logging.warning(f"CSV文件缺少必要列: {missing_columns}")
            return False
        
        return True
        
    except Exception as e:
        logging.error(f"验证CSV文件失败: {e}")
        return False