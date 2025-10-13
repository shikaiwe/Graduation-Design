"""
数据采集配置文件
配置沪深300成分股数据采集的相关参数
"""

import os
from datetime import datetime

# 基础路径配置
BASE_DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'data')
COMPONENTS_FILE = os.path.join(BASE_DATA_DIR, 'components', 'hs300_components_cutoff_2024.csv')
OUTPUT_DIR = os.path.join(BASE_DATA_DIR, 'daily_prices')

# 时间范围配置（支持可配置）
START_DATE = "2019-01-01"  # 开始日期
END_DATE = "2024-12-31"    # 结束日期

# 数据采集配置
BATCH_SIZE = 20  # 每批次处理的股票数量
REQUEST_DELAY = 2  # 请求间隔时间（秒），避免频率限制
MAX_RETRIES = 3   # 最大重试次数
RETRY_DELAY = 5   # 重试间隔时间（秒）

# AKshare接口配置
AKSHARE_STOCK_CODE_PREFIX_MAP = {
    # 上海证券交易所
    '600': 'sh',
    '601': 'sh', 
    '603': 'sh',
    '605': 'sh',
    '688': 'sh',
    # 深圳证券交易所
    '000': 'sz',
    '001': 'sz',
    '002': 'sz',
    '003': 'sz',
    '300': 'sz',
    '301': 'sz'
}

# 日志配置
LOG_CONFIG = {
    'level': 'INFO',
    'format': '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    'filename': os.path.join(BASE_DATA_DIR, 'logs', 'data_collector.log')
}

# 数据字段映射（AKshare返回字段到标准字段的映射）
FIELD_MAPPING = {
    '日期': 'date',
    '开盘': 'open',
    '收盘': 'close',
    '最高': 'high',
    '最低': 'low',
    '成交量': 'volume',
    '成交额': 'amount',
    '振幅': 'amplitude',
    '涨跌幅': 'change_rate',
    '涨跌额': 'change_amount',
    '换手率': 'turnover_rate'
}

def get_stock_symbol(stock_code):
    """
    根据股票代码生成AKshare所需的股票符号
    
    Args:
        stock_code (str): 股票代码
        
    Returns:
        str: 6位数字的股票代码，如 '600000'
    """
    # 将股票代码转换为字符串并补齐前导零到6位
    stock_code_str = str(stock_code).zfill(6)
    
    # 直接返回6位数字的股票代码，AKshare接口不需要交易所前缀
    return stock_code_str


# 导出配置变量
__all__ = [
    'BASE_DATA_DIR', 'COMPONENTS_FILE', 'OUTPUT_DIR',
    'START_DATE', 'END_DATE', 'BATCH_SIZE', 'REQUEST_DELAY', 
    'MAX_RETRIES', 'RETRY_DELAY', 'AKSHARE_STOCK_CODE_PREFIX_MAP',
    'LOG_CONFIG', 'FIELD_MAPPING', 'get_stock_symbol'
]