"""
获取沪深300指数成分股列表
通过akshare接口获取最新的沪深300指数成分股信息，并格式化成标准格式
"""

import akshare as ak
import pandas as pd
import os
from datetime import datetime
import logging

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def get_hs300_components():
    """
    获取沪深300指数成分股列表
    
    Returns:
        pandas.DataFrame: 包含沪深300成分股信息的DataFrame
    """
    try:
        logger.info("开始获取沪深300指数成分股列表...")
        
        # 使用akshare接口获取沪深300成分股
        hs300_df = ak.index_stock_cons_sina(symbol="000300")
        
        if hs300_df.empty:
            logger.warning("未获取到沪深300成分股数据")
            return pd.DataFrame()
        
        logger.info(f"成功获取 {len(hs300_df)} 只沪深300成分股")
        
        # 显示数据的基本信息
        logger.info(f"数据字段: {list(hs300_df.columns)}")
        
        return hs300_df
        
    except Exception as e:
        logger.error(f"获取沪深300成分股失败: {e}")
        return pd.DataFrame()

def format_components_data(df):
    """
    将获取的成分股数据格式化成标准格式
    
    Args:
        df (pandas.DataFrame): 原始成分股数据
        
    Returns:
        pandas.DataFrame: 格式化后的成分股数据
    """
    if df.empty:
        return pd.DataFrame()
    
    # 创建新的DataFrame来存储格式化后的数据
    formatted_data = []
    
    # 获取当前时间
    current_time = datetime.now().strftime("%Y/%m/%d %H:%M")
    
    # 处理每一只股票
    for _, row in df.iterrows():
        # 获取股票代码（保持原始格式，不进行补零或去零操作）
        stock_code = str(row.get('code', ''))
        
        # 获取股票名称
        stock_name = row.get('name', '')
        
        # 由于akshare接口不提供纳入日期信息，我们使用默认值
        # 在实际应用中，可能需要从其他数据源获取准确的纳入日期
        inclusion_date = "2024-12-16"  # 默认纳入日期
        
        # 指数代码固定为300
        index_code = "300"
        
        # 数据来源固定为backup
        data_source = "backup"
        
        formatted_data.append({
            '股票代码': stock_code,
            '股票名称': stock_name,
            '纳入日期': inclusion_date,
            '指数代码': index_code,
            '数据来源': data_source,
            '获取时间': current_time
        })
    
    # 创建DataFrame
    formatted_df = pd.DataFrame(formatted_data)
    
    return formatted_df

def save_components_to_csv(df, output_file=None):
    """
    将成分股数据保存到CSV文件
    
    Args:
        df (pandas.DataFrame): 成分股数据
        output_file (str, optional): 输出文件路径
    
    Returns:
        str: 保存的文件路径
    """
    if df.empty:
        logger.warning("数据为空，不进行保存")
        return ""
    
    # 如果没有指定输出文件，使用默认路径
    if output_file is None:
        # 创建输出目录
        output_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), 
                                 'data_pipeline', 'data', 'components')
        os.makedirs(output_dir, exist_ok=True)
        
        # 生成带时间戳的文件名
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_file = os.path.join(output_dir, f"hs300_components_{timestamp}.csv")
    
    try:
        # 保存数据
        df.to_csv(output_file, index=False, encoding='utf-8-sig')
        logger.info(f"成分股数据已保存到: {output_file}")
        return output_file
    except Exception as e:
        logger.error(f"保存成分股数据失败: {e}")
        return ""

def analyze_components(df):
    """
    分析成分股数据
    
    Args:
        df (pandas.DataFrame): 成分股数据
    """
    if df.empty:
        logger.warning("数据为空，无法进行分析")
        return
    
    logger.info("=== 沪深300成分股分析 ===")
    
    # 显示基本信息
    logger.info(f"总成分股数量: {len(df)}")
    
    # 显示前10只成分股
    logger.info("前10只成分股:")
    for i, row in df.head(10).iterrows():
        logger.info(f"  {i+1}. {row.get('股票代码', 'N/A')} - {row.get('股票名称', 'N/A')}")
    
    # 统计交易所分布
    if '股票代码' in df.columns:
        # 根据股票代码前缀判断交易所
        sh_stocks = df[df['股票代码'].str.startswith(('6', '9'))]  # 上海交易所
        sz_stocks = df[df['股票代码'].str.startswith(('0', '3'))]  # 深圳交易所
        
        logger.info(f"上海交易所股票数量: {len(sh_stocks)}")
        logger.info(f"深圳交易所股票数量: {len(sz_stocks)}")

def main():
    """
    主函数：获取并保存沪深300成分股列表
    """
    logger.info("=== 沪深300成分股获取程序开始 ===")
    
    # 获取成分股数据
    components_df = get_hs300_components()
    
    if components_df.empty:
        logger.error("未能获取到成分股数据，程序结束")
        return
    
    # 格式化数据
    formatted_df = format_components_data(components_df)
    
    if formatted_df.empty:
        logger.error("数据格式化失败，程序结束")
        return
    
    # 分析成分股数据
    analyze_components(formatted_df)
    
    # 保存数据
    saved_file = save_components_to_csv(formatted_df)
    
    if saved_file:
        logger.info(f"程序执行完成，数据已保存至: {saved_file}")
        
        # 显示数据格式信息
        logger.info("=== 数据格式信息 ===")
        logger.info(f"数据字段: {list(formatted_df.columns)}")
        logger.info("数据格式预览:")
        logger.info(formatted_df.head())
    else:
        logger.error("数据保存失败")
    
    logger.info("=== 程序执行结束 ===")

if __name__ == "__main__":
    main()