#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
特征工程主程序
整合股票数据分析核心流程
"""

import os
import sys
import pandas as pd
import logging
from datetime import datetime
from technical_indicators import TechnicalIndicators

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler('../data/logs/feature_engineering.log', encoding='utf-8')
    ]
)
logger = logging.getLogger(__name__)

def validate_data_structure(df: pd.DataFrame) -> bool:
    """
    验证数据文件结构是否符合预期
    
    参数:
        df: 要验证的数据框
        
    返回:
        bool: 数据是否有效
    """
    required_columns = ['date', '股票代码', 'open', 'close', 'high', 'low', 'volume']
    
    missing_columns = [col for col in required_columns if col not in df.columns]
    if missing_columns:
        logger.error(f"数据文件缺少必要列: {missing_columns}")
        return False
    
    # 检查数据完整性
    if df.empty:
        logger.error("数据文件为空")
        return False
    
    # 检查日期列格式
    try:
        df['date'] = pd.to_datetime(df['date'])
    except Exception as e:
        logger.error(f"日期列格式错误: {str(e)}")
        return False
    
    return True

def create_output_directories():
    """创建必要的输出目录"""
    directories = [
        '../data/features',
        '../data/logs'
    ]
    
    for directory in directories:
        if not os.path.exists(directory):
            os.makedirs(directory)
            logger.info(f"创建目录: {directory}")

def run_feature_engineering_pipeline():
    """运行完整的特征工程管道"""
    
    # 创建输出目录
    create_output_directories()
    
    # 配置技术指标参数（短线策略优化）
    config = {
        'return_periods': [1, 3, 5],        # 短线收益率计算周期
        'rolling_window': 10,              # 短线滚动窗口大小
        'momentum_period': 5,              # 短线动量指标周期
        'rsi_period': 6,                   # 短线RSI周期
        'turnover_window': 3,              # 短线换手率移动平均窗口
        'volatility_window': 5,           # 波动率计算窗口
        'macd_fast': 12,                  # MACD快速线周期
        'macd_slow': 26,                  # MACD慢速线周期
        'macd_signal': 9                  # MACD信号线周期
    }
    
    # 输入输出文件路径
    input_file = "../data/daily_prices/Merge/hs300_daily_prices_merged.csv"
    output_file = "../data/features/technical_indicators.csv"
    log_file = "../data/logs/technical_indicators_processing_log.json"
    
    logger.info("开始股票数据分析核心流程")
    logger.info(f"输入文件: {input_file}")
    logger.info(f"输出文件: {output_file}")
    
    try:
        # 检查输入文件是否存在
        if not os.path.exists(input_file):
            logger.error(f"输入文件不存在: {input_file}")
            return False
        
        # 创建技术指标计算器
        calculator = TechnicalIndicators(config)
        
        # 1. 从数据源读取日线交易数据
        logger.info("步骤1: 读取日线交易数据")
        df = calculator.load_data(input_file)
        
        # 验证数据结构
        if not validate_data_structure(df):
            return False
        
        # 2. 计算技术指标因子
        logger.info("步骤2: 计算技术指标因子")
        df_with_indicators = calculator.process_all_indicators(df)
        
        # 3. 输出结构化数据集
        logger.info("步骤3: 输出结构化数据集")
        df_with_indicators.to_csv(output_file, index=False, encoding='utf-8-sig')
        
        # 4. 保存处理日志
        logger.info("步骤4: 保存序列化处理日志")
        calculator.save_processing_log(log_file)
        
        # 输出结果统计
        logger.info("特征工程完成！结果统计:")
        logger.info(f"  原始数据行数: {len(df)}")
        logger.info(f"  处理后数据行数: {len(df_with_indicators)}")
        logger.info(f"  技术指标数量: {len([col for col in df_with_indicators.columns if col not in df.columns])}")
        logger.info(f"  数据时间范围: {df_with_indicators['date'].min()} 到 {df_with_indicators['date'].max()}")
        logger.info(f"  唯一股票数量: {df_with_indicators['股票代码'].nunique()}")
        
        # 显示生成的技术指标列
        indicator_columns = [col for col in df_with_indicators.columns 
                           if any(keyword in col for keyword in ['return', 'rolling', 'momentum', 'rsi', 'turnover', 'volatility', 'macd'])]
        logger.info(f"  生成的短线策略技术指标: {indicator_columns}")
        
        return True
        
    except Exception as e:
        logger.error(f"特征工程管道执行失败: {str(e)}")
        return False

def main():
    """主函数"""
    logger.info("=" * 60)
    logger.info("股票数据分析核心流程 - 特征工程")
    logger.info("=" * 60)
    
    success = run_feature_engineering_pipeline()
    
    if success:
        logger.info("特征工程任务完成！")
        return 0
    else:
        logger.error("特征工程任务失败！")
        return 1

if __name__ == "__main__":
    exit(main())