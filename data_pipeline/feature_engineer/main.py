#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
特征工程主程序
整合完整的特征工程流程：技术指标、基本因子、宏观因子、EPU因子和因子选择
"""

import os
import sys
import pandas as pd
import logging
from datetime import datetime
import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from technical_indicators import TechnicalIndicators
from fundamental_factors import FundamentalFactors
from macro_factors import MacroFactors
from epu_factors import EPUFactors
from factor_selection import FactorSelection

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
    
    # 配置参数
    tech_config = {
        'return_periods': [1, 3, 5],
        'rolling_window': 10,
        'momentum_period': 5,
        'rsi_period': 6,
        'turnover_window': 3,
        'volatility_window': 5,
        'macd_fast': 12,
        'macd_slow': 26,
        'macd_signal': 9
    }
    
    fundamental_config = {
        'factors': ['pe', 'pb', 'roe', 'roa', 'revenue_growth', 'profit_growth']
    }
    
    macro_config = {
        'factors': ['gdp_growth', 'cpi', 'interest_rate', 'm2_growth']
    }
    
    epu_config = {
        'epu_file_path': '../data/EPU/China_Mainland_Paper_EPU.xlsx',
        'lag_periods': [1, 3, 6, 12],
        'growth_periods': [1, 3, 6, 12]
    }
    
    selection_config = {
        'correlation_threshold': 0.8,
        'pca_variance_threshold': 0.95,
        'top_k_factors': 50,
        'selection_methods': ['correlation', 'pca', 'feature_importance']
    }
    
    logger.info("开始完整的特征工程流程")
    
    try:
        # 1. 技术指标计算
        logger.info("步骤1: 计算技术指标因子")
        tech_calculator = TechnicalIndicators(tech_config)
        tech_input_file = "../data/daily_prices/Merge/hs300_daily_prices_merged.csv"
        
        # 初始化默认值
        stock_codes = ['000001', '000002', '000858']  # 示例股票代码
        start_date = '2020-01-01'
        end_date = '2024-12-31'
        price_data = pd.DataFrame()
        
        if os.path.exists(tech_input_file):
            df_tech = tech_calculator.load_data(tech_input_file)
            if validate_data_structure(df_tech):
                df_tech_factors = tech_calculator.process_all_indicators(df_tech)
                df_tech_factors.to_csv("../data/features/technical_indicators.csv", index=False, encoding='utf-8-sig')
                tech_calculator.save_processing_log("../data/logs/technical_indicators_processing_log.json")
                logger.info("技术指标计算完成")
                
                # 从技术指标数据中获取股票代码和日期范围
                stock_codes = df_tech['股票代码'].unique().tolist()
                start_date = df_tech['date'].min().strftime('%Y-%m-%d')
                end_date = df_tech['date'].max().strftime('%Y-%m-%d')
                price_data = df_tech[['date', '股票代码', 'close']].rename(columns={'date': '日期'})
            else:
                logger.warning("技术指标数据验证失败，跳过此步骤")
        else:
            logger.warning(f"技术指标输入文件不存在: {tech_input_file}")
        
        # 2. 基本因子计算
        logger.info("步骤2: 计算基本因子")
        fundamental_calculator = FundamentalFactors(fundamental_config)
        
        df_fundamental = fundamental_calculator.process_all_factors(stock_codes, start_date, end_date, price_data)
        if not df_fundamental.empty:
            df_fundamental.to_csv("../data/features/fundamental_factors.csv", index=False, encoding='utf-8-sig')
            fundamental_calculator.save_processing_log("../data/logs/fundamental_factors_processing_log.json")
            logger.info("基本因子计算完成")
        
        # 3. 宏观因子计算
        logger.info("步骤3: 计算宏观因子")
        macro_calculator = MacroFactors(macro_config)
        df_macro = macro_calculator.process_all_factors(start_date, end_date)
        if not df_macro.empty:
            df_macro.to_csv("../data/features/macro_factors.csv", index=False, encoding='utf-8-sig')
            macro_calculator.save_processing_log("../data/logs/macro_factors_processing_log.json")
            logger.info("宏观因子计算完成")
        
        # 4. EPU因子计算
        logger.info("步骤4: 计算EPU因子")
        epu_calculator = EPUFactors(epu_config)
        df_epu = epu_calculator.process_all_factors()
        if not df_epu.empty:
            df_epu.to_csv("../data/features/epu_factors.csv", index=False, encoding='utf-8-sig')
            epu_calculator.save_processing_log("../data/logs/epu_factors_processing_log.json")
            logger.info("EPU因子计算完成")
        
        # 5. 因子选择
        logger.info("步骤5: 进行因子选择分析")
        selector = FactorSelection(selection_config)
        factors, returns = selector.load_all_factors()
        
        if not factors.empty:
            final_factors = selector.select_final_factors(factors, returns)
            
            results = {
                'selected_factors': final_factors,
                'total_factors_analyzed': len(factors.columns),
                'selection_config': selection_config
            }
            
            selector.save_selection_results(results, "../data/features/factor_selection_results.json")
            logger.info(f"因子选择完成，最终选择{len(final_factors)}个因子")
        
        logger.info("完整的特征工程流程执行完成！")
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