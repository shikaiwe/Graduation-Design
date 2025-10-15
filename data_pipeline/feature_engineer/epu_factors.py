#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
EPU因子计算模块
实现EPU指数的滞后值、同比增长率等因子的计算
"""

import pandas as pd
import numpy as np
import logging
from typing import Dict, Any, Optional
import json
from datetime import datetime

# 配置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class EPUFactors:
    """EPU因子计算器"""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        初始化EPU因子计算器
        
        参数:
            config: 配置参数字典
        """
        self.config = config or {
            'epu_file_path': 'data_pipeline/data/EPU/China_Mainland_Paper_EPU.xlsx',
            'lag_periods': [1, 3, 6, 12],  # 滞后周期（月）
            'growth_periods': [1, 3, 6, 12]  # 增长率计算周期
        }
        
        self.processing_log = {
            'start_time': datetime.now().isoformat(),
            'config': self.config,
            'steps': []
        }
    
    def _log_step(self, step_name: str, details: Dict[str, Any]):
        """记录处理步骤"""
        step_info = {
            'step': step_name,
            'timestamp': datetime.now().isoformat(),
            'details': details
        }
        self.processing_log['steps'].append(step_info)
        logger.info(f"完成步骤: {step_name}")
    
    def load_epu_data(self) -> pd.DataFrame:
        """
        加载EPU数据
        
        返回:
            pd.DataFrame: 处理后的EPU数据
        """
        try:
            logger.info(f"加载EPU数据: {self.config['epu_file_path']}")
            
            # 读取EPU Excel文件
            epu_data = pd.read_excel(self.config['epu_file_path'])
            
            # 清理数据 - 只保留有用的列
            useful_columns = []
            for col in epu_data.columns:
                if 'year' in col.lower() or 'month' in col.lower() or 'epu' in col.lower():
                    useful_columns.append(col)
            
            if len(useful_columns) < 3:
                # 如果自动识别失败，使用前3列
                useful_columns = epu_data.columns[:3].tolist()
            
            epu_data = epu_data[useful_columns]
            
            # 重命名列
            column_mapping = {}
            for col in epu_data.columns:
                col_lower = col.lower()
                if 'year' in col_lower:
                    column_mapping[col] = 'year'
                elif 'month' in col_lower:
                    column_mapping[col] = 'month'
                elif 'epu' in col_lower:
                    column_mapping[col] = 'epu'
            
            epu_data = epu_data.rename(columns=column_mapping)
            
            # 创建日期列
            epu_data['date'] = pd.to_datetime(
                epu_data['year'].astype(str) + '-' + epu_data['month'].astype(str) + '-01'
            )
            
            # 按日期排序
            epu_data = epu_data.sort_values('date')
            
            # 清理EPU列（去除可能的非数值字符）
            epu_data['epu'] = pd.to_numeric(epu_data['epu'], errors='coerce')
            
            # 删除空值
            epu_data = epu_data.dropna(subset=['epu'])
            
            self._log_step('load_epu_data', {
                'original_columns': useful_columns,
                'final_columns': epu_data.columns.tolist(),
                'date_range': f"{epu_data['date'].min()} to {epu_data['date'].max()}",
                'rows_obtained': len(epu_data)
            })
            
            return epu_data[['date', 'epu']]
            
        except Exception as e:
            logger.error(f"加载EPU数据失败: {str(e)}")
            raise
    
    def calculate_lag_factors(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        计算EPU滞后因子
        
        参数:
            df: EPU数据DataFrame
            
        返回:
            pd.DataFrame: 包含滞后因子的数据
        """
        try:
            df = df.copy()
            
            # 计算各滞后周期的EPU值
            for period in self.config['lag_periods']:
                lag_col = f'epu_lag_{period}m'
                df[lag_col] = df['epu'].shift(period)
            
            self._log_step('calculate_lag_factors', {
                'lag_periods': self.config['lag_periods'],
                'lag_columns_created': [f'epu_lag_{period}m' for period in self.config['lag_periods']]
            })
            
            return df
            
        except Exception as e:
            logger.error(f"计算滞后因子失败: {str(e)}")
            raise
    
    def calculate_growth_factors(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        计算EPU增长率因子
        
        参数:
            df: EPU数据DataFrame
            
        返回:
            pd.DataFrame: 包含增长率因子的数据
        """
        try:
            df = df.copy()
            
            # 计算同比增长率
            for period in self.config['growth_periods']:
                growth_col = f'epu_yoy_{period}m'
                df[growth_col] = df['epu'].pct_change(periods=period)
            
            # 计算环比增长率
            df['epu_mom'] = df['epu'].pct_change(periods=1)  # 月环比
            
            self._log_step('calculate_growth_factors', {
                'growth_periods': self.config['growth_periods'],
                'growth_columns_created': [f'epu_yoy_{period}m' for period in self.config['growth_periods']] + ['epu_mom']
            })
            
            return df
            
        except Exception as e:
            logger.error(f"计算增长率因子失败: {str(e)}")
            raise
    
    def calculate_volatility_factors(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        计算EPU波动率因子
        
        参数:
            df: EPU数据DataFrame
            
        返回:
            pd.DataFrame: 包含波动率因子的数据
        """
        try:
            df = df.copy()
            
            # 计算滚动波动率（12个月）
            df['epu_volatility_12m'] = df['epu'].rolling(window=12).std()
            
            # 计算滚动波动率（6个月）
            df['epu_volatility_6m'] = df['epu'].rolling(window=6).std()
            
            # 计算Z-score标准化值
            df['epu_zscore'] = (df['epu'] - df['epu'].mean()) / df['epu'].std()
            
            self._log_step('calculate_volatility_factors', {
                'volatility_columns_created': ['epu_volatility_12m', 'epu_volatility_6m', 'epu_zscore']
            })
            
            return df
            
        except Exception as e:
            logger.error(f"计算波动率因子失败: {str(e)}")
            raise
    
    def process_all_factors(self) -> pd.DataFrame:
        """
        批量计算所有EPU因子
        
        返回:
            pd.DataFrame: 包含所有EPU因子的数据
        """
        try:
            logger.info("开始计算EPU因子...")
            
            # 加载EPU数据
            df = self.load_epu_data()
            
            if df.empty:
                logger.warning("未加载到EPU数据，返回空DataFrame")
                return df
            
            # 计算滞后因子
            df = self.calculate_lag_factors(df)
            
            # 计算增长率因子
            df = self.calculate_growth_factors(df)
            
            # 计算波动率因子
            df = self.calculate_volatility_factors(df)
            
            # 重命名日期列为标准格式
            df = df.rename(columns={'date': '日期'})
            
            # 完成处理记录
            self.processing_log['end_time'] = datetime.now().isoformat()
            self.processing_log['total_rows'] = str(len(df))
            self.processing_log['factors_calculated'] = df.columns.tolist()
            
            logger.info("EPU因子计算完成！")
            return df
            
        except Exception as e:
            logger.error(f"批量计算EPU因子失败: {str(e)}")
            raise
    
    def save_processing_log(self, file_path: str):
        """保存处理日志到文件"""
        try:
            with open(file_path, 'w', encoding='utf-8') as f:
                json.dump(self.processing_log, f, ensure_ascii=False, indent=2)
            logger.info(f"处理日志已保存到: {file_path}")
        except Exception as e:
            logger.error(f"保存处理日志失败: {str(e)}")


def main():
    """主函数示例"""
    # 配置参数
    config = {
        'epu_file_path': 'data_pipeline/data/EPU/China_Mainland_Paper_EPU.xlsx',
        'lag_periods': [1, 3, 6, 12],
        'growth_periods': [1, 3, 6, 12]
    }
    
    # 创建EPU因子计算器
    calculator = EPUFactors(config)
    
    try:
        # 计算EPU因子
        df_epu_factors = calculator.process_all_factors()
        
        if not df_epu_factors.empty:
            # 保存结果
            output_file = "data_pipeline/data/features/epu_factors.csv"
            df_epu_factors.to_csv(output_file, index=False, encoding='utf-8-sig')
            
            # 保存处理日志
            log_file = "data_pipeline/data/logs/epu_factors_processing_log.json"
            calculator.save_processing_log(log_file)
            
            logger.info(f"EPU因子计算完成！结果已保存到: {output_file}")
        else:
            logger.warning("未生成EPU因子数据")
        
    except Exception as e:
        logger.error(f"EPU因子计算失败: {str(e)}")


if __name__ == "__main__":
    main()