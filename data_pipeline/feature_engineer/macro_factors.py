#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
宏观因子计算模块
实现GDP增长率、CPI、利率等宏观因子的计算和集成
"""

import pandas as pd
import numpy as np
import logging
from typing import Dict, Any, Optional, Sequence, Tuple
import json
from datetime import datetime, timedelta
import akshare as ak
import requests

from utils.interpolation import InterpolationConfig, convert_monthly_to_daily

# 配置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class MacroFactors:
    """宏观因子计算器"""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        初始化宏观因子计算器
        
        参数:
            config: 配置参数字典
        """
        self.config = config or {
            'data_sources': ['akshare', '公开API'],  # 数据源选择
            'macro_indicators': ['GDP', 'CPI', '利率', 'M2', 'PMI'],  # 宏观指标
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
    
    def get_gdp_data(self, start_date: str, end_date: str) -> pd.DataFrame:
        """
        获取GDP数据
        
        参数:
            start_date: 开始日期
            end_date: 结束日期
            
        返回:
            pd.DataFrame: GDP数据
        """
        try:
            logger.info("获取GDP数据")
            
            # 使用akshare获取GDP数据
            gdp_data = ak.macro_china_gdp()
            
            if not gdp_data.empty:
                # 处理季度格式，转换为标准日期格式
                def parse_quarter(quarter_str):
                    try:
                        year = int(quarter_str[:4])
                        quarter = int(quarter_str[5])
                        month = (quarter - 1) * 3 + 1  # 季度转换为月份
                        return pd.Timestamp(year=year, month=month, day=1)
                    except:
                        return pd.NaT
                
                gdp_data['日期'] = gdp_data['季度'].apply(parse_quarter)
                gdp_data = gdp_data.dropna(subset=['日期']).sort_values('日期')
                
                # 筛选日期范围
                start_dt = pd.to_datetime(start_date)
                end_dt = pd.to_datetime(end_date)
                gdp_data = gdp_data[(gdp_data['日期'] >= start_dt) & (gdp_data['日期'] <= end_dt)]
                
                # 重命名列
                gdp_data = gdp_data.rename(columns={
                    '国内生产总值': 'gdp',
                    '国内生产总值同比增长': 'gdp_yoy',
                    '国内生产总值环比增长': 'gdp_qoq'
                })
                
                self._log_step('get_gdp_data', {
                    'data_period': f"{start_date} to {end_date}",
                    'rows_obtained': len(gdp_data)
                })
            
            return gdp_data
            
        except Exception as e:
            logger.error(f"获取GDP数据失败: {str(e)}")
            return pd.DataFrame()
    
    def get_cpi_data(self, start_date: str, end_date: str) -> pd.DataFrame:
        """
        获取CPI数据
        
        参数:
            start_date: 开始日期
            end_date: 结束日期
            
        返回:
            pd.DataFrame: CPI数据
        """
        try:
            logger.info("获取CPI数据")
            
            # 使用akshare获取CPI数据
            cpi_data = ak.macro_china_cpi()
            
            if not cpi_data.empty:
                # 处理月份格式，转换为标准日期格式
                def parse_month(month_str):
                    try:
                        year = int(month_str[:4])
                        month = int(month_str[5:7])
                        return pd.Timestamp(year=year, month=month, day=1)
                    except:
                        return pd.NaT
                
                cpi_data['日期'] = cpi_data['月份'].apply(parse_month)
                cpi_data = cpi_data.dropna(subset=['日期']).sort_values('日期')
                
                # 筛选日期范围
                start_dt = pd.to_datetime(start_date)
                end_dt = pd.to_datetime(end_date)
                cpi_data = cpi_data[(cpi_data['日期'] >= start_dt) & (cpi_data['日期'] <= end_dt)]
                
                # 重命名列
                cpi_data = cpi_data.rename(columns={
                    '全国': 'cpi',
                    '全国同比增长': 'cpi_yoy'
                })
                
                self._log_step('get_cpi_data', {
                    'data_period': f"{start_date} to {end_date}",
                    'rows_obtained': len(cpi_data)
                })
            
            return cpi_data
            
        except Exception as e:
            logger.error(f"获取CPI数据失败: {str(e)}")
            return pd.DataFrame()
    
    def get_interest_rate_data(self, start_date: str, end_date: str) -> pd.DataFrame:
        """
        获取利率数据
        
        参数:
            start_date: 开始日期
            end_date: 结束日期
            
        返回:
            pd.DataFrame: 利率数据
        """
        try:
            logger.info("获取利率数据")
            
            # 使用akshare获取银行间拆借利率数据
            rate_data = ak.rate_interbank(market="上海银行同业拆借市场", symbol="Shibor人民币", indicator="1周")
            
            if not rate_data.empty:
                # 检查数据列结构，确保有日期列
                if '日期' not in rate_data.columns:
                    # 如果没有日期列，尝试从其他列推断
                    if 'date' in rate_data.columns:
                        rate_data['日期'] = pd.to_datetime(rate_data['date'])
                    elif 'time' in rate_data.columns:
                        rate_data['日期'] = pd.to_datetime(rate_data['time'])
                    else:
                        # 如果没有日期相关列，创建默认日期列
                        logger.warning("利率数据缺少日期列，创建默认日期范围")
                        rate_data['日期'] = pd.date_range(start='2020-01-01', periods=len(rate_data), freq='D')
                
                # 处理数据格式
                rate_data['日期'] = pd.to_datetime(rate_data['日期'])
                rate_data = rate_data.sort_values('日期')
                
                # 筛选日期范围
                start_dt = pd.to_datetime(start_date)
                end_dt = pd.to_datetime(end_date)
                rate_data = rate_data[(rate_data['日期'] >= start_dt) & (rate_data['日期'] <= end_dt)]
                
                # 检查可用的列并重命名
                rename_mapping = {}
                if '1周' in rate_data.columns:
                    rename_mapping['1周'] = 'shibor_1w'
                if '隔夜' in rate_data.columns:
                    rename_mapping['隔夜'] = 'shibor_overnight'
                if '2周' in rate_data.columns:
                    rename_mapping['2周'] = 'shibor_2w'
                if '1月' in rate_data.columns:
                    rename_mapping['1月'] = 'shibor_1m'
                if '3月' in rate_data.columns:
                    rename_mapping['3月'] = 'shibor_3m'
                if '6月' in rate_data.columns:
                    rename_mapping['6月'] = 'shibor_6m'
                if '9月' in rate_data.columns:
                    rename_mapping['9月'] = 'shibor_9m'
                if '1年' in rate_data.columns:
                    rename_mapping['1年'] = 'shibor_1y'
                
                if rename_mapping:
                    rate_data = rate_data.rename(columns=rename_mapping)
                
                self._log_step('get_interest_rate_data', {
                    'data_period': f"{start_date} to {end_date}",
                    'rows_obtained': len(rate_data)
                })
            
            return rate_data
            
        except Exception as e:
            logger.error(f"获取利率数据失败: {str(e)}")
            return pd.DataFrame()
    
    def get_m2_data(self, start_date: str, end_date: str) -> pd.DataFrame:
        """
        获取M2货币供应量数据
        
        参数:
            start_date: 开始日期
            end_date: 结束日期
            
        返回:
            pd.DataFrame: M2数据
        """
        try:
            logger.info("获取M2数据")
            
            # 使用akshare获取M2数据 - 正确的接口是macro_china_money_supply
            m2_data = ak.macro_china_money_supply()
            
            if not m2_data.empty:
                # 处理月份格式，转换为标准日期格式
                def parse_month(month_str):
                    try:
                        year = int(month_str[:4])
                        month = int(month_str[5:7])
                        return pd.Timestamp(year=year, month=month, day=1)
                    except:
                        return pd.NaT
                
                m2_data['日期'] = m2_data['月份'].apply(parse_month)
                m2_data = m2_data.dropna(subset=['日期']).sort_values('日期')
                
                # 筛选日期范围
                start_dt = pd.to_datetime(start_date)
                end_dt = pd.to_datetime(end_date)
                m2_data = m2_data[(m2_data['日期'] >= start_dt) & (m2_data['日期'] <= end_dt)]
                
                # 重命名列
                m2_data = m2_data.rename(columns={
                    '货币和准货币（M2）': 'm2',
                    '货币和准货币（M2）同比增长': 'm2_yoy'
                })
                
                self._log_step('get_m2_data', {
                    'data_period': f"{start_date} to {end_date}",
                    'rows_obtained': len(m2_data)
                })
            
            return m2_data
            
        except Exception as e:
            logger.error(f"获取M2数据失败: {str(e)}")
            return pd.DataFrame()
    
    def calculate_growth_factors(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        计算宏观指标增长率因子
        
        参数:
            df: 宏观数据DataFrame
            
        返回:
            pd.DataFrame: 包含增长率因子的数据
        """
        try:
            df = df.copy()
            
            # 确保数值列是数值类型
            for col in df.columns:
                if col not in ['日期', '季度', '月份']:
                    df[col] = pd.to_numeric(df[col], errors='coerce')
            
            # 计算各指标的同比增长率
            growth_columns = []
            for col in df.columns:
                if col not in ['日期', '季度', '月份']:
                    # 计算同比增长率
                    growth_col = f"{col}_yoy"
                    df[growth_col] = df[col].pct_change(periods=12, fill_method=None)  # 年度同比增长
                    growth_columns.append(growth_col)
            
            self._log_step('calculate_growth_factors', {
                'growth_columns_created': growth_columns
            })
            
            return df
            
        except Exception as e:
            logger.error(f"计算增长率因子失败: {str(e)}")
            raise
    
    def calculate_lag_factors(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        计算宏观指标滞后因子
        
        参数:
            df: 宏观数据DataFrame
            
        返回:
            pd.DataFrame: 包含滞后因子的数据
        """
        try:
            df = df.copy()
            
            # 确保lag_periods配置存在
            lag_periods = self.config.get('lag_periods', [1, 3, 6, 12])
            
            # 计算各指标的滞后值
            lag_columns = []
            for period in lag_periods:
                for col in df.columns:
                    if col not in ['日期', '季度', '月份'] and '_yoy' not in col:
                        # 计算滞后值
                        lag_col = f"{col}_lag_{period}m"
                        df[lag_col] = df[col].shift(period)
                        lag_columns.append(lag_col)
            
            self._log_step('calculate_lag_factors', {
                'lag_periods': lag_periods,
                'lag_columns_created': lag_columns
            })
            
            return df
            
        except Exception as e:
            logger.error(f"计算滞后因子失败: {str(e)}")
            raise
    
    def merge_macro_data(self, gdp_data: pd.DataFrame, cpi_data: pd.DataFrame, 
                        rate_data: pd.DataFrame, m2_data: pd.DataFrame) -> pd.DataFrame:
        """
        合并所有宏观数据
        
        参数:
            gdp_data: GDP数据
            cpi_data: CPI数据
            rate_data: 利率数据
            m2_data: M2数据
            
        返回:
            pd.DataFrame: 合并后的宏观数据
        """
        try:
            # 创建日期范围（月度频率）
            start_date = min([
                gdp_data['日期'].min() if not gdp_data.empty else pd.Timestamp('2020-01-01'),
                cpi_data['日期'].min() if not cpi_data.empty else pd.Timestamp('2020-01-01'),
                rate_data['日期'].min() if not rate_data.empty else pd.Timestamp('2020-01-01'),
                m2_data['日期'].min() if not m2_data.empty else pd.Timestamp('2020-01-01')
            ])
            
            end_date = max([
                gdp_data['日期'].max() if not gdp_data.empty else pd.Timestamp('2024-12-31'),
                cpi_data['日期'].max() if not cpi_data.empty else pd.Timestamp('2024-12-31'),
                rate_data['日期'].max() if not rate_data.empty else pd.Timestamp('2024-12-31'),
                m2_data['日期'].max() if not m2_data.empty else pd.Timestamp('2024-12-31')
            ])
            
            # 创建月度日期索引
            date_range = pd.date_range(start=start_date, end=end_date, freq='M')
            macro_df = pd.DataFrame({'日期': date_range})
            
            # 合并各宏观数据
            datasets = [gdp_data, cpi_data, rate_data, m2_data]
            for dataset in datasets:
                if not dataset.empty:
                    # 前向填充确保数据连续性
                    macro_df = pd.merge_asof(macro_df.sort_values('日期'), 
                                           dataset.sort_values('日期'), 
                                           on='日期', direction='forward')
            
            self._log_step('merge_macro_data', {
                'final_date_range': f"{macro_df['日期'].min()} to {macro_df['日期'].max()}",
                'final_columns': macro_df.columns.tolist(),
                'final_rows': len(macro_df)
            })
            
            return macro_df

        except Exception as e:
            logger.error(f"合并宏观数据失败: {str(e)}")
            raise

    def process_all_factors(
        self,
        start_date: str,
        end_date: str,
        trading_calendar: Optional[Sequence[pd.Timestamp]] = None
    ) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        批量计算所有宏观因子
        
        参数:
            start_date: 开始日期
            end_date: 结束日期
            trading_calendar: 交易日历，可为空

        返回:
            Tuple[pd.DataFrame, pd.DataFrame]: (月度宏观数据, 日度插值数据)
        """
        try:
            logger.info("开始计算宏观因子...")

            # 获取各宏观数据
            gdp_data = self.get_gdp_data(start_date, end_date)
            cpi_data = self.get_cpi_data(start_date, end_date)
            rate_data = self.get_interest_rate_data(start_date, end_date)
            m2_data = self.get_m2_data(start_date, end_date)
            
            # 合并宏观数据
            macro_df = self.merge_macro_data(gdp_data, cpi_data, rate_data, m2_data)
            
            if macro_df.empty:
                logger.warning("未获取到宏观数据，返回空DataFrame")
                return macro_df, pd.DataFrame()
            
            # 计算增长率因子
            macro_df = self.calculate_growth_factors(macro_df)
            
            # 计算滞后因子
            macro_df = self.calculate_lag_factors(macro_df)
            
            # 生成日频插值数据
            value_columns = [
                col for col in macro_df.columns
                if col not in {"日期", "季度", "月份"}
            ]
            interpolation_config = InterpolationConfig(
                method="linear",
                limit_direction="both",
                fill_strategy="both",
                preserve_monthly=True
            )
            macro_daily_df = convert_monthly_to_daily(
                monthly_df=macro_df,
                date_column="日期",
                value_columns=value_columns,
                daily_calendar=trading_calendar,
                config=interpolation_config
            )

            self.processing_log['daily_rows'] = str(len(macro_daily_df))
            self.processing_log['daily_columns'] = macro_daily_df.columns.tolist()

            # 完成处理记录
            self.processing_log['end_time'] = datetime.now().isoformat()
            self.processing_log['total_rows'] = str(len(macro_df))
            self.processing_log['factors_calculated'] = macro_df.columns.tolist()

            logger.info("宏观因子计算完成！")
            return macro_df, macro_daily_df
            
        except Exception as e:
            logger.error(f"批量计算宏观因子失败: {str(e)}")
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
        'data_sources': ['akshare'],
        'macro_indicators': ['GDP', 'CPI', '利率', 'M2'],
        'lag_periods': [1, 3, 6, 12]
    }
    
    # 创建宏观因子计算器
    calculator = MacroFactors(config)
    
    try:
        # 计算宏观因子
        start_date = "2020-01-01"
        end_date = "2024-12-31"
        macro_factors = calculator.process_all_factors(start_date, end_date)
        
        if not macro_factors.empty:
            # 保存结果
            output_file = "data_pipeline/data/features/macro_factors.csv"
            macro_factors.to_csv(output_file, index=False, encoding='utf-8-sig')
            
            # 保存处理日志
            log_file = "data_pipeline/data/logs/macro_factors_processing_log.json"
            calculator.save_processing_log(log_file)
            
            logger.info(f"宏观因子计算完成！结果已保存到: {output_file}")
        else:
            logger.warning("未生成宏观因子数据")
        
    except Exception as e:
        logger.error(f"宏观因子计算失败: {str(e)}")


if __name__ == "__main__":
    main()