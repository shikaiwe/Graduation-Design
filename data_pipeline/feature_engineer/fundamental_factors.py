#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
基本因子计算模块
实现市盈率、市净率、ROE、ROA、营收增长率等基本因子的计算
"""

import pandas as pd
import numpy as np
import logging
from typing import Dict, Any, Optional
import json
from datetime import datetime
import akshare as ak

# 配置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class FundamentalFactors:
    """基本因子计算器"""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        初始化基本因子计算器
        
        参数:
            config: 配置参数字典
        """
        self.config = config or {
            'data_source': 'akshare',  # 数据源选择
            'fiscal_periods': [1, 4],  # 财务数据周期（季度、年度）
            'growth_periods': [1, 4, 8]  # 增长率计算周期
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
    
    def get_financial_data(self, stock_codes: list, start_date: str, end_date: str) -> pd.DataFrame:
        """
        获取财务数据（市盈率、市净率等）
        
        参数:
            stock_codes: 股票代码列表
            start_date: 开始日期
            end_date: 结束日期
            
        返回:
            pd.DataFrame: 财务数据
        """
        try:
            logger.info(f"开始获取{len(stock_codes)}只股票的财务数据")
            
            financial_data = []
            
            for stock_code in stock_codes:
                try:
                    stock_code_str = str(stock_code).zfill(6)
                    
                    # 获取资产负债表数据
                    balance_sheet = ak.stock_financial_debt_ths(symbol=stock_code_str, indicator="按报告期")
                    
                    # 获取利润表数据
                    income_statement = ak.stock_financial_benefit_ths(symbol=stock_code_str, indicator="按报告期")
                    
                    # 获取现金流量表数据
                    cash_flow = ak.stock_financial_cash_ths(symbol=stock_code_str, indicator="按报告期")
                    
                    # 合并财务数据
                    if not balance_sheet.empty and not income_statement.empty:
                        # 提取关键财务指标
                        fin_data = pd.DataFrame()
                        
                        # 处理日期列
                        if '报告期' in balance_sheet.columns:
                            fin_data['日期'] = pd.to_datetime(balance_sheet['报告期'])
                        
                        # 添加股票代码
                        fin_data['股票代码'] = stock_code
                        
                        # 从资产负债表提取数据
                        if '所有者权益（或股东权益）合计' in balance_sheet.columns:
                            fin_data['所有者权益'] = balance_sheet['所有者权益（或股东权益）合计']
                        
                        if '负债和所有者权益（或股东权益）合计' in balance_sheet.columns:
                            fin_data['总资产'] = balance_sheet['负债和所有者权益（或股东权益）合计']
                        
                        # 从利润表提取数据
                        if '*净利润' in income_statement.columns:
                            fin_data['净利润'] = income_statement['*净利润']
                        
                        if '营业总收入' in income_statement.columns:
                            fin_data['营业总收入'] = income_statement['营业总收入']
                        
                        # 计算基本财务指标
                        if not fin_data.empty:
                            # 计算ROE（净资产收益率）
                            if '净利润' in fin_data.columns and '所有者权益' in fin_data.columns:
                                fin_data['净资产收益率'] = fin_data['净利润'] / fin_data['所有者权益']
                            
                            # 计算ROA（总资产报酬率）
                            if '净利润' in fin_data.columns and '总资产' in fin_data.columns:
                                fin_data['总资产报酬率'] = fin_data['净利润'] / fin_data['总资产']
                            
                            # 计算营收增长率
                            if '营业总收入' in fin_data.columns:
                                fin_data['营业总收入同比增长率'] = fin_data['营业总收入'].pct_change()
                            
                            # 计算净利润增长率
                            if '净利润' in fin_data.columns:
                                fin_data['净利润同比增长率'] = fin_data['净利润'].pct_change()
                            
                            financial_data.append(fin_data)
                        else:
                            logger.warning(f"股票{stock_code}财务数据提取失败")
                    
                except Exception as e:
                    logger.warning(f"获取股票{stock_code}财务数据失败: {str(e)}")
                    continue
            
            if financial_data:
                df = pd.concat(financial_data, ignore_index=True)
                df['日期'] = pd.to_datetime(df['日期'])
                df = df.sort_values(['日期', '股票代码'])
                
                # 筛选日期范围
                start_dt = pd.to_datetime(start_date)
                end_dt = pd.to_datetime(end_date)
                df = df[(df['日期'] >= start_dt) & (df['日期'] <= end_dt)]
                
                self._log_step('get_financial_data', {
                    'stock_count': len(stock_codes),
                    'data_period': f"{start_date} to {end_date}",
                    'rows_obtained': len(df),
                    'available_indicators': df.columns.tolist()
                })
                
                return df
            else:
                logger.warning("未获取到任何财务数据")
                return pd.DataFrame()
                
        except Exception as e:
            logger.error(f"获取财务数据失败: {str(e)}")
            raise
    
    def calculate_pe_ratio(self, df: pd.DataFrame, price_data: pd.DataFrame) -> pd.DataFrame:
        """
        计算市盈率相关因子
        
        参数:
            df: 包含财务数据的DataFrame
            price_data: 包含股价数据的DataFrame
            
        返回:
            pd.DataFrame: 包含市盈率因子的数据
        """
        try:
            df = df.copy()
            
            # 合并股价数据来计算市盈率
            if not price_data.empty and '日期' in price_data.columns and '收盘价' in price_data.columns:
                # 获取最新的股价数据
                latest_prices = price_data.groupby('股票代码').last().reset_index()[['股票代码', '收盘价']]
                
                # 合并股价数据
                df_with_price = pd.merge(df, latest_prices, on='股票代码', how='left')
                
                # 计算市盈率（假设每股收益 = 净利润 / 总股本，这里简化处理）
                if '净利润' in df_with_price.columns and '收盘价' in df_with_price.columns:
                    # 简化计算：市盈率 = 股价 / (净利润 / 总股本)，这里假设总股本为1亿
                    df_with_price['市盈率'] = df_with_price['收盘价'] / (df_with_price['净利润'] / 100000000)
                    
                    # 计算市盈率分位数（行业内）
                    df_with_price['pe_ratio_quantile'] = df_with_price.groupby('日期')['市盈率'].transform(
                        lambda x: pd.qcut(x, 5, labels=False, duplicates='drop')
                    )
                    
                    # 计算市盈率变化率
                    df_with_price['pe_ratio_change'] = df_with_price.groupby('股票代码')['市盈率'].pct_change()
                    
                    df = df_with_price
            
            self._log_step('calculate_pe_ratio', {
                'pe_ratio_available': '市盈率' in df.columns
            })
            
            return df
            
        except Exception as e:
            logger.error(f"计算市盈率因子失败: {str(e)}")
            raise
    
    def calculate_pb_ratio(self, df: pd.DataFrame, price_data: pd.DataFrame) -> pd.DataFrame:
        """
        计算市净率相关因子
        
        参数:
            df: 包含财务数据的DataFrame
            price_data: 包含股价数据的DataFrame
            
        返回:
            pd.DataFrame: 包含市净率因子的数据
        """
        try:
            df = df.copy()
            
            # 合并股价数据来计算市净率
            if not price_data.empty and '日期' in price_data.columns and '收盘价' in price_data.columns:
                # 获取最新的股价数据
                latest_prices = price_data.groupby('股票代码').last().reset_index()[['股票代码', '收盘价']]
                
                # 合并股价数据
                df_with_price = pd.merge(df, latest_prices, on='股票代码', how='left')
                
                # 计算市净率（市净率 = 股价 / 每股净资产）
                if '所有者权益' in df_with_price.columns and '收盘价' in df_with_price.columns:
                    # 简化计算：市净率 = 股价 / (所有者权益 / 总股本)，这里假设总股本为1亿
                    df_with_price['市净率'] = df_with_price['收盘价'] / (df_with_price['所有者权益'] / 100000000)
                    
                    # 计算市净率分位数
                    df_with_price['pb_ratio_quantile'] = df_with_price.groupby('日期')['市净率'].transform(
                        lambda x: pd.qcut(x, 5, labels=False, duplicates='drop')
                    )
                    
                    # 计算市净率变化率
                    df_with_price['pb_ratio_change'] = df_with_price.groupby('股票代码')['市净率'].pct_change()
                    
                    df = df_with_price
            
            self._log_step('calculate_pb_ratio', {
                'pb_ratio_available': '市净率' in df.columns
            })
            
            return df
            
        except Exception as e:
            logger.error(f"计算市净率因子失败: {str(e)}")
            raise
    
    def calculate_roe_roa(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        计算ROE和ROA相关因子
        
        参数:
            df: 包含财务数据的DataFrame
            
        返回:
            pd.DataFrame: 包含ROE/ROA因子的数据
        """
        try:
            df = df.copy()
            
            if '净资产收益率' in df.columns:
                # ROE分位数
                df['roe_quantile'] = df.groupby('日期')['净资产收益率'].transform(
                    lambda x: pd.qcut(x, 5, labels=False, duplicates='drop')
                )
                
                # ROE变化率
                df['roe_change'] = df.groupby('股票代码')['净资产收益率'].pct_change()
            
            if '总资产报酬率' in df.columns:
                # ROA分位数
                df['roa_quantile'] = df.groupby('日期')['总资产报酬率'].transform(
                    lambda x: pd.qcut(x, 5, labels=False, duplicates='drop')
                )
                
                # ROA变化率
                df['roa_change'] = df.groupby('股票代码')['总资产报酬率'].pct_change()
            
            self._log_step('calculate_roe_roa', {
                'roe_available': '净资产收益率' in df.columns,
                'roa_available': '总资产报酬率' in df.columns
            })
            
            return df
            
        except Exception as e:
            logger.error(f"计算ROE/ROA因子失败: {str(e)}")
            raise
    
    def calculate_growth_rates(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        计算营收和利润增长率因子
        
        参数:
            df: 包含财务数据的DataFrame
            
        返回:
            pd.DataFrame: 包含增长率因子的数据
        """
        try:
            df = df.copy()
            
            if '营业总收入同比增长率' in df.columns:
                # 营收增长率分位数
                df['revenue_growth_quantile'] = df.groupby('日期')['营业总收入同比增长率'].transform(
                    lambda x: pd.qcut(x, 5, labels=False, duplicates='drop')
                )
            
            if '净利润同比增长率' in df.columns:
                # 利润增长率分位数
                df['profit_growth_quantile'] = df.groupby('日期')['净利润同比增长率'].transform(
                    lambda x: pd.qcut(x, 5, labels=False, duplicates='drop')
                )
            
            self._log_step('calculate_growth_rates', {
                'revenue_growth_available': '营业总收入同比增长率' in df.columns,
                'profit_growth_available': '净利润同比增长率' in df.columns
            })
            
            return df
            
        except Exception as e:
            logger.error(f"计算增长率因子失败: {str(e)}")
            raise
    
    def process_all_factors(self, stock_codes: list, start_date: str, end_date: str, price_data: pd.DataFrame = None) -> pd.DataFrame:
        """
        批量计算所有基本因子
        
        参数:
            stock_codes: 股票代码列表
            start_date: 开始日期
            end_date: 结束日期
            price_data: 股价数据（用于计算市盈率、市净率）
            
        返回:
            pd.DataFrame: 包含所有基本因子的数据
        """
        try:
            logger.info("开始计算基本因子...")
            
            # 获取财务数据
            df = self.get_financial_data(stock_codes, start_date, end_date)
            
            if df.empty:
                logger.warning("未获取到财务数据，返回空DataFrame")
                return df
            
            # 计算市盈率因子（需要股价数据）
            df = self.calculate_pe_ratio(df, price_data)
            
            # 计算市净率因子（需要股价数据）
            df = self.calculate_pb_ratio(df, price_data)
            
            # 计算ROE/ROA因子
            df = self.calculate_roe_roa(df)
            
            # 计算增长率因子
            df = self.calculate_growth_rates(df)
            
            # 完成处理记录
            self.processing_log['end_time'] = datetime.now().isoformat()
            self.processing_log['total_rows'] = str(len(df))
            self.processing_log['factors_calculated'] = [
                col for col in df.columns if col not in ['日期', '股票代码']
            ]
            
            logger.info("基本因子计算完成！")
            return df
            
        except Exception as e:
            logger.error(f"批量计算基本因子失败: {str(e)}")
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
        'data_source': 'akshare'
    }
    
    # 创建基本因子计算器
    calculator = FundamentalFactors(config)
    
    try:
        # 读取沪深300成分股
        components_file = "data_pipeline/data/components/hs300_components_full.csv"
        components_df = pd.read_csv(components_file)
        stock_codes = components_df['股票代码'].tolist()[:10]  # 测试用前10只股票
        
        # 计算基本因子
        start_date = "2020-01-01"
        end_date = "2024-12-31"
        df_factors = calculator.process_all_factors(stock_codes, start_date, end_date)
        
        if not df_factors.empty:
            # 保存结果
            output_file = "data_pipeline/data/features/fundamental_factors.csv"
            df_factors.to_csv(output_file, index=False, encoding='utf-8-sig')
            
            # 保存处理日志
            log_file = "data_pipeline/data/logs/fundamental_factors_processing_log.json"
            calculator.save_processing_log(log_file)
            
            logger.info(f"基本因子计算完成！结果已保存到: {output_file}")
        else:
            logger.warning("未生成基本因子数据")
        
    except Exception as e:
        logger.error(f"基本因子计算失败: {str(e)}")


if __name__ == "__main__":
    main()