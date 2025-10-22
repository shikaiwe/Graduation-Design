#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
基本因子计算模块
实现市盈率、市净率、ROE、ROA、营收增长率等基本因子的计算
"""

import pandas as pd
import numpy as np
from tqdm import tqdm
import logging
from typing import Dict, Any, Optional
import json
from datetime import datetime
import akshare as ak
import os
from pathlib import Path

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
            'growth_periods': [1, 4, 8],  # 增长率计算周期
            'output_base_dir': Path('data_pipeline/data/features')
        }
        
        self.processing_log = {
            'start_time': datetime.now().isoformat(),
            'config': self.config,
            'steps': []
        }
        self.processed_dates = set()
        self._load_processed_dates()
    
    def _load_processed_dates(self):
        """
        检查已保存的批次文件，加载已处理的报告期日期。
        """
        batches_dir = self.config['output_base_dir'] / 'fundamental_factors_batches'
        if not batches_dir.exists():
            logger.info("因子数据批处理目录不存在，无需加载已处理日期。")
            return

        try:
            data_files = sorted(batches_dir.glob("fundamental_factors_batch_*.csv"))

            if not data_files:
                logger.info("未找到已处理的因子数据文件。")
                return

            for file_path in data_files:
                try:
                    # 从文件名中提取日期
                    date_str = file_path.stem.replace('fundamental_factors_batch_', '')
                    self.processed_dates.add(date_str)
                except Exception as e:
                    logger.warning(f"从文件名 {file_path} 提取日期失败: {e}")
            
            logger.info(f"已加载 {len(self.processed_dates)} 个已处理的报告期。")

        except Exception as e:
            logger.error(f"加载已处理报告期失败: {e}")
    
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
        使用东方财富数据源，按报告期批量获取数据，更稳定。
        
        参数:
            stock_codes: 股票代码列表
            start_date: 开始日期
            end_date: 结束日期
            
        返回:
            pd.DataFrame: 财务数据
        """
        try:
            logger.info(f"开始使用东方财富数据源获取{len(stock_codes)}只股票的财务数据")
            
            # 1. 生成报告期日期列表并过滤已处理的日期
            all_report_dates = pd.date_range(start=start_date, end=end_date, freq='Q-DEC').strftime('%Y%m%d').tolist()
            report_dates_to_fetch = [date for date in all_report_dates if date not in self.processed_dates]
            
            logger.info(f"共需处理 {len(all_report_dates)} 个报告期，其中 {len(report_dates_to_fetch)} 个需要新获取。")

            batches_dir = self.config['output_base_dir'] / 'fundamental_factors_batches'
            batches_dir.mkdir(parents=True, exist_ok=True)

            # 2. 按需遍历，获取并保存新数据
            for date in tqdm(report_dates_to_fetch, desc="获取并保存各报告期财务数据"):
                try:
                    zcfz_df = ak.stock_zcfz_em(date=date)
                    lrb_df = ak.stock_lrb_em(date=date)
                    
                    zcfz_subset = zcfz_df[['股票代码', '股东权益合计', '资产-总资产']]
                    lrb_cols = ['股票代码', '净利润', '营业总收入']
                    if '基本每股收益' in lrb_df.columns:
                        lrb_cols.append('基本每股收益')
                    lrb_subset = lrb_df[lrb_cols]
                    
                    merged_df = pd.merge(zcfz_subset, lrb_subset, on='股票代码', how='inner')
                    merged_df['日期'] = pd.to_datetime(date, format='%Y%m%d')
                    # 公告日期：优先取利润表的公告日期，其次资产负债表
                    announce_col = None
                    for c in ['公告日期', '公告日']:
                        if c in lrb_df.columns:
                            announce_col = c
                            break
                    if announce_col is None:
                        for c in ['公告日期', '公告日']:
                            if c in zcfz_df.columns:
                                announce_col = c
                                break
                    if announce_col is not None:
                        code_announce = lrb_df[['股票代码', announce_col]] if announce_col in lrb_df.columns else zcfz_df[['股票代码', announce_col]]
                        code_announce = code_announce.rename(columns={announce_col: '公告日期'})
                        merged_df = pd.merge(merged_df, code_announce, on='股票代码', how='left')
                        merged_df['公告日期'] = pd.to_datetime(merged_df['公告日期'], errors='coerce')
                    
                    # 保存当期数据到批次文件
                    batch_file_path = batches_dir / f"fundamental_factors_batch_{date}.csv"
                    merged_df.to_csv(batch_file_path, index=False, encoding='utf-8-sig')
                    logger.info(f"已成功获取并保存报告期 {date} 的数据到 {batch_file_path}")
                    
                except Exception as e:
                    logger.warning(f"获取或保存报告期 {date} 的财务数据失败: {e}")
                    continue
            
            # 3. 从本地加载所有需要的批次文件
            all_financial_data = []
            for date in all_report_dates:
                batch_file_path = batches_dir / f"fundamental_factors_batch_{date}.csv"
                if batch_file_path.exists():
                    try:
                        batch_df = pd.read_csv(batch_file_path)
                        batch_df['日期'] = pd.to_datetime(date, format='%Y%m%d')
                        all_financial_data.append(batch_df)
                    except Exception as e:
                        logger.warning(f"加载批次文件 {batch_file_path} 失败: {e}")
                else:
                    logger.warning(f"预期的批次文件不存在: {batch_file_path}")

            if not all_financial_data:
                logger.warning("在指定日期范围内未能加载任何本地财务数据")
                return pd.DataFrame()
                
            # 4. 合并所有报告期的数据并筛选
            df = pd.concat(all_financial_data, ignore_index=True)
            df['日期'] = pd.to_datetime(df['日期'])

            # 确保用于筛选的股票代码和DataFrame中的股票代码都是字符串类型，避免不匹配
            df['股票代码'] = df['股票代码'].astype(str)
            stock_codes_str = [str(code) for code in stock_codes]

            df = df[df['股票代码'].isin(stock_codes_str)].copy()

            if df.empty:
                logger.warning("指定股票在指定日期范围内未找到财务数据")
                return pd.DataFrame()

            # 5. 重命名和数据类型转换
            df.rename(columns={
                '股东权益合计': '所有者权益',
                '资产-总资产': '总资产',
            }, inplace=True)
            
            numeric_cols = ['所有者权益', '总资产', '净利润', '营业总收入']
            for col in numeric_cols:
                df[col] = pd.to_numeric(df[col], errors='coerce')

            # 6. 排序、填充、清理
            df.sort_values(['股票代码', '日期'], inplace=True)
            
            # 关键指标全部为空的行才删除
            df.dropna(subset=['所有者权益', '总资产', '净利润'], how='all', inplace=True)

            # 按股票分组，用前向填充处理缺失值
            df[numeric_cols] = df.groupby('股票代码')[numeric_cols].transform(lambda x: x.ffill())
            
            # 再次清理可能存在的无法填充的行
            df.dropna(subset=['所有者权益', '总资产', '净利润'], how='all', inplace=True)

            # 7. 计算衍生指标
            # 计算ROE和ROA，处理分母为0的情况
            df['净资产收益率'] = np.where(df['所有者权益'] != 0, df['净利润'] / df['所有者权益'], np.nan)
            df['总资产报酬率'] = np.where(df['总资产'] != 0, df['净利润'] / df['总资产'], np.nan)
            
            # 按股票分组计算同比增长率
            df['营业总收入同比增长率'] = df.groupby('股票代码')['营业总收入'].pct_change(fill_method=None)
            df['净利润同比增长率'] = df.groupby('股票代码')['净利润'].pct_change(fill_method=None)

            # 替换无穷大值为NaN
            df.replace([np.inf, -np.inf], np.nan, inplace=True)

            # 8. 计算估值指标：PE与PE_TTM（基于真实数据，无估算）
            df = self.calculate_valuation_metrics(df)
            
            self._log_step('get_financial_data', {
                'stock_count': len(stock_codes),
                'data_period': f"{start_date} to {end_date}",
                'rows_obtained': len(df),
                'available_indicators': df.columns.tolist()
            })
            
            # 8. 保存最终整合的数据
            final_output_path = self.config['output_base_dir'] / 'fundamental_factors.csv'
            final_output_path.parent.mkdir(parents=True, exist_ok=True)
            
            df.to_csv(final_output_path, index=False, encoding='utf-8-sig')
            logger.info(f"最终的基本面因子数据已保存到 {final_output_path}")

            return df.reset_index(drop=True)
                
        except Exception as e:
            logger.error(f"获取财务数据过程中发生未知错误: {str(e)}")
            raise
    
    def calculate_valuation_metrics(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        计算估值指标（基于真实、可验证数据，报告期末口径）：
        - PE = 报告期末收盘价 / 基本每股收益（若EPS<=0或缺失则NaN）
        - PE_TTM = 报告期末总市值 / 最近12个月净利润（若任一缺失或TTM<=0则NaN）
        数据与对齐：
        - 报告期末价格：使用报告期末日或之后首个交易日的“收盘价”
        - 基本每股收益：利润表 '基本每股收益'（若缺失，不估算）
        - 净利润TTM：按报告期“净利润”滚动四期求和（需满4期）
        - 总市值：仅在历史口径可得时使用；当前管线若无法获取期末总市值，则PE_TTM为NaN并记录说明
        """
        try:
            df = df.copy()
            # 计算净利润TTM（按股票代码滚动四期）
            df = df.sort_values(['股票代码', '日期'])
            if '净利润' in df.columns:
                df['净利润_TTM'] = df.groupby('股票代码')['净利润'].rolling(window=4, min_periods=4).sum().reset_index(level=0, drop=True)
            else:
                df['净利润_TTM'] = np.nan

            # 准备获取报告期末收盘价：对每支股票的所有报告期末日期，抓取该日期±3交易日内数据，取>=报告期末的首个交易日收盘
            unique_codes = df['股票代码'].dropna().unique().tolist()
            report_dates_map = df.groupby('股票代码')['日期'].unique().to_dict()

            close_records = []
            for code in unique_codes:
                dates = sorted(pd.to_datetime(report_dates_map.get(code, [])))
                if not dates:
                    continue
                # 将日期窗口合并为一次最小范围抓取以减少请求量
                start_dt = (min(dates) - pd.Timedelta(days=5)).strftime('%Y%m%d')
                end_dt = (max(dates) + pd.Timedelta(days=5)).strftime('%Y%m%d')
                try:
                    # symbol需含交易所后缀，简单处理：上证以'600/601/603/605'为sh，深证以'000/001/002/003/300'为sz
                    sym = code
                    if code.startswith(('600', '601', '603', '605')):
                        symbol = f"sh{code}"
                    else:
                        symbol = f"sz{code}"
                    kline = ak.stock_zh_a_hist(symbol=symbol, period='daily', start_date=start_dt, end_date=end_dt, adjust='qfq')
                    if kline is None or kline.empty:
                        continue
                    # 规范列
                    if '日期' not in kline.columns and '日期' in kline.rename(columns={'交易日期':'日期'}).columns:
                        kline = kline.rename(columns={'交易日期': '日期'})
                    kline['日期'] = pd.to_datetime(kline['日期'])
                    price_col = '收盘'
                    if price_col not in kline.columns:
                        # 兼容英文列
                        if 'close' in kline.columns:
                            price_col = 'close'
                    for rd in dates:
                        # 找到>=报告期末的首个交易日
                        sub = kline[kline['日期'] >= rd].sort_values('日期').head(1)
                        if not sub.empty and price_col in sub.columns:
                            close_price = pd.to_numeric(sub.iloc[0][price_col], errors='coerce')
                            close_date = sub.iloc[0]['日期']
                            close_records.append((code, rd, close_date, close_price))
                except Exception:
                    continue
            if close_records:
                close_df = pd.DataFrame(close_records, columns=['股票代码', '日期', '对齐交易日', '报告期末收盘价'])
                df = pd.merge(df, close_df, on=['股票代码', '日期'], how='left')
            else:
                df['报告期末收盘价'] = np.nan

            # 计算PE（报告期末收盘价 / 基本每股收益）
            if '基本每股收益' in df.columns and '报告期末收盘价' in df.columns:
                df['PE'] = np.where((df['基本每股收益'] > 0) & (~df['基本每股收益'].isna()) & (~df['报告期末收盘价'].isna()),
                                  df['报告期末收盘价'] / df['基本每股收益'], np.nan)
            else:
                df['PE'] = np.nan

            # 计算PE_TTM（总市值 / 净利润_TTM）
            # 历史口径总市值若缺失，则置NaN并在口径说明中注明
            if '总市值' in df.columns:
                df['PE_TTM'] = np.where((df['净利润_TTM'] > 0) & (~df['净利润_TTM'].isna()) & (~df['总市值'].isna()),
                                        df['总市值'] / df['净利润_TTM'], np.nan)
            else:
                df['PE_TTM'] = np.nan

            # 透明与可验证：标注口径
            df['估值口径说明'] = (
                np.where(~df['PE'].isna(), 'PE=报告期末收盘价/基本每股收益; ', '') +
                np.where(~df['PE_TTM'].isna(), 'PE_TTM=报告期末总市值/净利润TTM; ', '') +
                np.where(df['PE_TTM'].isna(), 'PE_TTM缺失：期末总市值不可得; ', '')
            )

            return df
        except Exception as e:
            logger.warning(f"估值指标计算失败: {e}")
            return df
    
    def calculate_pe_ratio(self, df: pd.DataFrame, price_data: pd.DataFrame) -> pd.DataFrame:
        """
        [Deprecated] 旧版市盈率计算，保留以兼容历史调用，已不再推荐使用。
        新版估值计算请使用 calculate_valuation_metrics。
        """
        try:
            logger.warning("calculate_pe_ratio 已废弃，请使用 calculate_valuation_metrics。")
            return df.copy()
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
        start_date = "2019-01-01"
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