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
            from tqdm import tqdm
            
            logger.info("开始获取各报告期财务数据...")
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
            logger.info("开始加载本地批次文件...")
            all_financial_data = []
            for date in tqdm(all_report_dates, desc="加载批次文件"):
                batch_file_path = batches_dir / f"fundamental_factors_batch_{date}.csv"
                if batch_file_path.exists():
                    try:
                        batch_df = pd.read_csv(batch_file_path, encoding='utf-8-sig')
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

            # 确保用于筛选的股票代码和DataFrame中的股票代码都是6位字符串（左侧补零），避免不匹配
            df['股票代码'] = df['股票代码'].astype(str).str.strip().str.zfill(6)
            stock_codes_str = [str(code).zfill(6) for code in stock_codes]

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

            # 准备获取报告期末收盘价：使用本地股价数据文件
            unique_codes = [str(c).zfill(6) for c in df['股票代码'].dropna().unique().tolist()]
            report_dates_map = df.groupby('股票代码')['日期'].unique().to_dict()

            close_records = []
            
            # 加载本地股价数据文件（优先使用项目内数据路径，其次尝试备用路径）
            try:
                primary_path = Path("data_pipeline/data/daily_prices/Merge/hs300_daily_prices_merged.csv")
                fallback_path = Path("e:/Design/Graduation-Design/data_pipeline/data/daily_prices/Merge/hs300_daily_prices_merged.csv")
                price_file_path = primary_path if primary_path.exists() else fallback_path
                price_data = pd.read_csv(price_file_path, encoding='utf-8-sig')
                price_data['date'] = pd.to_datetime(price_data['date'])
                # 将股票代码标准化为6位字符串，左侧补零，确保与基础数据对齐
                price_data['股票代码'] = price_data['股票代码'].astype(str).str.strip().str.zfill(6)
                logger.info(f"成功加载本地股价数据，共{len(price_data)}行，来源: {price_file_path}")
                # 预计算每只股票的首个交易日，用于识别上市前报告期
                first_trade_date_per_code = price_data.groupby('股票代码')['date'].min().to_dict()
            except Exception as e:
                logger.error(f"加载本地股价数据文件失败: {e}")
                price_data = pd.DataFrame()

            logger.info("开始处理股价数据...")
            for code in tqdm(unique_codes, desc="处理股价数据"):
                dates = sorted(pd.to_datetime(report_dates_map.get(code, [])))
                if not dates:
                    continue
                
                # 为每个报告期单独获取股价数据，确保时间精确对齐
                for rd in dates:
                    try:
                        # 从本地数据中查找该股票在报告期后的首个交易日收盘价
                        stock_price_data = price_data[price_data['股票代码'] == str(code)]
                        if stock_price_data.empty:
                            logger.warning(f"股票{code}在本地股价数据中未找到")
                            continue
                        
                        # 上市首日（用于判断报告期是否早于上市）
                        first_trade_date = first_trade_date_per_code.get(str(code))
                        if pd.notna(first_trade_date) and rd < first_trade_date:
                            # 报告期早于上市日期：此类报告期没有可对齐的收盘价，跳过并提示一次
                            logger.debug(f"股票{code}报告期{rd}早于上市首日{first_trade_date}，跳过对齐")
                            continue
                        
                        # 找到>=报告期末的首个交易日
                        sub = stock_price_data[stock_price_data['date'] >= rd].sort_values('date').head(1)
                        if not sub.empty and 'close' in sub.columns:
                            close_price = pd.to_numeric(sub.iloc[0]['close'], errors='coerce')
                            close_date = sub.iloc[0]['date']
                            
                            # 检查时间对齐是否合理（报告期与交易日差距不应过大）
                            days_diff = (close_date - rd).days
                            if days_diff > 90:  # 放宽到90天，同时对极端差距打印warning
                                logger.warning(f"股票{code}报告期{rd}与交易日{close_date}时间差距过大: {days_diff}天")
                                continue
                                
                            close_records.append((code, rd, close_date, close_price))
                            logger.info(f"股票{code}报告期{rd}的收盘价: {close_price} (交易日: {close_date})")
                        else:
                            logger.warning(f"股票{code}报告期{rd}未找到合适的收盘价数据")
                    except Exception as e:
                        logger.warning(f"获取股票{code}报告期{rd}的股价数据失败: {e}")
                        continue
            if close_records:
                close_df = pd.DataFrame(close_records, columns=['股票代码', '日期', '对齐交易日', '报告期末收盘价'])
                # 确保合并前检查数据完整性
                if not df.empty:
                    df = pd.merge(df, close_df, on=['股票代码', '日期'], how='left')
                else:
                    df['报告期末收盘价'] = np.nan
            else:
                df['报告期末收盘价'] = np.nan

            # 计算PE（报告期末收盘价 / 基本每股收益）
            if '基本每股收益' in df.columns and '报告期末收盘价' in df.columns:
                df['PE'] = np.where((df['基本每股收益'] > 0) & (~df['基本每股收益'].isna()) & (~df['报告期末收盘价'].isna()),
                                  df['报告期末收盘价'] / df['基本每股收益'], np.nan)
            else:
                df['PE'] = np.nan

            # 使用东方财富接口获取的总市值数据
            if '总市值_em' in df.columns:
                df['总市值'] = df['总市值_em'].copy()
                logger.info(f"使用东方财富接口总市值数据，共{len(df[~df['总市值'].isna()])}条有效记录")
            else:
                df['总市值'] = np.nan
                logger.warning("总市值_em列不存在，总市值设置为NaN")
            
            # 使用东方财富stock_value_em接口获取估值数据
            def get_stock_value_data(code):
                """获取个股估值数据
                
                Args:
                    code: 股票代码
                    
                Returns:
                    pd.DataFrame: 估值数据DataFrame
                """
                try:
                    # 使用东方财富接口获取个股估值数据
                    value_data = ak.stock_value_em(symbol=code)
                    
                    if not value_data.empty:
                        # 规范化列名：去空格、统一中英文括号、半角/全角，提升兼容性
                        normalized_cols = []
                        for col in value_data.columns:
                            c = str(col)
                            c = c.replace('（', '(').replace('）', ')')
                            c = c.replace(' ', '')
                            normalized_cols.append(c)
                        value_data.columns = normalized_cols

                        # 将日期列标准化为“日期”
                        date_candidates = ['日期', '数据日期', '交易日期', 'date', 'Date']
                        for dc in date_candidates:
                            if dc in value_data.columns:
                                if dc != '日期':
                                    value_data = value_data.rename(columns={dc: '日期'})
                                break

                        if '日期' in value_data.columns:
                            value_data['日期'] = pd.to_datetime(value_data['日期'], errors='coerce').dt.strftime('%Y-%m-%d')

                        # 打印一次列名，便于排查字段不匹配问题
                        logger.info(f"stock_value_em({code}) columns: {list(value_data.columns)}; rows={len(value_data)}")
                        return value_data
                    else:
                        logger.warning(f"股票{code}的估值数据为空")
                        return pd.DataFrame()
                except Exception as e:
                    logger.warning(f"获取股票{code}估值数据失败: {e}")
                    return pd.DataFrame()
            
            # 初始化东方财富接口返回的估值指标列
            em_metrics = ['PE_TTM_em', 'PE静_em', '市净率_em', 'PEG值_em', '市现率_em', '市销率_em', 
                         '总市值_em', '流通市值_em', '总股本_em', '流通股本_em']
            for metric in em_metrics:
                df[metric] = np.nan
            
            # 批量获取各股票的估值数据
            logger.info("开始使用东方财富接口获取估值数据...")
            for code in tqdm(unique_codes, desc="获取估值数据"):
                try:
                    # 获取该股票的估值数据
                    value_data = get_stock_value_data(code)
                    
                    if not value_data.empty and not df[df['股票代码'] == code].empty:
                        # 获取该股票的所有报告期
                        stock_dates = df[df['股票代码'] == code]['日期'].unique()
                        
                        for report_date in stock_dates:
                            # 查找最接近报告期的估值数据（兼容多种日期列名）
                            date_col_candidates = ['日期', '交易日期', 'date', 'Date']
                            date_col = next((c for c in date_col_candidates if c in value_data.columns), None)
                            if date_col is None:
                                logger.debug(f"股票{code}估值数据缺少日期列，候选未命中: {date_col_candidates}")
                                continue

                            # 转换报告期为字符串格式以便比较
                            report_date_str = pd.to_datetime(report_date).strftime('%Y-%m-%d')

                            # 将估值数据日期转为日期字符串用于比较（若还未转换）
                            try:
                                value_data[date_col] = pd.to_datetime(value_data[date_col]).dt.strftime('%Y-%m-%d')
                            except Exception:
                                logger.debug(f"股票{code}估值数据日期列转换失败: {date_col}")
                                continue
                            
                            # 筛选在报告期之前或等于报告期的记录
                            valid_dates = value_data[value_data[date_col] <= report_date_str]
                            
                            if not valid_dates.empty:
                                # 获取最接近报告期的记录（按日期排序后取最后一条）
                                valid_dates = valid_dates.sort_values(by=date_col)
                                closest_record = valid_dates.iloc[-1]
                                
                                # 计算日期差距
                                date_diff = (pd.to_datetime(report_date_str) - 
                                            pd.to_datetime(closest_record[date_col])).days
                                
                                # 如果差距在合理范围内（放宽到120天），则使用该数据
                                if date_diff <= 120:
                                    # 更新对应报告期的估值指标
                                    mask = (df['股票代码'] == code) & (df['日期'] == report_date)
                                    # 映射东方财富接口返回的字段到自定义字段（兼容多候选名）
                                    multi_field_mapping = {
                                        ('市盈率(TTM)', 'PE(TTM)', '滚动市盈率', 'PE_TTM'): 'PE_TTM_em',
                                        ('市盈率(静)', 'PE(静)', '静态市盈率', 'PE_静'): 'PE静_em',
                                        ('市净率', 'PB', 'PB(市净率)'): '市净率_em',
                                        ('PEG', 'PEG值'): 'PEG值_em',
                                        ('市现率', 'PCF', 'P/CF'): '市现率_em',
                                        ('市销率', 'PS', 'P/S'): '市销率_em',
                                        ('总市值', '总市值(元)', '总市值_元'): '总市值_em',
                                        ('流通市值', '流通市值(元)', '流通市值_元'): '流通市值_em',
                                        ('总股本', '总股本(股)', '总股本_股'): '总股本_em',
                                        ('流通股本', '流通股本(股)', '流通股本_股'): '流通股本_em'
                                    }

                                    for source_field_candidates, target_field in multi_field_mapping.items():
                                        matched = False
                                        for candidate in source_field_candidates:
                                            if candidate in closest_record.index and pd.notna(closest_record[candidate]):
                                                try:
                                                    value = float(closest_record[candidate])
                                                    df.loc[mask, target_field] = value
                                                    matched = True
                                                    break
                                                except (ValueError, TypeError):
                                                    logger.warning(f"股票{code}报告期{report_date}字段{candidate}值转换失败")
                                        if not matched:
                                            logger.debug(f"股票{code}报告期{report_date}未匹配到{target_field}的任何候选列: {source_field_candidates}")

                                    logger.info(f"股票{code}报告期{report_date}使用东方财富估值数据（日期差:{date_diff}天）")
                                else:
                                    logger.debug(f"股票{code}报告期{report_date_str}最近估值数据距今{date_diff}天，超出阈值")
                except Exception as e:
                    logger.warning(f"处理股票{code}估值数据失败: {e}")
            
            # 计算PE_TTM（完全使用东方财富接口数据）
            if 'PE_TTM_em' in df.columns:
                df['PE_TTM'] = df['PE_TTM_em'].copy()
                logger.info(f"使用东方财富接口PE_TTM数据，共{len(df[~df['PE_TTM'].isna()])}条有效记录")
            else:
                df['PE_TTM'] = np.nan
                logger.warning("PE_TTM_em列不存在，PE_TTM设置为NaN")

            # 透明与可验证：标注口径
            # 统一使用东方财富接口数据
            df['pe_data_source'] = '东方财富接口'
            df['pb_data_source'] = '东方财富接口'
            
            # 确保所有东方财富接口估值指标都被包含在最终结果中
            # 将东方财富接口指标重命名为标准字段名，便于后续使用
            em_to_standard_mapping = {
                'PE静_em': 'PE静',
                'PEG值_em': 'PEG',
                '市现率_em': '市现率',
                '市销率_em': '市销率',
                '流通市值_em': '流通市值',
                '流通股本_em': '流通股本'
            }
            
            for em_field, standard_field in em_to_standard_mapping.items():
                if em_field in df.columns:
                    df[standard_field] = df[em_field].copy()
                    logger.info(f"使用东方财富接口{standard_field}数据，共{len(df[~df[standard_field].isna()])}条有效记录")
                else:
                    df[standard_field] = np.nan
                    logger.warning(f"{em_field}列不存在，{standard_field}设置为NaN")
            
            df['估值口径说明'] = (
                np.where(~df['PE'].isna(), 'PE=报告期末收盘价/基本每股收益; ', '') +
                'PE_TTM=东方财富接口数据; ' +
                '市净率=东方财富接口数据; ' +
                'PE静=东方财富接口数据; ' +
                'PEG=东方财富接口数据; ' +
                '市现率=东方财富接口数据; ' +
                '市销率=东方财富接口数据; ' +
                '流通市值=东方财富接口数据; ' +
                '流通股本=东方财富接口数据; ' +
                np.where(df['PE_TTM'].isna(), 'PE_TTM缺失：东方财富接口数据不可得; ', '') +
                '数据来源标识：pe_data_source列（东方财富接口）; ' +
                'pb_data_source列（东方财富接口）; ' +
                '时间对齐：股价按报告期末后首个交易日; '
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
            
            # 使用东方财富接口返回的市净率数据
            if '市净率_em' in df.columns:
                df['市净率'] = df['市净率_em'].copy()
                logger.info(f"使用东方财富接口市净率数据，共{len(df[~df['市净率'].isna()])}条有效记录")
            else:
                df['市净率'] = np.nan
                logger.warning("市净率_em列不存在，市净率设置为NaN")
                
            # 计算市净率分位数
            if '市净率' in df.columns:
                df['市净率分位数'] = df.groupby('日期')['市净率'].transform(
                    lambda x: x.rank(pct=True) if not x.isna().all() else np.nan
                )
                
                # 计算市净率变化率
                df['市净率变化率'] = df.groupby('股票代码')['市净率'].pct_change()
                
                # 添加数据来源标识
                df['pb_data_source'] = '东方财富接口'
                
                logger.info(f"市净率计算完成，有效记录数：{len(df[~df['市净率'].isna()])}")
            
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
            from tqdm import tqdm
            
            logger.info("开始计算基本因子...")
            
            # 创建整体进度条
            with tqdm(total=5, desc="整体进度") as pbar:
                # 获取财务数据
                pbar.set_description("获取财务数据")
                df = self.get_financial_data(stock_codes, start_date, end_date)
                pbar.update(1)
                
                if df.empty:
                    logger.warning("未获取到财务数据，返回空DataFrame")
                    pbar.update(4)  # 跳过剩余步骤
                    return df
                
                # 计算估值指标（使用新方法）
                pbar.set_description("计算估值指标")
                df = self.calculate_valuation_metrics(df)
                pbar.update(1)
                
                # 计算市净率因子（需要股价数据）
                pbar.set_description("计算市净率因子")
                df = self.calculate_pb_ratio(df, price_data)
                pbar.update(1)
                
                # 计算ROE/ROA因子
                pbar.set_description("计算ROE/ROA因子")
                df = self.calculate_roe_roa(df)
                pbar.update(1)
                
                # 计算增长率因子
                pbar.set_description("计算增长率因子")
                df = self.calculate_growth_rates(df)
                pbar.update(1)
                
                # 完成处理记录
                self.processing_log['end_time'] = datetime.now().isoformat()
                self.processing_log['total_rows'] = str(len(df))
                self.processing_log['factors_calculated'] = [
                    col for col in df.columns if col not in ['日期', '股票代码']
                ]
                
                pbar.set_description("基本因子计算完成")
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
        'data_source': 'akshare',
        'output_base_dir': Path('data_pipeline/data/features')
    }
    
    # 创建基本因子计算器
    calculator = FundamentalFactors(config)
    
    try:
        # 读取沪深300成分股（使用cutoff_2024文件）
        components_file = "data_pipeline/data/components/hs300_components_cutoff_2024.csv"
        components_df = pd.read_csv(components_file)
        stock_codes = components_df['股票代码'].tolist()  # 使用所有股票代码
        
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