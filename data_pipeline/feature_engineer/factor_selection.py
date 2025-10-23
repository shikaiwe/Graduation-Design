#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
因子选择模块
实现相关性分析、主成分分析（PCA）等因子选择方法
"""

import pandas as pd
import numpy as np
import logging
from typing import Dict, Any, Optional, List, Tuple
import json
from datetime import datetime
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import SelectKBest, f_regression, mutual_info_regression
import matplotlib.pyplot as plt
import seaborn as sns

# 配置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class FactorSelection:
    """因子选择器"""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        初始化因子选择器
        
        参数:
            config: 配置参数字典
        """
        self.config = config or {
            'correlation_threshold': 0.8,  # 相关性阈值
            'pca_variance_threshold': 0.95,  # PCA方差解释阈值
            'top_k_factors': 50,  # 保留的因子数量
            'selection_methods': ['correlation', 'pca', 'feature_importance']  # 选择方法
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
    
    def load_all_factors(self) -> Tuple[pd.DataFrame, pd.Series]:
        """
        加载所有因子数据和股票收益数据
        
        返回:
            Tuple[pd.DataFrame, pd.Series]: (因子数据, 收益数据)
        """
        try:
            logger.info("加载所有因子数据...")
            
            # 统一解析数据目录，兼容不同工作目录
            import os
            script_dir = os.path.dirname(os.path.abspath(__file__))
            features_dir = os.path.abspath(os.path.join(script_dir, "..", "data", "features"))
            tech_path = os.path.join(features_dir, "technical_indicators.csv")
            fundamental_path = os.path.join(features_dir, "fundamental_factors.csv")
            macro_path = os.path.join(features_dir, "macro_factors.csv")
            epu_path = os.path.join(features_dir, "epu_factors.csv")
            
            # 行情数据（用于收益）
            prices_path = os.path.abspath(os.path.join(script_dir, "..", "data", "daily_prices", "Merge", "hs300_daily_prices_merged.csv"))
            
            # 加载技术指标因子
            tech_factors = pd.read_csv(tech_path)
            
            # 加载基本因子
            fundamental_factors = pd.read_csv(fundamental_path)
            
            # 加载宏观因子
            macro_factors = pd.read_csv(macro_path)
            
            # 加载EPU因子
            epu_factors = pd.read_csv(epu_path)
            
            # 合并所有因子数据
            all_factors = tech_factors.copy()
            
            # 动态解析列名以兼容不同数据源
            def resolve_col(df, candidates: List[str]) -> Optional[str]:
                for c in candidates:
                    if c in df.columns:
                        return c
                return None
            date_candidates = ['日期', 'Date', 'date', 'trade_date', 'dt']
            code_candidates = ['股票代码', 'ts_code', 'code', '证券代码', 'Symbol', 'stock_code', 'symbol']
            
            # all_factors 的日期与代码列名
            all_date_col = resolve_col(all_factors, date_candidates)
            all_code_col = resolve_col(all_factors, code_candidates)
            
            # 合并基本因子（按可用公共列进行合并）
            if not fundamental_factors.empty:
                fund_date_col = resolve_col(fundamental_factors, date_candidates)
                fund_code_col = resolve_col(fundamental_factors, code_candidates)
                merge_cols = []
                if all_date_col and fund_date_col:
                    # 统一日期类型
                    fundamental_factors[fund_date_col] = pd.to_datetime(fundamental_factors[fund_date_col], errors='coerce')
                    all_factors[all_date_col] = pd.to_datetime(all_factors[all_date_col], errors='coerce')
                    merge_cols.append((all_date_col, fund_date_col))
                if all_code_col and fund_code_col:
                    merge_cols.append((all_code_col, fund_code_col))
                if merge_cols:
                    # 构造左右键名映射后合并
                    left_on = [l for l, _ in merge_cols]
                    right_on = [r for _, r in merge_cols]
                    all_factors = pd.merge(all_factors, fundamental_factors, left_on=left_on, right_on=right_on, how='left')
                else:
                    logger.warning('基本因子缺少可用于合并的公共列，已跳过合并。')
            
            # 合并宏观因子（按日期 asof 合并）
            if not macro_factors.empty:
                macro_date_col = resolve_col(macro_factors, date_candidates)
                if all_date_col and macro_date_col:
                    macro_factors[macro_date_col] = pd.to_datetime(macro_factors[macro_date_col], errors='coerce')
                    all_factors[all_date_col] = pd.to_datetime(all_factors[all_date_col], errors='coerce')
                    all_factors = pd.merge_asof(
                        all_factors.sort_values(all_date_col),
                        macro_factors.sort_values(macro_date_col),
                        left_on=all_date_col,
                        right_on=macro_date_col,
                        direction='backward'
                    )
                else:
                    logger.warning('宏观因子缺少日期列，已跳过合并。')
            
            # 合并EPU因子（按日期 asof 合并）
            if not epu_factors.empty:
                epu_date_col = resolve_col(epu_factors, date_candidates)
                if all_date_col and epu_date_col:
                    epu_factors[epu_date_col] = pd.to_datetime(epu_factors[epu_date_col], errors='coerce')
                    all_factors[all_date_col] = pd.to_datetime(all_factors[all_date_col], errors='coerce')
                    all_factors = pd.merge_asof(
                        all_factors.sort_values(all_date_col),
                        epu_factors.sort_values(epu_date_col),
                        left_on=all_date_col,
                        right_on=epu_date_col,
                        direction='backward'
                    )
                else:
                    logger.warning('EPU因子缺少日期列，已跳过合并。')
            
            # 合并行情数据以获得收益
            try:
                price_df = pd.read_csv(prices_path)
                price_date_col = resolve_col(price_df, date_candidates + ['date'])
                price_code_col = resolve_col(price_df, code_candidates)
                if price_date_col is None or price_code_col is None:
                    logger.warning('行情数据缺少日期或代码列，跳过收益合并。')
                else:
                    # 统一日期
                    price_df[price_date_col] = pd.to_datetime(price_df[price_date_col], errors='coerce')
                    all_factors[all_date_col] = pd.to_datetime(all_factors[all_date_col], errors='coerce')
                    
                    # 统一代码格式：保留6位数字，不含交易所后缀
                    def normalize_code_series(s: pd.Series) -> pd.Series:
                        return s.astype(str).str.extract(r'(\d+)')[0].str[-6:].str.zfill(6)
                    price_df[price_code_col] = normalize_code_series(price_df[price_code_col])
                    all_factors[all_code_col] = normalize_code_series(all_factors[all_code_col])
                    
                    # 选择可用的收益列（优先 change_rate, 次选 pct_chg 等）
                    price_return_candidates = ['change_rate', 'pct_chg', 'return', 'ret', 'daily_return', '收益率']
                    price_ret_col = resolve_col(price_df, price_return_candidates)
                    if price_ret_col is None:
                        logger.warning('行情数据中未发现收益列（如 change_rate/pct_chg），仅合并价格列将无法做相关性/互信息。')
                        cols_to_merge = [price_date_col, price_code_col]
                    else:
                        cols_to_merge = [price_date_col, price_code_col, price_ret_col]
                    
                    # 合并（精确匹配日期+代码）
                    all_factors = pd.merge(
                        all_factors,
                        price_df[cols_to_merge].rename(columns={price_date_col: all_date_col, price_code_col: all_code_col}),
                        on=[all_date_col, all_code_col],
                        how='left'
                    )
                    
                    # 诊断信息：收益列存在性与非空计数
                    diag_ret_col = resolve_col(all_factors, ['change_rate', 'pct_chg', 'return', 'ret', 'daily_return', '收益率'])
                    if diag_ret_col:
                        non_null = all_factors[diag_ret_col].notna().sum()
                        logger.info(f"收益列 '{diag_ret_col}' 已合并，非空样本数: {non_null}")
                    else:
                        logger.warning('合并后仍未找到收益列。')
            except FileNotFoundError:
                logger.warning('未找到行情文件，跳过收益合并。')
            except Exception as e:
                logger.warning(f'合并行情收益失败：{e}')
            
            # 提取收益数据（兼容不同列名）
            return_candidates = ['收益率', 'return', 'ret', 'daily_return', 'pct_chg', 'change_rate']
            return_col = resolve_col(all_factors, return_candidates)
            if return_col:
                returns = all_factors[return_col]
                drop_cols = [return_col]
            else:
                returns = None
                drop_cols = []
            # 删除通用的非因子列（使用已解析的列名）
            for c in [all_date_col, all_code_col]:
                if c:
                    drop_cols.append(c)
            factors = all_factors.drop(columns=drop_cols, errors='ignore')
            
            # 清理数据
            factors = factors.dropna(axis=1, how='all')  # 删除全空列
            factors = factors.ffill().bfill()  # 填充空值（兼容未来版本）
            
            # 仅保留数值型因子，且强制数值化（无法转换的置为NaN后再填充/删除）
            for col in factors.columns:
                if not pd.api.types.is_numeric_dtype(factors[col]):
                    factors[col] = pd.to_numeric(factors[col], errors='coerce')
            # 再次删除全空列，并填充缺失值
            factors = factors.dropna(axis=1, how='all').ffill().bfill()
            
            # 保证收益为数值型
            if returns is not None:
                returns = pd.to_numeric(returns, errors='coerce')
                returns = returns.ffill().bfill()
            
            self._log_step('load_all_factors', {
                'total_factors': len(factors.columns),
                'factor_types': ['technical', 'fundamental', 'macro', 'epu'],
                'final_shape': factors.shape
            })
            
            return factors, returns
            
        except Exception as e:
            logger.error(f"加载因子数据失败: {str(e)}")
            raise
    
    def correlation_analysis(self, factors: pd.DataFrame, returns: pd.Series) -> pd.DataFrame:
        """
        进行相关性分析
        
        参数:
            factors: 因子数据
            returns: 收益数据
            
        返回:
            pd.DataFrame: 相关性分析结果
        """
        try:
            logger.info("进行相关性分析...")
            
            # 计算因子与收益的相关性
            correlations = {}
            for col in factors.columns:
                try:
                    corr = factors[col].corr(returns)
                    correlations[col] = corr
                except:
                    correlations[col] = np.nan
            
            # 创建相关性结果DataFrame
            corr_df = pd.DataFrame({
                'factor': list(correlations.keys()),
                'correlation_with_returns': list(correlations.values())
            }).sort_values('correlation_with_returns', key=abs, ascending=False)
            
            # 筛选高相关性因子
            high_corr_factors = corr_df[
                abs(corr_df['correlation_with_returns']) > self.config['correlation_threshold']
            ]
            
            # 计算因子间的相关性矩阵
            factor_corr_matrix = factors.corr()
            
            # 识别高度相关的因子对
            high_corr_pairs = []
            for i in range(len(factor_corr_matrix.columns)):
                for j in range(i+1, len(factor_corr_matrix.columns)):
                    corr_val = abs(factor_corr_matrix.iloc[i, j])
                    if corr_val > self.config['correlation_threshold']:
                        high_corr_pairs.append({
                            'factor1': factor_corr_matrix.columns[i],
                            'factor2': factor_corr_matrix.columns[j],
                            'correlation': corr_val
                        })
            
            self._log_step('correlation_analysis', {
                'total_factors_analyzed': len(factors.columns),
                'high_correlation_factors': len(high_corr_factors),
                'high_correlation_pairs': len(high_corr_pairs),
                'max_correlation': corr_df['correlation_with_returns'].abs().max()
            })
            
            return corr_df
            
        except Exception as e:
            logger.error(f"相关性分析失败: {str(e)}")
            raise
    
    def pca_analysis(self, factors: pd.DataFrame) -> Dict[str, Any]:
        """
        进行主成分分析（PCA）
        
        参数:
            factors: 因子数据
            
        返回:
            Dict[str, Any]: PCA分析结果
        """
        try:
            logger.info("进行主成分分析...")
            
            # 复制并清洗特征，确保无 NaN/Inf/极端值后再标准化
            X = factors.copy()
            # 数值化与替换无穷
            for col in X.columns:
                X[col] = pd.to_numeric(X[col], errors='coerce')
            X.replace([np.inf, -np.inf], np.nan, inplace=True)
            
            # 分位数截断缓解极端值
            q_low, q_high = 0.01, 0.99
            for col in X.columns:
                s = X[col]
                if s.notna().sum() == 0:
                    continue
                lo = s.quantile(q_low)
                hi = s.quantile(q_high)
                X[col] = s.clip(lower=lo, upper=hi)
            
            # 用中位数填充，删除零方差列
            med = X.median(numeric_only=True)
            for col in list(X.columns):
                if X[col].notna().sum() == 0:
                    X.drop(columns=[col], inplace=True)
                else:
                    X[col] = X[col].fillna(med.get(col, 0))
            nunique = X.nunique()
            zero_var = nunique[nunique <= 1].index.tolist()
            if zero_var:
                X.drop(columns=zero_var, inplace=True)
                logger.info(f"PCA前删除零方差列: {len(zero_var)}")
            
            # 标准化
            scaler = StandardScaler()
            X_scaled = scaler.fit_transform(X)
            
            # 进行PCA
            pca = PCA()
            pca_result = pca.fit(X_scaled)
            
            # 计算累计方差解释
            cumulative_variance = np.cumsum(pca_result.explained_variance_ratio_)
            
            # 确定达到方差阈值的成分数量
            n_components = np.argmax(cumulative_variance >= self.config['pca_variance_threshold']) + 1
            
            # 获取主成分载荷
            loadings = pd.DataFrame(
                pca_result.components_[:n_components].T,
                index=X.columns,
                columns=[f'PC{i+1}' for i in range(n_components)]
            )
            
            pca_results = {
                'n_components': n_components,
                'explained_variance_ratio': pca_result.explained_variance_ratio_[:n_components],
                'cumulative_variance': cumulative_variance[:n_components],
                'loadings': loadings,
                'components': pca_result.components_[:n_components]
            }
            
            self._log_step('pca_analysis', {
                'original_dimensions': factors.shape[1],
                'reduced_dimensions': n_components,
                'variance_explained': float(cumulative_variance[n_components-1])
            })
            
            return pca_results
            
        except Exception as e:
            logger.error(f"PCA分析失败: {str(e)}")
            raise
    
    def feature_importance_analysis(self, factors: pd.DataFrame, returns: pd.Series) -> pd.DataFrame:
        """
        进行特征重要性分析
        
        参数:
            factors: 因子数据
            returns: 收益数据
            
        返回:
            pd.DataFrame: 特征重要性结果
        """
        try:
            logger.info("进行特征重要性分析...")
            
            # 深拷贝并统一为数值，处理无穷与极端值
            X = factors.copy()
            y = returns.copy() if returns is not None else None
            
            # 将所有列转为数值型（无法转换的设为 NaN）
            for col in X.columns:
                X[col] = pd.to_numeric(X[col], errors='coerce')
            # 替换无穷为 NaN
            X.replace([np.inf, -np.inf], np.nan, inplace=True)
            
            # 分位数截断（winsorize）以缓解异常值影响
            q_low, q_high = 0.01, 0.99
            for col in X.columns:
                series = X[col]
                if series.notna().sum() == 0:
                    continue
                low = series.quantile(q_low)
                high = series.quantile(q_high)
                X[col] = series.clip(lower=low, upper=high)
            
            # 用中位数填充缺失；全缺失列则删除
            medians = X.median(numeric_only=True)
            for col in list(X.columns):
                if X[col].notna().sum() == 0:
                    X.drop(columns=[col], inplace=True)
                else:
                    X[col] = X[col].fillna(medians.get(col, 0))
            
            # 删除零方差列（常数列）
            nunique = X.nunique()
            zero_var_cols = nunique[nunique <= 1].index.tolist()
            if zero_var_cols:
                X.drop(columns=zero_var_cols, inplace=True)
                logger.info(f"已删除零方差列: {len(zero_var_cols)}")
            
            # 处理 y（收益）
            if y is None:
                logger.warning("收益数据缺失，跳过特征重要性分析并返回空结果")
                return pd.DataFrame(columns=['factor', 'importance_score'])
            y = pd.to_numeric(y, errors='coerce')
            y = y.replace([np.inf, -np.inf], np.nan)
            if y.notna().sum() == 0:
                logger.warning("收益数据全部为缺失，跳过特征重要性分析并返回空结果")
                return pd.DataFrame(columns=['factor', 'importance_score'])
            y = y.fillna(y.median())
            
            # 对齐 X 与 y 的索引并删除残余的缺失行
            common_index = X.index.intersection(y.index)
            X = X.loc[common_index]
            y = y.loc[common_index]
            # 确保没有非有限值
            X = X.replace([np.inf, -np.inf], np.nan)
            invalid_rows = X.isna().any(axis=1)
            if invalid_rows.any():
                X = X[~invalid_rows]
                y = y[~invalid_rows]
                logger.info(f"已删除含缺失/非有限值的样本行: {invalid_rows.sum()}")
            
            # 标准化特征
            scaler = StandardScaler()
            X_scaled = scaler.fit_transform(X)
            
            # 使用互信息进行特征选择
            selector = SelectKBest(score_func=mutual_info_regression, k='all')
            selector.fit(X_scaled, y)
            
            # 获取特征重要性得分
            importance_scores = selector.scores_
            
            # 创建重要性结果DataFrame
            importance_df = pd.DataFrame({
                'factor': X.columns,
                'importance_score': importance_scores
            }).sort_values('importance_score', ascending=False)
            
            # 选择Top K因子
            top_k_factors = importance_df.head(self.config['top_k_factors'])
            
            self._log_step('feature_importance_analysis', {
                'total_factors': len(X.columns),
                'top_k_selected': len(top_k_factors),
                'max_importance': float(np.nanmax(importance_scores))
            })
            
            return importance_df
            
        except Exception as e:
            logger.error(f"特征重要性分析失败: {str(e)}")
            raise
    
    def select_final_factors(self, factors: pd.DataFrame, returns: pd.Series) -> List[str]:
        """
        综合选择最终因子
        
        参数:
            factors: 因子数据
            returns: 收益数据
            
        返回:
            List[str]: 最终选择的因子列表
        """
        try:
            logger.info("综合选择最终因子...")
            
            selected_factors = set()
            
            # 相关性分析选择（仅在收益可用时执行）
            if 'correlation' in self.config['selection_methods'] and returns is not None and pd.api.types.is_numeric_dtype(returns):
                try:
                    corr_results = self.correlation_analysis(factors, returns)
                    high_corr_factors = corr_results.head(self.config['top_k_factors'])['factor'].tolist()
                    selected_factors.update(high_corr_factors)
                except Exception as e:
                    logger.warning(f"相关性分析跳过：{e}")
            else:
                logger.warning("收益不可用或非数值，相关性分析已跳过")
            
            # 特征重要性选择（若收益不可用则 importance_results 为空，自动跳过）
            if 'feature_importance' in self.config['selection_methods']:
                importance_results = self.feature_importance_analysis(factors, returns)
                if not importance_results.empty:
                    important_factors = importance_results.head(self.config['top_k_factors'])['factor'].tolist()
                    selected_factors.update(important_factors)
                else:
                    logger.warning("特征重要性结果为空，已跳过该步骤")
            
            # PCA分析（用于降维，不直接选择因子）
            if 'pca' in self.config['selection_methods']:
                pca_results = self.pca_analysis(factors)
                # PCA主要用于理解数据结构，不直接选择因子
            
            # 若因收益不可用，且未选择到任何因子，则退化为方差选 Top-K
            if not selected_factors:
                logger.warning("未从相关性/特征重要性中选出因子，将采用方差最大的 Top-K 作为降级方案")
                variances = factors.var(numeric_only=True).sort_values(ascending=False)
                selected_factors = set(variances.head(self.config['top_k_factors']).index.tolist())
            
            # 转换为列表并排序
            final_factors = list(selected_factors)
            final_factors.sort()
            
            self._log_step('select_final_factors', {
                'final_factors_selected': len(final_factors),
                'selection_methods_used': self.config['selection_methods']
            })
            
            return final_factors
            
        except Exception as e:
            logger.error(f"最终因子选择失败: {str(e)}")
            raise
    
    def save_selection_results(self, results: Dict[str, Any], file_path: str):
        """保存选择结果到文件"""
        try:
            # 转换为可序列化的格式
            serializable_results = {}
            for key, value in results.items():
                if isinstance(value, pd.DataFrame):
                    serializable_results[key] = value.to_dict()
                elif isinstance(value, np.ndarray):
                    serializable_results[key] = value.tolist()
                else:
                    serializable_results[key] = value
            
            with open(file_path, 'w', encoding='utf-8') as f:
                json.dump(serializable_results, f, ensure_ascii=False, indent=2)
            logger.info(f"选择结果已保存到: {file_path}")
        except Exception as e:
            logger.error(f"保存选择结果失败: {str(e)}")


def main():
    """主函数示例"""
    # 配置参数
    config = {
        'correlation_threshold': 0.8,
        'pca_variance_threshold': 0.95,
        'top_k_factors': 50,
        'selection_methods': ['correlation', 'pca', 'feature_importance']
    }
    
    # 创建因子选择器
    selector = FactorSelection(config)
    
    try:
        # 加载因子数据
        factors, returns = selector.load_all_factors()
        
        if factors.empty:
            logger.warning("未加载到因子数据")
            return
        
        # 进行因子选择
        final_factors = selector.select_final_factors(factors, returns)
        
        # 保存结果
        results = {
            'selected_factors': final_factors,
            'total_factors_analyzed': len(factors.columns),
            'selection_config': config
        }
        
        output_file = "data_pipeline/data/features/factor_selection_results.json"
        selector.save_selection_results(results, output_file)
        
        logger.info(f"因子选择完成！最终选择{len(final_factors)}个因子")
        logger.info(f"结果已保存到: {output_file}")
        
    except Exception as e:
        logger.error(f"因子选择失败: {str(e)}")


if __name__ == "__main__":
    main()