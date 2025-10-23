#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
因子选择模块 - 重构版本

实现多种因子选择方法，包括相关性分析、主成分分析（PCA）、特征重要性分析等。
采用模块化设计，支持可扩展的因子选择策略。

主要特性：
- 模块化架构设计
- 统一的配置管理
- 完善的错误处理
- 详细的日志记录
- 可扩展的因子选择策略
"""

import pandas as pd
import numpy as np
import logging
from abc import ABC, abstractmethod
from typing import Dict, Any, Optional, List, Tuple, Union
import json
from datetime import datetime
from pathlib import Path
from dataclasses import dataclass, asdict
from enum import Enum
import warnings
import hashlib
import pickle
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed
from functools import lru_cache
import gc
import sys
import psutil
import os

# 第三方库导入
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import mutual_info_regression

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('factor_selection.log', encoding='utf-8')
    ]
)
logger = logging.getLogger(__name__)


class FactorSelectionMethod(Enum):
    """因子选择方法枚举"""
    CORRELATION = "correlation"
    PCA = "pca" 
    FEATURE_IMPORTANCE = "feature_importance"
    VARIANCE = "variance"


@dataclass
class FactorSelectionConfig:
    """因子选择配置类"""
    correlation_threshold: float = 0.8
    pca_variance_threshold: float = 0.95
    top_k_factors: int = 50
    selection_methods: List[FactorSelectionMethod] = None
    data_paths: Dict[str, str] = None
    
    def __post_init__(self):
        if self.selection_methods is None:
            self.selection_methods = [
                FactorSelectionMethod.CORRELATION,
                FactorSelectionMethod.PCA,
                FactorSelectionMethod.FEATURE_IMPORTANCE
            ]
        if self.data_paths is None:
            self.data_paths = {
                'technical': "data_pipeline/data/features/technical_indicators.csv",
                'fundamental': "data_pipeline/data/features/fundamental_factors.csv", 
                'macro': "data_pipeline/data/features/macro_factors.csv",
                'epu': "data_pipeline/data/features/epu_factors.csv",
                'prices': "data_pipeline/data/daily_prices/Merge/hs300_daily_prices_merged.csv"
            }
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典格式"""
        result = asdict(self)
        result['selection_methods'] = [method.value for method in self.selection_methods]
        return result


class DataLoader:
    """数据加载器 - 负责统一的数据加载和预处理"""
    
    def __init__(self, config: FactorSelectionConfig):
        self.config = config
        self.logger = logging.getLogger(f"{__name__}.DataLoader")
        self.cache_dir = Path("data_pipeline/cache")
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self._memory_monitor_enabled = True
        self._memory_threshold_mb = 1000  # 内存阈值1GB
        
    def _check_memory_usage(self) -> bool:
        """检查内存使用情况"""
        if not self._memory_monitor_enabled:
            return True
            
        try:
            process = psutil.Process(os.getpid())
            memory_mb = process.memory_info().rss / 1024 / 1024
            
            if memory_mb > self._memory_threshold_mb:
                self.logger.warning(f"内存使用过高: {memory_mb:.2f}MB, 超过阈值 {self._memory_threshold_mb}MB")
                return False
            return True
        except Exception:
            # 如果无法获取内存信息，继续执行
            return True
    
    def _optimize_memory_usage(self, df: pd.DataFrame) -> pd.DataFrame:
        """优化DataFrame内存使用"""
        if df.empty:
            return df
            
        # 优化数值类型 - 使用更安全的方法
        for col in df.select_dtypes(include=['float64']).columns:
            try:
                # 先转换为float32，如果失败则保持原类型
                df[col] = df[col].astype('float32')
            except (ValueError, TypeError):
                # 如果转换失败，跳过该列
                continue
            
        for col in df.select_dtypes(include=['int64']).columns:
            try:
                # 根据数值范围选择合适的整数类型
                col_min = df[col].min()
                col_max = df[col].max()
                
                if col_min >= -128 and col_max <= 127:
                    df[col] = df[col].astype('int8')
                elif col_min >= -32768 and col_max <= 32767:
                    df[col] = df[col].astype('int16')
                elif col_min >= -2147483648 and col_max <= 2147483647:
                    df[col] = df[col].astype('int32')
                else:
                    # 保持int64
                    pass
            except (ValueError, TypeError):
                # 如果转换失败，跳过该列
                continue
            
        # 优化字符串类型
        for col in df.select_dtypes(include=['object']).columns:
            try:
                if df[col].nunique() / len(df) < 0.5:  # 低基数字符串
                    df[col] = df[col].astype('category')
            except (ValueError, TypeError):
                # 如果转换失败，跳过该列
                continue
                
        return df
    
    def _cleanup_memory(self):
        """清理内存"""
        gc.collect()
        if hasattr(gc, 'collect'):
            gc.collect()  # 强制垃圾回收
    
    def _get_cache_key(self, operation: str, params: Dict[str, Any]) -> str:
        """生成缓存键"""
        param_str = json.dumps(params, sort_keys=True)
        key = f"{operation}_{hashlib.md5(param_str.encode()).hexdigest()}"
        return key
    
    def _load_from_cache(self, cache_key: str) -> Optional[Any]:
        """从缓存加载数据"""
        cache_file = self.cache_dir / f"{cache_key}.pkl"
        if cache_file.exists():
            try:
                with open(cache_file, 'rb') as f:
                    cached_data = pickle.load(f)
                self.logger.info(f"从缓存加载数据: {cache_key}")
                return cached_data
            except Exception as e:
                self.logger.warning(f"缓存加载失败 {cache_key}: {str(e)}")
        return None
    
    def _save_to_cache(self, cache_key: str, data: Any):
        """保存数据到缓存"""
        cache_file = self.cache_dir / f"{cache_key}.pkl"
        try:
            with open(cache_file, 'wb') as f:
                pickle.dump(data, f)
            self.logger.info(f"数据已缓存: {cache_key}")
        except Exception as e:
            self.logger.warning(f"缓存保存失败 {cache_key}: {str(e)}")
    
    def load_all_factors(self) -> Tuple[pd.DataFrame, Optional[pd.Series]]:
        """
        加载所有因子数据和股票收益数据 - 优化版本（带缓存和内存优化）
        
        返回:
            Tuple[pd.DataFrame, Optional[pd.Series]]: (因子数据, 收益数据)
        """
        try:
            self.logger.info("开始加载所有因子数据...")
            
            # 检查内存使用情况
            if not self._check_memory_usage():
                self.logger.warning("内存使用过高，建议清理内存后再执行")
            
            # 检查缓存
            cache_key = self._get_cache_key('load_all_factors', self.config.to_dict())
            cached_result = self._load_from_cache(cache_key)
            
            if cached_result is not None:
                self.logger.info("从缓存加载完整数据")
                return cached_result
            
            # 加载各类型因子数据
            factor_dfs = []
            
            for factor_type, file_path in self.config.data_paths.items():
                if factor_type == 'prices':
                    continue  # 价格数据单独处理
                    
                df = self._load_single_factor_file(file_path, factor_type)
                if df is not None:
                    # 优化内存使用
                    df = self._optimize_memory_usage(df)
                    factor_dfs.append(df)
                    
                    # 检查内存使用情况
                    if not self._check_memory_usage():
                        self.logger.warning("内存使用过高，清理内存...")
                        self._cleanup_memory()
            
            # 合并所有因子数据
            if not factor_dfs:
                raise ValueError("未成功加载任何因子数据")
                
            merged_factors = self._merge_factor_data(factor_dfs)
            
            # 数据预处理
            processed_factors = self._preprocess_data(merged_factors)
            
            # 进一步优化内存使用
            processed_factors = self._optimize_memory_usage(processed_factors)
            
            # 加载收益数据
            returns = self._load_returns_data()
            if returns is not None:
                # 优化收益数据内存使用
                returns = pd.to_numeric(returns, downcast='float')
            
            self.logger.info(f"数据加载完成: 因子数量={len(processed_factors.columns)}, "
                          f"样本数量={len(processed_factors)}")
            
            if returns is not None:
                self.logger.info(f"收益数据加载完成: 样本数量={returns.notna().sum()}")
            
            # 缓存结果
            result = (processed_factors, returns)
            self._save_to_cache(cache_key, result)
            
            # 清理临时数据
            del factor_dfs, merged_factors
            self._cleanup_memory()
            
            return result
            
        except Exception as e:
            self.logger.error(f"数据加载失败: {str(e)}")
            raise
    
    def _load_single_factor_file(self, file_path: str, factor_type: str) -> Optional[pd.DataFrame]:
        """加载单个因子文件"""
        try:
            if not Path(file_path).exists():
                self.logger.warning(f"因子文件不存在: {file_path}")
                return None
            
            df = pd.read_csv(file_path)
            self.logger.info(f"成功加载 {factor_type} 因子数据: {len(df)} 行, {len(df.columns)} 列")
            return df
            
        except Exception as e:
            self.logger.warning(f"加载 {factor_type} 因子数据失败: {str(e)}")
            return None
    
    def _merge_factor_data(self, factor_dfs: List[pd.DataFrame]) -> pd.DataFrame:
        """合并因子数据"""
        if len(factor_dfs) == 1:
            return factor_dfs[0]
        
        # 识别公共键列
        base_df = factor_dfs[0]
        date_col, code_col = self._identify_key_columns(base_df)
        
        if date_col is None:
            raise ValueError("无法识别日期列，无法合并因子数据")
        
        # 按顺序合并
        merged_df = base_df.copy()
        
        for i, df in enumerate(factor_dfs[1:], 1):
            df_date_col, df_code_col = self._identify_key_columns(df)
            
            if df_date_col is None:
                self.logger.warning(f"跳过第{i}个数据框，无法识别日期列")
                continue
            
            # 标准化列名
            df_standardized = df.rename(columns={df_date_col: date_col})
            if df_code_col and code_col:
                df_standardized = df_standardized.rename(columns={df_code_col: code_col})
            
            # 合并数据
            merge_on = [date_col]
            if code_col and code_col in df_standardized.columns:
                merge_on.append(code_col)
            
            merged_df = pd.merge(merged_df, df_standardized, on=merge_on, how='left', suffixes=('', f'_{i}'))
        
        return merged_df
    
    def _identify_key_columns(self, df: pd.DataFrame) -> Tuple[Optional[str], Optional[str]]:
        """识别日期和代码列"""
        date_candidates = ['日期', 'Date', 'date', 'trade_date', 'dt']
        code_candidates = ['股票代码', 'ts_code', 'code', '证券代码', 'Symbol', 'stock_code', 'symbol']
        
        date_col = None
        code_col = None
        
        for candidate in date_candidates:
            if candidate in df.columns:
                date_col = candidate
                break
        
        for candidate in code_candidates:
            if candidate in df.columns:
                code_col = candidate
                break
        
        return date_col, code_col
    
    def _load_returns_data(self) -> Optional[pd.Series]:
        """加载收益数据"""
        try:
            prices_path = self.config.data_paths.get('prices')
            if not prices_path or not Path(prices_path).exists():
                self.logger.warning("收益数据文件不存在，跳过收益数据加载")
                return None
            
            price_df = pd.read_csv(prices_path)
            
            # 识别收益列
            return_candidates = ['收益率', 'return', 'ret', 'daily_return', 'pct_chg', 'change_rate', 'change_amount']
            return_col = None
            
            for candidate in return_candidates:
                if candidate in price_df.columns:
                    return_col = candidate
                    break
            
            if return_col is None:
                self.logger.warning("未识别到收益列")
                return None
            
            returns = price_df[return_col].copy()
            returns = pd.to_numeric(returns, errors='coerce')
            
            self.logger.info(f"成功加载收益数据: 样本数量={returns.notna().sum()}")
            return returns
            
        except Exception as e:
            self.logger.warning(f"收益数据加载失败: {str(e)}")
            return None
    
    def _preprocess_data(self, factors: pd.DataFrame) -> pd.DataFrame:
        """数据预处理 - 优化版本"""
        # 识别并移除键列
        date_col, code_col = self._identify_key_columns(factors)
        drop_cols = []
        
        if date_col:
            drop_cols.append(date_col)
        if code_col:
            drop_cols.append(code_col)
        
        # 提取纯因子数据
        factor_data = factors.drop(columns=drop_cols, errors='ignore')
        
        # 批量删除全空列
        factor_data = factor_data.loc[:, factor_data.notna().any(axis=0)]
        
        # 批量转换为数值型 - 使用apply替代循环
        factor_data = factor_data.apply(pd.to_numeric, errors='coerce')
        
        # 优化缺失值填充 - 使用更高效的方法
        # 先删除缺失值过多的列
        missing_threshold = 0.5  # 50%缺失值阈值
        missing_ratio = factor_data.isna().sum() / len(factor_data)
        valid_cols = missing_ratio[missing_ratio < missing_threshold].index
        factor_data = factor_data[valid_cols]
        
        # 使用更高效的填充方法
        if not factor_data.empty:
            # 使用中位数填充，比ffill/bfill更高效
            factor_data = factor_data.fillna(factor_data.median())
        
        return factor_data
        
class FactorAnalysisStrategy(ABC):
    """因子分析策略抽象基类"""
    
    def __init__(self, config: FactorSelectionConfig):
        self.config = config
        self.logger = logging.getLogger(f"{__name__}.{self.__class__.__name__}")
        self._cache_enabled = True
        self._memory_monitor_enabled = True
        self._memory_threshold_mb = 1500  # 内存阈值1.5GB
        
    def _check_memory_usage(self) -> bool:
        """检查内存使用情况"""
        if not self._memory_monitor_enabled:
            return True
            
        try:
            process = psutil.Process(os.getpid())
            memory_mb = process.memory_info().rss / 1024 / 1024
            
            if memory_mb > self._memory_threshold_mb:
                self.logger.warning(f"内存使用过高: {memory_mb:.2f}MB, 超过阈值 {self._memory_threshold_mb}MB")
                return False
            return True
        except Exception:
            # 如果无法获取内存信息，继续执行
            return True
    
    def _cleanup_memory(self):
        """清理内存"""
        gc.collect()
        if hasattr(gc, 'collect'):
            gc.collect()  # 强制垃圾回收
    
    @abstractmethod
    def analyze(self, factors: pd.DataFrame, returns: Optional[pd.Series] = None) -> Dict[str, Any]:
        """执行因子分析"""
        pass
    
    def _get_cache_key(self, factors: pd.DataFrame, returns: Optional[pd.Series], method_name: str) -> str:
        """生成缓存键"""
        # 基于数据特征和参数生成唯一缓存键
        key_data = {
            'method': method_name,
            'factors_shape': factors.shape,
            'factors_columns_hash': hashlib.md5(str(sorted(factors.columns)).encode()).hexdigest()[:8],
            'factors_data_hash': hashlib.md5(factors.values.tobytes()).hexdigest()[:16] if not factors.empty else 'empty',
            'returns_hash': hashlib.md5(returns.values.tobytes()).hexdigest()[:16] if returns is not None else 'none'
        }
        return hashlib.md5(str(key_data).encode()).hexdigest()
    
    def _preprocess_factors(self, factors: pd.DataFrame) -> pd.DataFrame:
        """统一的数据预处理 - 优化版本"""
        # 直接使用副本以确保数据安全
        X = factors.copy()
        
        # 批量转换为数值型
        X = X.apply(pd.to_numeric, errors='coerce')
        
        # 批量处理无穷值
        X = X.replace([np.inf, -np.inf], np.nan)
        
        # 优化分位数截断 - 使用向量化操作
        q_low, q_high = 0.01, 0.99
        
        # 批量计算分位数
        quantiles = X.quantile([q_low, q_high])
        
        # 向量化截断操作
        for col in X.columns:
            if X[col].notna().sum() > 0:
                low = quantiles.loc[q_low, col]
                high = quantiles.loc[q_high, col]
                X[col] = X[col].clip(lower=low, upper=high)
        
        # 批量填充缺失值
        medians = X.median(numeric_only=True)
        X = X.fillna(medians)
        
        # 批量删除零方差列
        variances = X.var()
        zero_var_cols = variances[variances <= 1e-10].index.tolist()
        if zero_var_cols:
            X = X.drop(columns=zero_var_cols)
            self.logger.info(f"删除零方差列: {len(zero_var_cols)}")
        
        return X


class CorrelationAnalysisStrategy(FactorAnalysisStrategy):
    """相关性分析策略"""
    
    def __init__(self, config: FactorSelectionConfig):
        super().__init__(config)
        self._cache = {}
        self._memory_monitor_enabled = True
        self._memory_threshold_mb = 1500  # 内存阈值1.5GB
    
    def analyze(self, factors: pd.DataFrame, returns: Optional[pd.Series] = None) -> Dict[str, Any]:
        """执行相关性分析 - 优化版本（带缓存）"""
        try:
            # 检查缓存
            cache_key = self._get_cache_key(factors, returns, 'correlation')
            if self._cache_enabled and cache_key in self._cache:
                self.logger.info("从缓存加载相关性分析结果")
                return self._cache[cache_key]
            
            self.logger.info("开始相关性分析...")
            
            if returns is None:
                raise ValueError("相关性分析需要收益数据")
            
            # 预处理数据
            X = self._preprocess_factors(factors)
            y = returns.copy()
            
            # 对齐索引
            common_index = X.index.intersection(y.index)
            X = X.loc[common_index]
            y = y.loc[common_index]
            
            # 优化相关性计算 - 使用向量化操作
            # 批量计算所有因子与收益的相关性
            correlations = X.corrwith(y)
            
            # 创建相关性结果DataFrame
            corr_df = pd.DataFrame({
                'factor': correlations.index,
                'correlation_with_returns': correlations.values
            }).sort_values('correlation_with_returns', key=abs, ascending=False)
            
            # 筛选高相关性因子
            high_corr_mask = abs(correlations) > self.config.correlation_threshold
            high_corr_factors = correlations[high_corr_mask].index.tolist()
            
            # 优化因子间相关性矩阵计算
            # 使用更高效的corr方法，并限制计算范围
            factor_corr_matrix = X.corr(method='pearson', min_periods=10)
            
            # 优化高度相关因子对识别 - 使用numpy向量化操作
            corr_matrix_values = factor_corr_matrix.values
            np.fill_diagonal(corr_matrix_values, 0)  # 忽略对角线
            
            # 使用numpy的argwhere找到高相关性对
            high_corr_indices = np.argwhere(np.abs(corr_matrix_values) > self.config.correlation_threshold)
            
            # 过滤掉重复的对（只保留上三角）
            high_corr_pairs = []
            for i, j in high_corr_indices:
                if i < j:  # 只保留上三角，避免重复
                    corr_val = corr_matrix_values[i, j]
                    high_corr_pairs.append({
                        'factor1': factor_corr_matrix.columns[i],
                        'factor2': factor_corr_matrix.columns[j],
                        'correlation': corr_val
                    })
            
            # 按相关性绝对值排序
            high_corr_pairs.sort(key=lambda x: abs(x['correlation']), reverse=True)
            
            results = {
                'correlation_df': corr_df,
                'high_correlation_factors': high_corr_factors,
                'factor_correlation_matrix': factor_corr_matrix,
                'high_correlation_pairs': high_corr_pairs,
                'analysis_type': 'correlation'
            }
            
            # 保存到缓存
            if self._cache_enabled:
                self._cache[cache_key] = results
            
            self.logger.info(f"相关性分析完成: 高相关性因子数量={len(high_corr_factors)}")
            return results
            
        except Exception as e:
            self.logger.error(f"相关性分析失败: {str(e)}")
            raise


class PCAAnalysisStrategy(FactorAnalysisStrategy):
    """PCA分析策略"""
    
    def __init__(self, config: FactorSelectionConfig):
        super().__init__(config)
        self._cache = {}
        self._memory_monitor_enabled = True
        self._memory_threshold_mb = 1500  # 内存阈值1.5GB
    
    def analyze(self, factors: pd.DataFrame, returns: Optional[pd.Series] = None) -> Dict[str, Any]:
        """执行PCA分析 - 优化版本（带缓存）"""
        try:
            # 检查缓存
            cache_key = self._get_cache_key(factors, returns, 'pca')
            if self._cache_enabled and cache_key in self._cache:
                self.logger.info("从缓存加载PCA分析结果")
                return self._cache[cache_key]
            
            self.logger.info("开始PCA分析...")
            
            # 预处理数据
            X = self._preprocess_factors(factors)
            
            # 如果特征数量过多，先进行初步筛选
            if X.shape[1] > 1000:
                self.logger.info("特征数量过多，进行初步筛选...")
                # 使用方差进行初步筛选，保留前500个高方差特征
                variances = X.var()
                top_features = variances.nlargest(500).index
                X = X[top_features]
            
            # 优化标准化 - 使用更高效的实现
            scaler = StandardScaler()
            X_scaled = scaler.fit_transform(X)
            
            # 优化PCA计算 - 使用随机SVD加速
            # 对于大数据集，使用随机SVD可以显著加速
            if X_scaled.shape[0] > 1000 or X_scaled.shape[1] > 100:
                pca = PCA(svd_solver='randomized', random_state=42)
            else:
                pca = PCA(svd_solver='full')
            
            pca_result = pca.fit(X_scaled)
            
            # 计算累计方差解释
            cumulative_variance = np.cumsum(pca_result.explained_variance_ratio_)
            
            # 确定达到方差阈值的成分数量
            n_components = np.argmax(cumulative_variance >= self.config.pca_variance_threshold) + 1
            
            # 限制最大成分数量，避免过度计算
            max_components = min(100, X_scaled.shape[1])
            n_components = min(n_components, max_components)
            
            # 获取主成分载荷
            loadings = pd.DataFrame(
                pca_result.components_[:n_components].T,
                index=X.columns,
                columns=[f'PC{i+1}' for i in range(n_components)]
            )
            
            results = {
                'n_components': n_components,
                'explained_variance_ratio': pca_result.explained_variance_ratio_[:n_components],
                'cumulative_variance': cumulative_variance[:n_components],
                'loadings': loadings,
                'components': pca_result.components_[:n_components],
                'analysis_type': 'pca'
            }
            
            # 保存到缓存
            if self._cache_enabled:
                self._cache[cache_key] = results
            
            self.logger.info(f"PCA分析完成: 原始维度={X.shape[1]}, 降维后={n_components}")
            return results
            
        except Exception as e:
            self.logger.error(f"PCA分析失败: {str(e)}")
            raise


class FeatureImportanceAnalysisStrategy(FactorAnalysisStrategy):
    """特征重要性分析策略"""
    
    def __init__(self, config: FactorSelectionConfig):
        super().__init__(config)
        self._cache = {}
        self._memory_monitor_enabled = True
        self._memory_threshold_mb = 1500  # 内存阈值1.5GB
    
    def analyze(self, factors: pd.DataFrame, returns: Optional[pd.Series] = None) -> Dict[str, Any]:
        """执行特征重要性分析 - 优化版本（带缓存）"""
        try:
            # 检查缓存
            cache_key = self._get_cache_key(factors, returns, 'feature_importance')
            if self._cache_enabled and cache_key in self._cache:
                self.logger.info("从缓存加载特征重要性分析结果")
                return self._cache[cache_key]
            
            self.logger.info("开始特征重要性分析...")
            
            if returns is None:
                raise ValueError("特征重要性分析需要收益数据")
            
            # 预处理数据
            X = self._preprocess_factors(factors)
            y = returns.copy()
            
            # 如果特征数量过多，先进行初步筛选
            if X.shape[1] > 1000:
                self.logger.info("特征数量过多，进行初步筛选...")
                # 使用方差进行初步筛选，保留前500个高方差特征
                variances = X.var()
                top_features = variances.nlargest(500).index
                X = X[top_features]
            
            # 对齐索引并处理缺失值
            common_index = X.index.intersection(y.index)
            X = X.loc[common_index]
            y = y.loc[common_index]
            
            # 确保y为数值型
            y = pd.to_numeric(y, errors='coerce')
            y = y.replace([np.inf, -np.inf], np.nan)
            
            if y.notna().sum() == 0:
                raise ValueError("收益数据全部为缺失")
            
            y = y.fillna(y.median())
            
            # 优化标准化 - 使用更高效的实现
            scaler = StandardScaler()
            X_scaled = scaler.fit_transform(X)
            
            # 优化特征选择 - 直接使用互信息计算，避免SelectKBest的开销
            # 设置自适应参数来平衡精度和速度
            n_neighbors = min(3, len(y) // 10)  # 自适应调整邻居数量
            
            importance_scores = mutual_info_regression(
                X_scaled, y,
                n_neighbors=n_neighbors,
                random_state=42
            )
            
            # 创建重要性结果DataFrame
            importance_df = pd.DataFrame({
                'factor': X.columns,
                'importance_score': importance_scores
            }).sort_values('importance_score', ascending=False)
            
            # 限制选择的因子数量
            top_k = min(self.config.top_k_factors, len(importance_df))
            
            results = {
                'importance_df': importance_df,
                'top_factors': importance_df.head(top_k)['factor'].tolist(),
                'analysis_type': 'feature_importance'
            }
            
            # 保存到缓存
            if self._cache_enabled:
                self._cache[cache_key] = results
            
            self.logger.info(f"特征重要性分析完成: 分析因子数量={len(X.columns)}")
            return results
            
        except Exception as e:
            self.logger.error(f"特征重要性分析失败: {str(e)}")
            raise


class VarianceAnalysisStrategy(FactorAnalysisStrategy):
    """方差分析策略（降级方案）"""
    
    def __init__(self, config: FactorSelectionConfig):
        super().__init__(config)
        self._cache = {}
        self._memory_monitor_enabled = True
        self._memory_threshold_mb = 1500  # 内存阈值1.5GB
    
    def analyze(self, factors: pd.DataFrame, returns: Optional[pd.Series] = None) -> Dict[str, Any]:
        """执行方差分析 - 优化版本（带缓存）"""
        try:
            # 检查缓存
            cache_key = self._get_cache_key(factors, returns, 'variance')
            if self._cache_enabled and cache_key in self._cache:
                self.logger.info("从缓存加载方差分析结果")
                return self._cache[cache_key]
            
            self.logger.info("开始方差分析...")
            
            # 预处理数据
            X = self._preprocess_factors(factors)
            
            # 计算每个因子的方差
            variances = X.var()
            
            # 创建方差结果DataFrame
            variance_df = pd.DataFrame({
                'factor': variances.index,
                'variance': variances.values
            }).sort_values('variance', ascending=False)
            
            # 选择方差最大的因子
            top_factors = variance_df.head(self.config.top_k_factors)['factor'].tolist()
            
            results = {
                'variance_df': variance_df,
                'top_factors': top_factors,
                'analysis_type': 'variance'
            }
            
            # 保存到缓存
            if self._cache_enabled:
                self._cache[cache_key] = results
            
            self.logger.info(f"方差分析完成: 选择前 {len(top_factors)} 个高方差因子")
            return results
            
        except Exception as e:
            self.logger.error(f"方差分析失败: {str(e)}")
            raise


class FactorSelectionManager:
    """因子选择管理器 - 协调各种分析策略的综合选择"""
    
    def __init__(self, config: FactorSelectionConfig):
        self.config = config
        self.logger = logging.getLogger(f"{__name__}.FactorSelectionManager")
        self.data_loader = DataLoader(config)
        self.analysis_strategies = self._initialize_strategies()
        self.processing_log = {
            'start_time': datetime.now().isoformat(),
            'config': config.to_dict(),
            'steps': []
        }
        self._memory_monitor_enabled = True
        self._memory_threshold_mb = 2000  # 内存阈值2GB
        
    def _check_memory_usage(self) -> bool:
        """检查内存使用情况"""
        if not self._memory_monitor_enabled:
            return True
            
        try:
            process = psutil.Process(os.getpid())
            memory_mb = process.memory_info().rss / 1024 / 1024
            
            if memory_mb > self._memory_threshold_mb:
                self.logger.warning(f"内存使用过高: {memory_mb:.2f}MB, 超过阈值 {self._memory_threshold_mb}MB")
                return False
            return True
        except Exception:
            # 如果无法获取内存信息，继续执行
            return True
    
    def _cleanup_memory(self):
        """清理内存"""
        gc.collect()
        if hasattr(gc, 'collect'):
            gc.collect()  # 强制垃圾回收
    
    def _initialize_strategies(self) -> Dict[FactorSelectionMethod, FactorAnalysisStrategy]:
        """初始化分析策略"""
        strategies = {}
        
        strategy_map = {
            FactorSelectionMethod.CORRELATION: CorrelationAnalysisStrategy,
            FactorSelectionMethod.PCA: PCAAnalysisStrategy,
            FactorSelectionMethod.FEATURE_IMPORTANCE: FeatureImportanceAnalysisStrategy,
            FactorSelectionMethod.VARIANCE: VarianceAnalysisStrategy
        }
        
        for method in self.config.selection_methods:
            if method in strategy_map:
                strategies[method] = strategy_map[method](self.config)
        
        return strategies
    
    def _log_step(self, step_name: str, details: Dict[str, Any]):
        """记录处理步骤"""
        step_info = {
            'step': step_name,
            'timestamp': datetime.now().isoformat(),
            'details': details
        }
        self.processing_log['steps'].append(step_info)
        self.logger.info(f"完成步骤: {step_name}")
    
    def select_factors(self) -> Dict[str, Any]:
        """执行因子选择流程 - 优化版本（并行处理和内存优化）"""
        try:
            self.logger.info("开始因子选择流程...")
            
            # 检查内存使用情况
            if not self._check_memory_usage():
                self.logger.warning("内存使用过高，建议清理内存后再执行")
            
            # 步骤1: 加载数据
            factors, returns = self.data_loader.load_all_factors()
            self._log_step('data_loading', {
                'factors_shape': factors.shape,
                'returns_available': returns is not None
            })
            
            if factors.empty:
                raise ValueError("未加载到有效的因子数据")
            
            # 步骤2: 并行执行各种分析策略
            analysis_results = {}
            selected_factors = set()
            
            # 使用线程池并行执行分析策略
            with ThreadPoolExecutor(max_workers=min(4, len(self.analysis_strategies))) as executor:
                # 提交所有分析任务
                future_to_method = {}
                for method, strategy in self.analysis_strategies.items():
                    future = executor.submit(self._execute_analysis, strategy, factors, returns, method)
                    future_to_method[future] = method
                
                # 收集结果
                for future in as_completed(future_to_method):
                    method = future_to_method[future]
                    try:
                        result = future.result()
                        analysis_results[method.value] = result
                        
                        # 根据分析类型提取选择的因子
                        if method == FactorSelectionMethod.CORRELATION:
                            selected_factors.update(result.get('high_correlation_factors', []))
                        elif method == FactorSelectionMethod.FEATURE_IMPORTANCE:
                            selected_factors.update(result.get('top_factors', []))
                        elif method == FactorSelectionMethod.VARIANCE:
                            selected_factors.update(result.get('top_factors', []))
                        
                        self._log_step(f'{method.value}_analysis', {
                            'factors_analyzed': len(factors.columns),
                            'factors_selected': len(selected_factors)
                        })
                        
                        # 检查内存使用情况
                        if not self._check_memory_usage():
                            self.logger.warning("内存使用过高，清理内存...")
                            self._cleanup_memory()
                        
                    except Exception as e:
                        self.logger.warning(f"{method.value} 分析失败: {str(e)}")
                        analysis_results[method.value] = {'error': str(e)}
            
            # 步骤3: 综合选择最终因子
            final_factors = self._select_final_factors(selected_factors, factors)
            
            # 步骤4: 准备结果
            results = {
                'selected_factors': final_factors,
                'total_factors_analyzed': len(factors.columns),
                'final_factors_count': len(final_factors),
                'analysis_results': analysis_results,
                'processing_log': self.processing_log,
                'selection_config': self.config.to_dict()
            }
            
            self._log_step('final_selection', {
                'final_factors_count': len(final_factors),
                'selection_methods_used': [m.value for m in self.config.selection_methods]
            })
            
            # 清理临时数据
            del factors, returns, analysis_results, selected_factors
            self._cleanup_memory()
            
            self.logger.info(f"因子选择完成! 最终选择 {len(final_factors)} 个因子")
            return results
            
        except Exception as e:
            self.logger.error(f"因子选择流程失败: {str(e)}")
            raise
    
    def _execute_analysis(self, strategy: FactorAnalysisStrategy, factors: pd.DataFrame, 
                         returns: Optional[pd.Series], method: FactorSelectionMethod) -> Dict[str, Any]:
        """执行单个分析策略（用于并行处理）"""
        try:
            self.logger.info(f"开始并行执行 {method.value} 分析...")
            result = strategy.analyze(factors, returns)
            self.logger.info(f"{method.value} 分析完成")
            return result
        except Exception as e:
            self.logger.error(f"{method.value} 分析执行失败: {str(e)}")
            raise
    
    def _select_final_factors(self, selected_factors: set, factors: pd.DataFrame) -> List[str]:
        """综合选择最终因子"""
        # 如果没有任何分析策略成功选择因子，使用方差作为降级方案
        if not selected_factors:
            self.logger.warning("所有分析策略均未选择到因子，使用方差分析作为降级方案")
            variance_strategy = VarianceAnalysisStrategy(self.config)
            try:
                result = variance_strategy.analyze(factors)
                selected_factors = set(result.get('top_factors', []))
            except Exception as e:
                self.logger.error(f"方差分析降级方案也失败: {str(e)}")
                # 如果方差分析也失败，选择前top_k个因子
                selected_factors = set(factors.columns[:self.config.top_k_factors])
        
        # 转换为列表并排序
        final_factors = list(selected_factors)
        final_factors.sort()
        
        # 限制因子数量
        if len(final_factors) > self.config.top_k_factors:
            final_factors = final_factors[:self.config.top_k_factors]
            self.logger.info(f"因子数量超过限制，保留前 {self.config.top_k_factors} 个因子")
        
        return final_factors
    
    def save_results(self, results: Dict[str, Any], file_path: str):
        """保存选择结果到文件"""
        try:
            # 创建目录
            Path(file_path).parent.mkdir(parents=True, exist_ok=True)
            
            # 转换为可序列化的格式
            serializable_results = self._make_serializable(results)
            
            with open(file_path, 'w', encoding='utf-8') as f:
                json.dump(serializable_results, f, ensure_ascii=False, indent=2)
            
            self.logger.info(f"选择结果已保存到: {file_path}")
            
        except Exception as e:
            self.logger.error(f"保存选择结果失败: {str(e)}")
            raise
    
    def _make_serializable(self, obj: Any) -> Any:
        """将对象转换为可序列化的格式"""
        if isinstance(obj, pd.DataFrame):
            return obj.to_dict()
        elif isinstance(obj, pd.Series):
            return obj.to_dict()
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, (np.intc, np.intp, np.int8, np.int16, np.int32, np.int64)):
            return int(obj)
        elif isinstance(obj, (np.float16, np.float32, np.float64)):
            return float(obj)
        elif isinstance(obj, (np.bool_)):
            return bool(obj)
        elif isinstance(obj, dict):
            return {k: self._make_serializable(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [self._make_serializable(item) for item in obj]
        elif isinstance(obj, tuple):
            return tuple(self._make_serializable(item) for item in obj)
        elif isinstance(obj, FactorSelectionMethod):
            return obj.value
        elif hasattr(obj, 'to_dict'):
            return obj.to_dict()
        else:
            return obj


class FactorSelection:
    """因子选择器 - 重构后的主类"""
    
    def __init__(self, config: Optional[Union[Dict[str, Any], FactorSelectionConfig]] = None):
        """
        初始化因子选择器
        
        参数:
            config: 配置参数字典或FactorSelectionConfig对象
        """
        if isinstance(config, FactorSelectionConfig):
            self.config = config
        else:
            self.config = FactorSelectionConfig(**config) if config else FactorSelectionConfig()
        
        self.manager = FactorSelectionManager(self.config)
        self.logger = logging.getLogger(f"{__name__}.FactorSelection")
    
    def run_selection(self, output_file: Optional[str] = None) -> Dict[str, Any]:
        """
        运行因子选择流程
        
        参数:
            output_file: 结果输出文件路径
            
        返回:
            Dict[str, Any]: 选择结果
        """
        try:
            self.logger.info("启动因子选择流程...")
            
            # 执行选择流程
            results = self.manager.select_factors()
            
            # 保存结果
            if output_file:
                self.manager.save_results(results, output_file)
            
            self.logger.info("因子选择流程完成!")
            return results
            
        except Exception as e:
            self.logger.error(f"因子选择流程执行失败: {str(e)}")
            raise


def main():
    """主函数 - 用于命令行执行"""
    # 配置日志
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler('factor_selection.log', encoding='utf-8')
        ]
    )
    
    logger = logging.getLogger(__name__)
    
    try:
        # 创建配置
        config = FactorSelectionConfig(
            data_paths={
                'technical': 'data_pipeline/data/features/technical_indicators.csv',
                'fundamental': 'data_pipeline/data/features/fundamental_factors.csv',
                'macro': 'data_pipeline/data/features/macro_factors.csv',
                'epu': 'data_pipeline/data/features/epu_factors.csv',
                'prices': 'data_pipeline/data/daily_prices/Merge/hs300_daily_prices_merged.csv'
            },
            selection_methods=[
                FactorSelectionMethod.CORRELATION,
                FactorSelectionMethod.FEATURE_IMPORTANCE,
                FactorSelectionMethod.VARIANCE
            ],
            top_k_factors=50,
            correlation_threshold=0.7,
            pca_variance_threshold=0.95
        )
        
        # 创建因子选择器
        selector = FactorSelection(config)
        
        # 运行选择流程
        output_file = '../results/factor_selection_results.json'
        results = selector.run_selection(output_file)
        
        # 输出结果摘要
        logger.info("=" * 60)
        logger.info("因子选择结果摘要:")
        logger.info(f"分析因子总数: {results['total_factors_analyzed']}")
        logger.info(f"最终选择因子数: {results['final_factors_count']}")
        logger.info(f"选择方法: {results['selection_config']['selection_methods']}")
        logger.info(f"结果文件: {output_file}")
        logger.info("=" * 60)
        
        # 打印选择的因子
        logger.info("选择的因子列表:")
        for i, factor in enumerate(results['selected_factors'], 1):
            logger.info(f"{i:2d}. {factor}")
        
        return results
        
    except Exception as e:
        logger.error(f"因子选择程序执行失败: {str(e)}")
        raise


if __name__ == "__main__":
    main()