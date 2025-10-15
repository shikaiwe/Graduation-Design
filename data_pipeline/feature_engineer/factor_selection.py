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
            
            # 加载技术指标因子
            tech_factors = pd.read_csv("../data/features/technical_indicators.csv")
            
            # 加载基本因子
            fundamental_factors = pd.read_csv("../data/features/fundamental_factors.csv")
            
            # 加载宏观因子
            macro_factors = pd.read_csv("../data/features/macro_factors.csv")
            
            # 加载EPU因子
            epu_factors = pd.read_csv("../data/features/epu_factors.csv")
            
            # 合并所有因子数据
            all_factors = tech_factors.copy()
            
            # 合并基本因子
            if not fundamental_factors.empty:
                all_factors = pd.merge(all_factors, fundamental_factors, 
                                     on=['日期', '股票代码'], how='left')
            
            # 合并宏观因子（需要处理日期格式）
            if not macro_factors.empty:
                macro_factors['日期'] = pd.to_datetime(macro_factors['日期'])
                all_factors['日期'] = pd.to_datetime(all_factors['日期'])
                all_factors = pd.merge_asof(all_factors.sort_values('日期'), 
                                          macro_factors.sort_values('日期'), 
                                          on='日期', direction='backward')
            
            # 合并EPU因子
            if not epu_factors.empty:
                epu_factors['日期'] = pd.to_datetime(epu_factors['日期'])
                all_factors = pd.merge_asof(all_factors.sort_values('日期'), 
                                          epu_factors.sort_values('日期'), 
                                          on='日期', direction='backward')
            
            # 提取收益数据（假设收益率列名为'收益率'）
            if '收益率' in all_factors.columns:
                returns = all_factors['收益率']
                factors = all_factors.drop(columns=['收益率', '日期', '股票代码'], errors='ignore')
            else:
                returns = None
                factors = all_factors.drop(columns=['日期', '股票代码'], errors='ignore')
            
            # 清理数据
            factors = factors.dropna(axis=1, how='all')  # 删除全空列
            factors = factors.fillna(method='ffill').fillna(method='bfill')  # 填充空值
            
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
            
            # 数据标准化
            scaler = StandardScaler()
            factors_scaled = scaler.fit_transform(factors)
            
            # 进行PCA
            pca = PCA()
            pca_result = pca.fit(factors_scaled)
            
            # 计算累计方差解释
            cumulative_variance = np.cumsum(pca_result.explained_variance_ratio_)
            
            # 确定达到方差阈值的成分数量
            n_components = np.argmax(cumulative_variance >= self.config['pca_variance_threshold']) + 1
            
            # 获取主成分载荷
            loadings = pd.DataFrame(
                pca_result.components_[:n_components].T,
                index=factors.columns,
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
                'variance_explained': cumulative_variance[n_components-1]
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
            
            # 使用互信息进行特征选择
            selector = SelectKBest(score_func=mutual_info_regression, k='all')
            selector.fit(factors, returns)
            
            # 获取特征重要性得分
            importance_scores = selector.scores_
            
            # 创建重要性结果DataFrame
            importance_df = pd.DataFrame({
                'factor': factors.columns,
                'importance_score': importance_scores
            }).sort_values('importance_score', ascending=False)
            
            # 选择Top K因子
            top_k_factors = importance_df.head(self.config['top_k_factors'])
            
            self._log_step('feature_importance_analysis', {
                'total_factors': len(factors.columns),
                'top_k_selected': len(top_k_factors),
                'max_importance': importance_scores.max()
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
            
            # 相关性分析选择
            if 'correlation' in self.config['selection_methods']:
                corr_results = self.correlation_analysis(factors, returns)
                high_corr_factors = corr_results.head(self.config['top_k_factors'])['factor'].tolist()
                selected_factors.update(high_corr_factors)
            
            # 特征重要性选择
            if 'feature_importance' in self.config['selection_methods']:
                importance_results = self.feature_importance_analysis(factors, returns)
                important_factors = importance_results.head(self.config['top_k_factors'])['factor'].tolist()
                selected_factors.update(important_factors)
            
            # PCA分析（用于降维，不直接选择因子）
            if 'pca' in self.config['selection_methods']:
                pca_results = self.pca_analysis(factors)
                # PCA主要用于理解数据结构，不直接选择因子
            
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