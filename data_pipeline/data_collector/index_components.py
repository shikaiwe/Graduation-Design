# -*- coding: utf-8 -*-
"""
沪深300成分股获取模块
功能：
1. 获取最新成分股列表
2. 历史成分股记录查询
3. 成分股变更记录追踪
"""
import akshare as ak
import pandas as pd
from pathlib import Path
import logging

# 配置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class IndexComponents:
    """沪深300成分股获取类"""
    
    def __init__(self, data_dir: str = "data"):
        """
        初始化成分股获取类
        
        Args:
            data_dir: 数据存储目录
        """
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(parents=True, exist_ok=True)
        # 创建index_components子目录
        self.index_components_dir = self.data_dir / "index_components"
        self.index_components_dir.mkdir(parents=True, exist_ok=True)
        self.components_file = self.index_components_dir / "hs300_components.csv"
    
    def get_latest_components(self) -> pd.DataFrame:
        """
        获取最新的沪深300成分股列表
        
        Returns:
            pd.DataFrame: 成分股数据，包含代码、名称等信息
        """
        try:
            # 使用正确的AKShare接口获取沪深300成分股
            components_df = ak.index_stock_cons_csindex(symbol="000300")
            
            if components_df.empty:
                logger.warning("未获取到成分股数据，尝试备用接口")
                # 备用接口
                components_df = ak.index_stock_cons(symbol="000300")
            
            if not components_df.empty:
                logger.info(f"成功获取到 {len(components_df)} 只成分股")
                # 保存数据
                components_df.to_csv(self.components_file, index=False, encoding='utf-8-sig')
                return components_df
            else:
                logger.error("所有接口均未获取到成分股数据")
                return pd.DataFrame()
                
        except Exception as e:
            logger.error(f"获取成分股失败: {e}")
            return pd.DataFrame()
    
    def load_components(self) -> pd.DataFrame:
        """
        从本地文件加载成分股数据
        
        Returns:
            pd.DataFrame: 成分股数据
        """
        if self.components_file.exists():
            try:
                components_df = pd.read_csv(self.components_file, encoding='utf-8-sig')
                logger.info(f"从本地加载 {len(components_df)} 只成分股")
                return components_df
            except Exception as e:
                logger.error(f"加载本地成分股数据失败: {e}")
                return pd.DataFrame()
        else:
            logger.warning("本地成分股文件不存在")
            return pd.DataFrame()
    
    def get_components(self, use_cache: bool = True) -> pd.DataFrame:
        """
        获取成分股数据，优先使用缓存
        
        Args:
            use_cache: 是否使用缓存数据
            
        Returns:
            pd.DataFrame: 成分股数据
        """
        if use_cache:
            cached_data = self.load_components()
            if not cached_data.empty:
                return cached_data
        
        return self.get_latest_components()

if __name__ == "__main__":
    # 测试代码
    index_components = IndexComponents()
    components = index_components.get_components()
    
    if not components.empty:
        print("沪深300成分股前10只:")
        print(components.head(10))
        print(f"\n总成分股数量: {len(components)}")
        print(f"数据列: {list(components.columns)}")
    else:
        print("未获取到成分股数据")