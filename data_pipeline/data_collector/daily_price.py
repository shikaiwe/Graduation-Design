import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent.parent))

import akshare as ak
import pandas as pd
from pathlib import Path
import logging
from datetime import datetime, timedelta
from tqdm import tqdm
from data_collector.index_components import IndexComponents
from data_collector.request_controller import RequestController, RequestConfig

# 配置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class DailyPriceCollector:
    """日频价格数据采集类"""
    
    def __init__(self, data_dir: str = "data"):
        """
        初始化日频数据采集类
        
        Args:
            data_dir: 数据存储目录
        """
        self.data_dir = Path(data_dir)
        self.price_dir = self.data_dir / "daily_prices"
        self.price_dir.mkdir(parents=True, exist_ok=True)
        
        # 获取成分股数据
        self.index_components = IndexComponents(data_dir)
        
        # 初始化智能请求控制器
        self.request_config = RequestConfig(
            base_delay=2.0,           # 基础延迟2秒
            max_delay=30.0,           # 最大延迟30秒
            max_retries=5,            # 最大重试5次
            backoff_factor=2.0,       # 退避因子2倍
            jitter=0.2,               # 20%随机抖动
            batch_size=30,            # 批次大小30
            requests_per_minute=20    # 每分钟20个请求
        )
        self.request_controller = RequestController(self.request_config)
        
        # 停止标志，用于GUI控制
        self.stop_requested = False
    
    def get_stock_data(self, symbol: str, start_date: str = None, end_date: str = None) -> pd.DataFrame:
        """
        获取单只股票的历史数据
        
        Args:
            symbol: 股票代码
            start_date: 开始日期，格式YYYYMMDD
            end_date: 结束日期，格式YYYYMMDD
            
        Returns:
            pd.DataFrame: 股票历史数据
        """
        def _get_stock_data():
            """内部函数，用于重试机制"""
            # 使用nonlocal声明变量，避免闭包问题
            nonlocal start_date, end_date
            
            # 如果没有指定日期，默认获取最近一年的数据
            if not end_date:
                end_date = datetime.now().strftime("%Y%m%d")
            if not start_date:
                start_date = (datetime.now() - timedelta(days=365)).strftime("%Y%m%d")
            
            # 根据股票代码确定市场
            if symbol.startswith('6'):
                market = 'sh'  # 上海
            elif symbol.startswith('0') or symbol.startswith('3'):
                market = 'sz'  # 深圳
            elif symbol.startswith('688') or symbol.startswith('689'):
                market = 'sh'  # 科创板
            else:
                market = 'sh'  # 默认上海
            
            # 构造完整的股票代码
            full_symbol = f"{market}{symbol}"
            
            # 使用AKShare获取股票历史数据
            stock_df = None
            
            # 首先尝试使用stock_zh_a_hist接口（需要纯数字股票代码，如000001）
            try:
                stock_df = ak.stock_zh_a_hist(symbol=symbol, period="daily", 
                                            start_date=start_date, end_date=end_date, 
                                            adjust="qfq")  # 前复权
                
                # 检查返回数据是否有效（包括检查是否为空的DataFrame）
                if stock_df is None or stock_df.empty or (hasattr(stock_df, 'shape') and stock_df.shape[0] == 0):
                    logger.warning(f"stock_zh_a_hist接口返回空数据，尝试使用stock_zh_a_daily接口")
                    stock_df = None
                else:
                    logger.info(f"成功使用stock_zh_a_hist接口获取股票 {symbol} 数据，共 {len(stock_df)} 条记录")
                    
            except Exception as e:
                logger.warning(f"stock_zh_a_hist接口失败: {e}，尝试使用stock_zh_a_daily接口")
                stock_df = None
            
            # 如果stock_zh_a_hist失败，尝试使用stock_zh_a_daily接口
            if stock_df is None:
                try:
                    stock_df = ak.stock_zh_a_daily(symbol=full_symbol, adjust="qfq")
                    
                    # 检查返回数据是否有效
                    if stock_df is None or stock_df.empty:
                        logger.error(f"stock_zh_a_daily接口也返回空数据")
                        return pd.DataFrame()
                    
                    # 如果指定了日期范围，进行过滤
                    if start_date and end_date:
                        # 确保日期列存在且格式正确
                        if 'date' in stock_df.columns:
                            # 修复日期过滤问题：需要先将字符串日期转换为datetime格式
                            stock_df['date'] = pd.to_datetime(stock_df['date'])
                            start_date_dt = pd.to_datetime(start_date)
                            end_date_dt = pd.to_datetime(end_date)
                            stock_df = stock_df[(stock_df['date'] >= start_date_dt) & (stock_df['date'] <= end_date_dt)]
                        elif '日期' in stock_df.columns:
                            # 对于中文列名，也需要进行日期格式转换
                            stock_df['日期'] = pd.to_datetime(stock_df['日期'])
                            start_date_dt = pd.to_datetime(start_date)
                            end_date_dt = pd.to_datetime(end_date)
                            stock_df = stock_df[(stock_df['日期'] >= start_date_dt) & (stock_df['日期'] <= end_date_dt)]
                    
                    logger.info(f"成功使用stock_zh_a_daily接口获取股票 {symbol} 数据，共 {len(stock_df)} 条记录")
                    
                except Exception as e2:
                    logger.error(f"所有接口均失败: {e2}")
                    return pd.DataFrame()
            
            # 统一数据列名和格式
            if not stock_df.empty:
                # 确保日期列名统一为'日期'
                if 'date' in stock_df.columns:
                    stock_df = stock_df.rename(columns={'date': '日期'})
                
                # 确保股票代码列存在且正确
                if '股票代码' not in stock_df.columns:
                    stock_df['股票代码'] = symbol
                
                stock_df['市场'] = market
                
                # 格式化成交额为整数格式，避免科学计数法
                if '成交额' in stock_df.columns:
                    stock_df['成交额'] = stock_df['成交额'].astype('int64')
                
                # 确保日期格式正确
                if '日期' in stock_df.columns:
                    # 如果日期是字符串格式，确保格式正确
                    if stock_df['日期'].dtype == 'object':
                        try:
                            # 尝试转换为标准日期格式
                            stock_df['日期'] = pd.to_datetime(stock_df['日期']).dt.strftime('%Y%m%d')
                        except:
                            pass
                
                logger.debug(f"数据预处理完成，股票 {symbol} 共 {len(stock_df)} 条记录")
            
            return stock_df
        
        try:
            # 使用智能请求控制器执行
            return self.request_controller.execute_with_retry(_get_stock_data)
            
        except Exception as e:
            logger.error(f"获取股票 {symbol} 数据最终失败: {e}")
            return pd.DataFrame()
    
    def batch_get_stock_data(self, symbols: list, start_date: str = None, end_date: str = None, 
                           max_workers: int = 5, use_multithreading: bool = True) -> pd.DataFrame:
        """
        批量获取多只股票数据（支持多线程智能请求控制）
        
        Args:
            symbols: 股票代码列表
            start_date: 开始日期，格式YYYYMMDD
            end_date: 结束日期，格式YYYYMMDD
            max_workers: 最大并发线程数
            use_multithreading: 是否使用多线程
            
        Returns:
            pd.DataFrame: 合并后的股票数据
        """
        def _process_symbol(symbol):
            """处理单个股票的函数"""
            # 检查是否请求停止
            if self.stop_requested:
                logger.info(f"用户请求停止，跳过股票 {symbol}")
                return None
                
            stock_data = self.get_stock_data(symbol, start_date, end_date)
            return stock_data if not stock_data.empty else None
        
        # 使用智能批量执行（支持多线程）
        results = self.request_controller.batch_execute(
            symbols, 
            _process_symbol, 
            progress_desc="获取股票数据",
            use_multithreading=use_multithreading,
            max_workers=max_workers
        )
        
        # 过滤掉空结果
        valid_results = [result for result in results if result is not None]
        
        if valid_results:
            combined_df = pd.concat(valid_results, ignore_index=True)
            logger.info(f"成功获取 {len(valid_results)} 只股票数据，共 {len(combined_df)} 条记录")
            
            # 输出请求统计信息
            stats = self.request_controller.get_statistics()
            logger.info(f"请求统计: 总请求 {stats['total_requests']}, 失败 {stats['failed_requests']}, 成功率 {stats['success_rate']:.2%}")
            
            return combined_df
        else:
            logger.warning("未获取到任何股票数据")
            return pd.DataFrame()
    
    def save_daily_prices(self, df: pd.DataFrame, date_str: str = None):
        """
        保存日频价格数据
        
        Args:
            df: 股票数据DataFrame
            date_str: 日期字符串，用于文件名
        """
        if df.empty:
            logger.warning("空数据，不进行保存")
            return
        
        if not date_str:
            date_str = datetime.now().strftime("%Y%m%d")
        
        filename = self.price_dir / f"daily_prices_{date_str}.parquet"
        df.to_parquet(filename, index=False)
        logger.info(f"日频价格数据已保存至: {filename}")
    
    def collect_hs300_daily_data(self, sample_count: int = 10, start_date: str = None, end_date: str = None,
                              use_multithreading: bool = True, max_workers: int = 5):
        """
        采集沪深300成分股的日频数据（支持多线程）
        
        Args:
            sample_count: 采样数量，用于测试
            start_date: 开始日期，格式YYYYMMDD
            end_date: 结束日期，格式YYYYMMDD
            use_multithreading: 是否使用多线程
            max_workers: 最大并发线程数
        """
        # 获取成分股列表
        components = self.index_components.get_components()
        
        if components.empty:
            logger.error("未获取到成分股数据，无法继续")
            return
        
        logger.info(f"获取到 {len(components)} 只沪深300成分股")
        
        # 提取股票代码（使用正确的列名）
        stock_symbols = components['成分券代码'].tolist()
        
        # 采样测试
        if sample_count > 0:
            stock_symbols = stock_symbols[:sample_count]
            logger.info(f"采样前 {sample_count} 只股票进行测试")
        
        # 批量获取股票数据（支持多线程）
        stock_data = self.batch_get_stock_data(stock_symbols, start_date, end_date, 
                                             max_workers=max_workers, 
                                             use_multithreading=use_multithreading)
        
        if stock_data is not None and not stock_data.empty:
            # 生成文件名
            if start_date and end_date:
                date_str = f"{start_date}_{end_date}"
            else:
                date_str = datetime.now().strftime("%Y%m%d")
            # 保存数据
            self.save_daily_prices(stock_data, date_str)
            
            # 显示数据统计信息
            logger.info(f"数据采集完成，共获取 {len(stock_data)} 条记录")
            logger.info(f"数据时间范围: {stock_data['日期'].min()} 至 {stock_data['日期'].max()}")
            logger.info(f"覆盖股票数量: {stock_data['股票代码'].nunique()}")
        else:
            logger.error("未获取到任何股票数据")
    
    def collect_hs300_historical_data_resume(self, start_year=2019, end_year=2024, progress_callback=None):
        """
        断点续传方式获取沪深300历史数据
        
        Args:
            start_year: 开始年份
            end_year: 结束年份
            progress_callback: 进度回调函数，接收当前进度和总进度
        """
        logger.info("开始断点续传数据采集...")
        
        # 检查现有数据文件
        existing_files = list(self.price_dir.glob("daily_prices_*.parquet"))
        
        if not existing_files:
            logger.info("未找到现有数据文件，开始全新采集")
            return self.collect_hs300_historical_data(start_year, end_year, progress_callback)
        
        # 获取最新数据文件
        latest_file = max(existing_files, key=lambda x: x.stat().st_mtime)
        logger.info(f"检测到最新数据文件: {latest_file.name}")
        
        try:
            # 读取现有数据
            existing_df = pd.read_parquet(latest_file)
            
            # 获取已采集的股票代码
            collected_symbols = set(existing_df['股票代码'].unique()) if '股票代码' in existing_df.columns else set()
            logger.info(f"已采集股票数量: {len(collected_symbols)}")
            
            # 获取完整的沪深300成分股列表
            components = self.index_components.get_components()
            all_symbols = set(components['成分券代码'].tolist())
            logger.info(f"沪深300成分股总数: {len(all_symbols)}")
            
            # 计算需要采集的剩余股票
            remaining_symbols = list(all_symbols - collected_symbols)
            
            if not remaining_symbols:
                logger.info("所有股票数据已采集完成，无需继续采集")
                if progress_callback:
                    progress_callback(100, 100)
                return
            
            logger.info(f"剩余待采集股票数量: {len(remaining_symbols)}")
            logger.info(f"剩余股票代码示例: {remaining_symbols[:10]}")
            
            # 分批采集剩余股票数据
            batch_size = self.request_config.batch_size
            total_batches = (len(remaining_symbols) + batch_size - 1) // batch_size
            
            for i in range(0, len(remaining_symbols), batch_size):
                batch_symbols = remaining_symbols[i:i + batch_size]
                batch_num = i // batch_size + 1
                
                logger.info(f"正在采集第 {batch_num}/{total_batches} 批次，股票数量: {len(batch_symbols)}")
                
                # 更新进度
                if progress_callback:
                    progress_callback(batch_num, total_batches)
                
                # 获取批次数据
                batch_data = self.batch_get_stock_data(batch_symbols, start_year, end_year)
                
                if batch_data is not None and not batch_data.empty:
                    # 合并数据
                    combined_df = pd.concat([existing_df, batch_data], ignore_index=True)
                    
                    # 保存合并后的数据
                    filename = f"daily_prices_{start_year}_{end_year}.parquet"
                    filepath = self.price_dir / filename
                    combined_df.to_parquet(filepath, index=False)
                    
                    # 更新现有数据引用
                    existing_df = combined_df
                    collected_symbols.update(batch_symbols)
                    
                    logger.info(f"第 {batch_num} 批次数据采集完成，已保存")
                    logger.info(f"当前已采集股票总数: {len(collected_symbols)}")
                    logger.info(f"剩余股票数量: {len(all_symbols - collected_symbols)}")
                else:
                    logger.warning(f"第 {batch_num} 批次数据获取失败")
            
            logger.info("断点续传数据采集完成")
            
        except Exception as e:
            logger.error(f"断点续传过程中出现错误: {e}")
            raise
    
    def collect_hs300_historical_data(self, start_year: int = 2019, end_year: int = 2024, progress_callback=None):
        """
        采集2019-2024年沪深300成分股的历史数据（使用智能请求控制）
        
        Args:
            start_year: 开始年份
            end_year: 结束年份
            progress_callback: 进度回调函数，接收当前进度和总进度
        """
        logger.info(f"开始采集 {start_year} 年至 {end_year} 年沪深300历史数据")
        
        # 设置日期范围
        start_date = f"{start_year}0101"
        end_date = f"{end_year}1231"
        
        # 获取成分股列表
        components = self.index_components.get_components()
        
        if components.empty:
            logger.error("未获取到成分股数据，无法继续")
            return
        
        logger.info(f"获取到 {len(components)} 只沪深300成分股")
        
        # 提取股票代码
        stock_symbols = components['成分券代码'].tolist()
        
        # 使用智能批量处理，不再需要手动分批和延迟
        stock_data = self.batch_get_stock_data(stock_symbols, start_date, end_date)
        
        if not stock_data.empty:
            # 保存数据
            date_str = f"{start_year}_{end_year}"
            self.save_daily_prices(stock_data, date_str)
            
            # 显示最终统计信息
            logger.info(f"历史数据采集完成，共获取 {len(stock_data)} 条记录")
            logger.info(f"数据时间范围: {stock_data['日期'].min()} 至 {stock_data['日期'].max()}")
            logger.info(f"覆盖股票数量: {stock_data['股票代码'].nunique()}")
            logger.info(f"数据文件: daily_prices_{date_str}.parquet")
            
            # 更新进度为完成
            if progress_callback:
                progress_callback(100, 100)
        else:
            logger.error("未获取到任何历史数据")

if __name__ == "__main__":
    # 测试代码
    collector = DailyPriceCollector()
    
    # 使用断点续传功能继续获取数据
    collector.collect_hs300_historical_data_resume(start_year=2019, end_year=2024)
    
    # 测试历史数据采集（从头开始）
    # collector.collect_hs300_historical_data(start_year=2019, end_year=2024)
    
    # 采集前10只沪深300成分股的日频数据
    # collector.collect_hs300_daily_data(sample_count=10)