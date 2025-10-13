"""
沪深300成分股日频数据采集器
使用AKshare接口获取2019-2024年数据，支持分批处理和频率控制
"""

import os
import time
import logging
import pandas as pd
from datetime import datetime, timedelta
import akshare as ak
from typing import List, Dict, Optional
import traceback
from tqdm import tqdm

# 导入配置
import config


class HS300DataCollector:
    """沪深300成分股数据采集器"""
    
    def __init__(self):
        """初始化数据采集器"""
        self.setup_logging()
        self.components_df = None
        self.processed_stocks = set()
        self.failed_stocks = []
        
    def setup_logging(self):
        """设置日志配置"""
        # 创建日志目录
        log_dir = os.path.dirname(config.LOG_CONFIG['filename'])
        os.makedirs(log_dir, exist_ok=True)
        
        logging.basicConfig(
            level=getattr(logging, config.LOG_CONFIG['level']),
            format=config.LOG_CONFIG['format'],
            filename=config.LOG_CONFIG['filename'],
            filemode='a'
        )
        self.logger = logging.getLogger(__name__)
        
        # 同时输出到控制台
        console_handler = logging.StreamHandler()
        console_handler.setLevel(getattr(logging, config.LOG_CONFIG['level']))
        formatter = logging.Formatter(config.LOG_CONFIG['format'])
        console_handler.setFormatter(formatter)
        self.logger.addHandler(console_handler)
        
        self.logger.info("HS300数据采集器初始化完成")
    
    def load_components(self) -> pd.DataFrame:
        """
        加载沪深300成分股列表
        
        Returns:
            pd.DataFrame: 成分股数据框
        """
        try:
            # 确保股票代码列被读取为字符串类型
            self.components_df = pd.read_csv(config.COMPONENTS_FILE, dtype={'股票代码': str})
            self.logger.info(f"成功加载 {len(self.components_df)} 只成分股")
            return self.components_df
        except Exception as e:
            self.logger.error(f"加载成分股文件失败: {e}")
            raise
    
    def get_stock_data(self, stock_code: str, stock_name: str, 
                      start_date: str, end_date: str) -> Optional[pd.DataFrame]:
        """
        获取单只股票的日频数据
        
        Args:
            stock_code (str): 股票代码
            stock_name (str): 股票名称
            start_date (str): 开始日期，格式 'YYYY-MM-DD'
            end_date (str): 结束日期，格式 'YYYY-MM-DD'
            
        Returns:
            Optional[pd.DataFrame]: 股票数据框，失败返回None
        """
        symbol = config.get_stock_symbol(stock_code)
        
        for attempt in range(config.MAX_RETRIES):
            try:
                self.logger.info(f"获取股票数据: {stock_code} {stock_name} (尝试 {attempt + 1}/{config.MAX_RETRIES})")
                
                # 使用AKshare获取股票日频数据
                # 注意：AKshare接口要求日期格式为YYYYMMDD，需要转换
                start_date_ak = start_date.replace("-", "")
                end_date_ak = end_date.replace("-", "")
                
                stock_data = ak.stock_zh_a_hist(symbol=symbol, period="daily", 
                                              start_date=start_date_ak, end_date=end_date_ak,
                                              adjust="qfq")  # 前复权
                
                if stock_data.empty:
                    self.logger.warning(f"股票 {stock_code} {stock_name} 在指定时间段内无数据")
                    return None
                
                # 重命名列
                stock_data = stock_data.rename(columns=config.FIELD_MAPPING)
                
                # 添加股票信息
                stock_data['stock_code'] = stock_code
                stock_data['stock_name'] = stock_name
                stock_data['symbol'] = symbol
                
                self.logger.info(f"成功获取 {stock_code} {stock_name} 的 {len(stock_data)} 条数据")
                return stock_data
                
            except Exception as e:
                self.logger.warning(f"获取股票 {stock_code} 数据失败 (尝试 {attempt + 1}): {e}")
                
                if attempt < config.MAX_RETRIES - 1:
                    time.sleep(config.RETRY_DELAY)
                else:
                    self.logger.error(f"股票 {stock_code} 数据获取最终失败: {e}")
                    self.failed_stocks.append({
                        'stock_code': stock_code,
                        'stock_name': stock_name,
                        'error': str(e)
                    })
                    return None
    
    def save_batch_data(self, batch_data: List[pd.DataFrame], batch_num: int):
        """
        保存批次数据到CSV文件
        
        Args:
            batch_data (List[pd.DataFrame]): 批次数据列表
            batch_num (int): 批次编号
        """
        if not batch_data:
            self.logger.warning(f"批次 {batch_num} 无数据可保存")
            return
        
        # 合并批次数据
        combined_data = pd.concat(batch_data, ignore_index=True)
        
        # 创建输出目录
        os.makedirs(config.OUTPUT_DIR, exist_ok=True)
        
        # 生成文件名
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"hs300_daily_prices_batch_{batch_num:03d}_{timestamp}.csv"
        filepath = os.path.join(config.OUTPUT_DIR, filename)
        
        # 保存数据
        combined_data.to_csv(filepath, index=False, encoding='utf-8-sig')
        
        self.logger.info(f"批次 {batch_num} 数据已保存: {filepath} (包含 {len(combined_data)} 条记录)")
        
        # 记录已处理的股票
        batch_stocks = combined_data[['stock_code', 'stock_name']].drop_duplicates()
        for _, stock in batch_stocks.iterrows():
            self.processed_stocks.add((stock['stock_code'], stock['stock_name']))
    
    def process_batch(self, stock_batch: List[Dict], batch_num: int, total_batches: int):
        """
        处理一个批次的股票数据
        
        Args:
            stock_batch (List[Dict]): 批次股票列表
            batch_num (int): 批次编号
            total_batches (int): 总批次数量
        """
        self.logger.info(f"开始处理批次 {batch_num}/{total_batches}，包含 {len(stock_batch)} 只股票")
        
        batch_data = []
        
        # 创建批次进度条
        batch_desc = f"批次 {batch_num}/{total_batches}"
        with tqdm(stock_batch, desc=batch_desc, unit="股票", leave=False) as batch_pbar:
            for stock in batch_pbar:
                stock_code = stock['股票代码']
                stock_name = stock['股票名称']
                
                # 更新进度条描述
                batch_pbar.set_postfix({
                    "股票": f"{stock_code} {stock_name}",
                    "成功": len(batch_data)
                })
                
                # 检查是否已处理
                if (stock_code, stock_name) in self.processed_stocks:
                    batch_pbar.set_description(f"{batch_desc} - 跳过已处理")
                    continue
                
                # 获取股票数据
                stock_data = self.get_stock_data(stock_code, stock_name, config.START_DATE, config.END_DATE)
                
                if stock_data is not None:
                    batch_data.append(stock_data)
                    batch_pbar.set_description(f"{batch_desc} - 成功获取")
                else:
                    batch_pbar.set_description(f"{batch_desc} - 获取失败")
                
                # 请求间隔，避免频率限制
                time.sleep(config.REQUEST_DELAY)
        
        # 保存批次数据
        self.save_batch_data(batch_data, batch_num)
        
        self.logger.info(f"批次 {batch_num}/{total_batches} 处理完成，成功获取 {len(batch_data)} 只股票数据")
    
    def run(self):
        """运行数据采集程序"""
        self.logger.info("开始沪深300成分股数据采集")
        self.logger.info(f"时间范围: {config.START_DATE} 到 {config.END_DATE}")
        
        try:
            # 加载成分股
            components_df = self.load_components()
            
            # 分批处理
            total_stocks = len(components_df)
            num_batches = (total_stocks + config.BATCH_SIZE - 1) // config.BATCH_SIZE
            
            self.logger.info(f"总共 {total_stocks} 只股票，分为 {num_batches} 个批次处理")
            
            # 创建总体进度条
            with tqdm(total=num_batches, desc="总体进度", unit="批次", position=0) as main_pbar:
                for batch_num in range(num_batches):
                    start_idx = batch_num * config.BATCH_SIZE
                    end_idx = min((batch_num + 1) * config.BATCH_SIZE, total_stocks)
                    
                    stock_batch = components_df.iloc[start_idx:end_idx].to_dict('records')
                    
                    # 更新总体进度条描述
                    main_pbar.set_description(f"总体进度 - 批次 {batch_num + 1}/{num_batches}")
                    main_pbar.set_postfix({
                        "已处理股票": len(self.processed_stocks),
                        "失败股票": len(self.failed_stocks)
                    })
                    
                    # 处理当前批次
                    self.process_batch(stock_batch, batch_num + 1, num_batches)
                    
                    # 更新总体进度
                    main_pbar.update(1)
                    
                    # 批次间间隔
                    if batch_num < num_batches - 1:
                        main_pbar.set_description(f"总体进度 - 等待下一批次")
                        time.sleep(config.REQUEST_DELAY)
            
            # 输出统计信息
            self.print_statistics()
            
        except Exception as e:
            self.logger.error(f"数据采集过程中发生错误: {e}")
            self.logger.error(traceback.format_exc())
    
    def print_statistics(self):
        """打印采集统计信息"""
        self.logger.info("=" * 50)
        self.logger.info("数据采集统计信息:")
        self.logger.info(f"成功处理股票数量: {len(self.processed_stocks)}")
        self.logger.info(f"失败股票数量: {len(self.failed_stocks)}")
        
        if self.failed_stocks:
            self.logger.info("失败的股票列表:")
            for stock in self.failed_stocks:
                self.logger.info(f"  {stock['stock_code']} {stock['stock_name']}: {stock['error']}")
        
        self.logger.info("=" * 50)


def main():
    """主函数"""
    collector = HS300DataCollector()
    collector.run()


if __name__ == "__main__":
    main()