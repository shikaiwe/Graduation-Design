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
        self.error_occurred = False  # 标记是否发生错误
        self.current_batch_data = []  # 当前批次已成功获取的数据
        
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
        
        # 检查已处理的数据文件
        self.load_processed_stocks()
    
    def load_processed_stocks(self):
        """
        检查已处理的数据文件，加载已成功获取的股票信息
        """
        try:
            # 确保processed_stocks属性已存在
            if not hasattr(self, 'processed_stocks'):
                self.processed_stocks = set()
                
            if not os.path.exists(config.OUTPUT_DIR):
                self.logger.info("输出目录不存在，无需检查已处理数据")
                return
                
            # 查找所有批次数据文件
            import glob
            pattern = os.path.join(config.OUTPUT_DIR, "hs300_daily_prices_batch_*.csv")
            data_files = glob.glob(pattern)
            
            if not data_files:
                self.logger.info("未找到已处理的数据文件")
                return
                
            self.logger.info(f"找到 {len(data_files)} 个已处理的数据文件")
            
            for file_path in data_files:
                try:
                    # 读取文件的第一行获取列名，避免读取整个文件
                    with open(file_path, 'r', encoding='utf-8-sig') as f:
                        first_line = f.readline().strip()
                        
                    # 检查文件是否包含必要的列（支持多种列名格式）
                    columns = first_line.split(',')
                    code_col = None
                    name_col = None
                    
                    # 确定股票代码和名称列名
                    if 'stock_code' in columns and 'stock_name' in columns:
                        code_col, name_col = 'stock_code', 'stock_name'
                    elif '股票代码' in columns and 'stock_name' in columns:
                        code_col, name_col = '股票代码', 'stock_name'
                    elif '股票代码' in columns and '股票名称' in columns:
                        code_col, name_col = '股票代码', '股票名称'
                    
                    if code_col and name_col:
                        # 只读取股票代码和名称列
                        df = pd.read_csv(file_path, usecols=[code_col, name_col])
                        
                        # 添加到已处理股票集合
                        for _, row in df.iterrows():
                            self.processed_stocks.add((str(row[code_col]), row[name_col]))
                        self.logger.debug(f"成功处理文件 {file_path}，添加 {len(df)} 只股票")
                    else:
                        self.logger.warning(f"文件 {file_path} 缺少必要的股票代码和名称列")
                            
                except Exception as e:
                    self.logger.warning(f"读取已处理文件 {file_path} 失败: {e}")
                    continue
            
            self.logger.info(f"已加载 {len(self.processed_stocks)} 只已处理股票信息")
            
        except Exception as e:
            # 更详细的错误信息
            error_msg = f"检查已处理数据失败: {e}"
            if hasattr(e, '__class__'):
                error_msg += f" (错误类型: {e.__class__.__name__})"
            self.logger.error(error_msg)
    
    def load_components(self) -> pd.DataFrame:
        """
        加载沪深300成分股列表
        
        Returns:
            pd.DataFrame: 成分股数据框
        """
        try:
            # 确保股票代码列被读取为字符串类型
            self.components_df = pd.read_csv(config.COMPONENTS_FILE, dtype={'股票代码': str})
            
            # 过滤掉已处理的股票
            original_count = len(self.components_df)
            
            # 创建过滤条件
            mask = self.components_df.apply(
                lambda row: (str(row['股票代码']), row['股票名称']) not in self.processed_stocks, 
                axis=1
            )
            
            self.components_df = self.components_df[mask]
            
            self.logger.info(f"成功加载 {len(self.components_df)} 只成分股 (跳过 {original_count - len(self.components_df)} 只已处理股票)")
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
        
        # 生成文件名（使用固定文件名，支持增量保存）
        filename = f"hs300_daily_prices_batch_{batch_num:03d}.csv"
        filepath = os.path.join(config.OUTPUT_DIR, filename)
        
        # 检查文件是否已存在，如果存在则追加数据
        if os.path.exists(filepath):
            try:
                # 读取现有数据
                existing_data = pd.read_csv(filepath)
                # 合并新数据
                combined_data = pd.concat([existing_data, combined_data], ignore_index=True)
                # 去重（基于股票代码、日期和股票名称）
                combined_data = combined_data.drop_duplicates(
                    subset=['stock_code', 'date', 'stock_name'], 
                    keep='last'
                )
                self.logger.info(f"追加数据到现有文件: {filepath}")
            except Exception as e:
                self.logger.warning(f"读取现有文件失败，将覆盖保存: {e}")
        
        # 保存数据
        combined_data.to_csv(filepath, index=False, encoding='utf-8-sig')
        
        self.logger.info(f"批次 {batch_num} 数据已保存: {filepath} (包含 {len(combined_data)} 条记录)")
        
        # 记录已处理的股票
        batch_stocks = combined_data[['stock_code', 'stock_name']].drop_duplicates()
        for _, stock in batch_stocks.iterrows():
            self.processed_stocks.add((str(stock['stock_code']), stock['stock_name']))
    
    def save_failed_stocks(self):
        """
        保存失败股票信息到文件
        """
        if not self.failed_stocks:
            return
            
        # 创建失败记录目录
        failed_dir = os.path.join(config.BASE_DATA_DIR, 'failed_records')
        os.makedirs(failed_dir, exist_ok=True)
        
        # 生成文件名
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"failed_stocks_{timestamp}.csv"
        filepath = os.path.join(failed_dir, filename)
        
        # 保存失败股票信息
        failed_df = pd.DataFrame(self.failed_stocks)
        failed_df.to_csv(filepath, index=False, encoding='utf-8-sig')
        
        self.logger.info(f"失败股票信息已保存: {filepath}")
    
    def process_batch(self, stock_batch: List[Dict], batch_num: int, total_batches: int):
        """
        处理一个批次的股票数据
        
        Args:
            stock_batch (List[Dict]): 批次股票列表
            batch_num (int): 批次编号
            total_batches (int): 总批次数量
        """
        self.logger.info(f"开始处理批次 {batch_num}/{total_batches}，包含 {len(stock_batch)} 只股票")
        
        # 重置当前批次数据
        self.current_batch_data = []
        
        # 创建批次进度条
        batch_desc = f"批次 {batch_num}/{total_batches}"
        with tqdm(stock_batch, desc=batch_desc, unit="股票", leave=False) as batch_pbar:
            for stock in batch_pbar:
                stock_code = stock['股票代码']
                stock_name = stock['股票名称']
                
                # 检查是否发生错误，如果发生错误则立即停止
                if self.error_occurred:
                    self.logger.warning(f"检测到错误，停止处理批次 {batch_num}")
                    break
                
                # 更新进度条描述
                batch_pbar.set_postfix({
                    "股票": f"{stock_code} {stock_name}",
                    "成功": len(self.current_batch_data)
                })
                
                # 检查是否已处理
                if (stock_code, stock_name) in self.processed_stocks:
                    batch_pbar.set_description(f"{batch_desc} - 跳过已处理")
                    continue
                
                # 获取股票数据
                stock_data = self.get_stock_data(stock_code, stock_name, config.START_DATE, config.END_DATE)
                
                if stock_data is not None:
                    self.current_batch_data.append(stock_data)
                    batch_pbar.set_description(f"{batch_desc} - 成功获取")
                    
                    # 立即保存成功获取的数据，避免数据丢失
                    self.save_batch_data(self.current_batch_data, batch_num)
                else:
                    batch_pbar.set_description(f"{batch_desc} - 获取失败")
                    
                    # 如果获取失败，标记错误并停止处理
                    self.error_occurred = True
                    self.logger.error(f"股票 {stock_code} {stock_name} 获取失败，停止处理批次 {batch_num}")
                    break
                
                # 请求间隔，避免频率限制
                time.sleep(config.REQUEST_DELAY)
        
        # 如果批次正常完成，保存最终数据
        if not self.error_occurred:
            self.save_batch_data(self.current_batch_data, batch_num)
            self.logger.info(f"批次 {batch_num}/{total_batches} 处理完成，成功获取 {len(self.current_batch_data)} 只股票数据")
        else:
            self.logger.warning(f"批次 {batch_num}/{total_batches} 因错误提前结束，已保存 {len(self.current_batch_data)} 只股票数据")
    
    def run(self):
        """运行数据采集程序"""
        self.logger.info("开始沪深300成分股数据采集")
        self.logger.info(f"时间范围: {config.START_DATE} 到 {config.END_DATE}")
        
        try:
            # 加载已处理股票信息（断点续传功能）
            self.load_processed_stocks()
            
            # 加载成分股
            components_df = self.load_components()
            
            # 检查是否还有需要处理的股票
            if len(components_df) == 0:
                self.logger.info("所有股票数据已处理完成，无需再次采集")
                self.print_statistics()
                return
            
            # 分批处理
            total_stocks = len(components_df)
            num_batches = (total_stocks + config.BATCH_SIZE - 1) // config.BATCH_SIZE
            
            self.logger.info(f"总共 {total_stocks} 只股票待处理，分为 {num_batches} 个批次处理")
            
            # 创建总体进度条
            with tqdm(total=num_batches, desc="总体进度", unit="批次", position=0) as main_pbar:
                for batch_num in range(num_batches):
                    # 检查是否发生错误，如果发生错误则立即停止
                    if self.error_occurred:
                        self.logger.error("检测到错误，停止数据采集程序")
                        break
                    
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
                    if batch_num < num_batches - 1 and not self.error_occurred:
                        main_pbar.set_description(f"总体进度 - 等待下一批次")
                        time.sleep(config.REQUEST_DELAY)
            
            # 输出统计信息
            self.print_statistics()
            
            # 如果有错误发生，提示用户
            if self.error_occurred:
                self.logger.warning("数据采集因错误提前结束，请检查失败股票并重新运行程序")
            
        except Exception as e:
            self.logger.error(f"数据采集过程中发生错误: {e}")
            self.logger.error(traceback.format_exc())
    
    def print_statistics(self):
        """打印采集统计信息"""
        self.logger.info("=" * 50)
        self.logger.info("数据采集统计信息:")
        self.logger.info(f"成功处理股票数量: {len(self.processed_stocks)}")
        self.logger.info(f"失败股票数量: {len(self.failed_stocks)}")
        
        # 保存失败股票信息
        self.save_failed_stocks()
        
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