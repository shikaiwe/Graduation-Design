#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
股票日线数据合并程序

功能：将分批存储的沪深300成分股日线数据合并成一个完整的CSV文件
作者：毕业设计项目
日期：2024年
"""

import os
import pandas as pd
import glob
from pathlib import Path
import logging
from datetime import datetime

# 配置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def merge_daily_prices(input_dir, output_file):
    """
    合并分批的股票日线数据
    
    参数:
        input_dir (str): 输入目录路径，包含分批的CSV文件
        output_file (str): 输出文件路径，合并后的完整数据文件
    
    返回:
        bool: 合并操作是否成功
    """
    try:
        # 检查输入目录是否存在
        if not os.path.exists(input_dir):
            logger.error(f"输入目录不存在: {input_dir}")
            return False
        
        # 查找所有分批数据文件
        pattern = os.path.join(input_dir, "hs300_daily_prices_batch_*.csv")
        batch_files = sorted(glob.glob(pattern))
        
        if not batch_files:
            logger.error(f"在目录 {input_dir} 中未找到任何分批数据文件")
            return False
        
        logger.info(f"找到 {len(batch_files)} 个分批数据文件")
        
        # 读取并合并所有文件
        data_frames = []
        total_rows = 0
        
        for i, file_path in enumerate(batch_files, 1):
            logger.info(f"正在读取第 {i} 个文件: {os.path.basename(file_path)}")
            
            try:
                # 读取CSV文件
                df = pd.read_csv(file_path)
                
                # 检查数据质量
                if df.empty:
                    logger.warning(f"文件 {file_path} 为空，跳过")
                    continue
                
                # 检查必要的列是否存在
                required_columns = ['date', '股票代码', 'open', 'close', 'high', 'low', 'volume']
                missing_columns = [col for col in required_columns if col not in df.columns]
                if missing_columns:
                    logger.warning(f"文件 {file_path} 缺少列: {missing_columns}")
                    continue
                
                # 添加批次信息（可选，用于追踪数据来源）
                df['batch_file'] = os.path.basename(file_path)
                
                data_frames.append(df)
                total_rows += len(df)
                
                logger.info(f"文件 {file_path} 读取成功，包含 {len(df)} 行数据")
                
            except Exception as e:
                logger.error(f"读取文件 {file_path} 时出错: {str(e)}")
                continue
        
        if not data_frames:
            logger.error("没有成功读取任何数据文件")
            return False
        
        # 合并所有数据框
        logger.info("开始合并数据...")
        merged_df = pd.concat(data_frames, ignore_index=True)
        
        # 数据清洗和去重
        logger.info("进行数据清洗...")
        
        # 按日期和股票代码去重（保留最后出现的记录）
        merged_df = merged_df.drop_duplicates(subset=['date', '股票代码'], keep='last')
        
        # 按日期和股票代码排序
        merged_df = merged_df.sort_values(['date', '股票代码'])
        
        # 重置索引
        merged_df = merged_df.reset_index(drop=True)
        
        # 检查合并后的数据质量
        logger.info(f"合并后数据统计:")
        logger.info(f"  总行数: {len(merged_df)}")
        logger.info(f"  唯一日期数: {merged_df['date'].nunique()}")
        logger.info(f"  唯一股票数: {merged_df['股票代码'].nunique()}")
        logger.info(f"  数据时间范围: {merged_df['date'].min()} 到 {merged_df['date'].max()}")
        
        # 确保输出目录存在
        output_dir = os.path.dirname(output_file)
        if output_dir and not os.path.exists(output_dir):
            os.makedirs(output_dir)
            logger.info(f"创建输出目录: {output_dir}")
        
        # 保存合并后的数据
        logger.info(f"正在保存合并数据到: {output_file}")
        merged_df.to_csv(output_file, index=False, encoding='utf-8-sig')
        
        logger.info("数据合并完成！")
        return True
        
    except Exception as e:
        logger.error(f"合并数据时发生错误: {str(e)}")
        return False

def main():
    """主函数"""
    # 配置路径
    input_directory = r"e:\Design\Graduation-Design\data_pipeline\data\daily_prices"
    # 创建Merge文件夹路径
    merge_directory = os.path.join(input_directory, "Merge")
    output_filename = os.path.join(merge_directory, "hs300_daily_prices_merged.csv")
    
    logger.info("开始合并沪深300成分股日线数据")
    logger.info(f"输入目录: {input_directory}")
    logger.info(f"输出目录: {merge_directory}")
    logger.info(f"输出文件: {output_filename}")
    
    # 执行合并
    success = merge_daily_prices(input_directory, output_filename)
    
    if success:
        logger.info("数据合并任务完成！")
    else:
        logger.error("数据合并任务失败！")
        return 1
    
    return 0

if __name__ == "__main__":
    exit(main())