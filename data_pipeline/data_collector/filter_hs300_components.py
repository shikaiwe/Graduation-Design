#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
沪深300成分股数据筛选程序
"""

import pandas as pd
import os
from datetime import datetime


def filter_hs300_components_by_date(input_file, output_file, cutoff_year=2025):
    """
    筛选沪深300成分股数据，保留截止到指定年份之前的股票数据
    
    参数:
        input_file (str): 输入CSV文件路径
        output_file (str): 输出CSV文件路径
        cutoff_year (int): 截止年份，默认为2025
    
    返回:
        tuple: (筛选后的数据行数, 总数据行数)
    """
    
    # 检查输入文件是否存在
    if not os.path.exists(input_file):
        raise FileNotFoundError(f"输入文件不存在: {input_file}")
    
    # 读取CSV文件
    print(f"正在读取文件: {input_file}")
    df = pd.read_csv(input_file, encoding='utf-8')
    
    # 检查必要的列是否存在
    required_columns = ['股票代码', '股票名称', '纳入日期', '指数代码', '数据来源', '获取时间']
    missing_columns = [col for col in required_columns if col not in df.columns]
    if missing_columns:
        raise ValueError(f"CSV文件缺少必要的列: {missing_columns}")
    
    # 转换日期列为datetime格式
    df['纳入日期'] = pd.to_datetime(df['纳入日期'], errors='coerce')
    
    # 筛选截止到2024年之前的数据
    cutoff_date = datetime(cutoff_year, 1, 1)
    filtered_df = df[df['纳入日期'] < cutoff_date]
    
    # 统计信息
    total_rows = len(df)
    filtered_rows = len(filtered_df)
    
    print(f"原始数据行数: {total_rows}")
    print(f"筛选后数据行数: {filtered_rows}")
    print(f"筛选条件: 纳入日期 < {cutoff_year}-01-01")
    
    # 保存筛选结果
    if filtered_rows > 0:
        # 确保输出目录存在
        os.makedirs(os.path.dirname(output_file), exist_ok=True)
        
        # 保存筛选结果
        filtered_df.to_csv(output_file, index=False, encoding='utf-8')
        print(f"筛选结果已保存至: {output_file}")
        
        # 显示前几行筛选结果
        print("\n筛选结果预览:")
        print(filtered_df.head())
    else:
        print("警告: 没有找到符合筛选条件的数据")
        # 创建空的输出文件
        filtered_df.to_csv(output_file, index=False, encoding='utf-8')
        print(f"已创建空的输出文件: {output_file}")
    
    return filtered_rows, total_rows


def main():
    """主函数"""
    
    # 文件路径配置
    input_file = r"e:\Design\Graduation-Design\data_pipeline\data\components\hs300_components_full.csv"
    output_file = r"e:\Design\Graduation-Design\data_pipeline\data\components\hs300_components_cutoff_2024.csv"
    
    try:
        # 执行筛选
        filtered_rows, total_rows = filter_hs300_components_by_date(
            input_file, output_file, cutoff_year=2025
        )
        
        # 输出统计信息
        print(f"\n=== 筛选完成 ===")
        print(f"原始数据总量: {total_rows} 行")
        print(f"筛选后数据量: {filtered_rows} 行")
        print(f"筛选比例: {filtered_rows/total_rows*100:.2f}%")
        
    except Exception as e:
        print(f"错误: {e}")
        return 1
    
    return 0


if __name__ == "__main__":
    exit(main())