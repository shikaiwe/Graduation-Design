#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
检查HS300成分股数据完整性的脚本

功能：
1. 整合所有每日价格数据文件
2. 对比HS300成分股列表，检查是否有缺失的股票
3. 生成缺失股票报告
"""

import os
import pandas as pd
import glob
from pathlib import Path

def load_hs300_components(components_file_path):
    """
    加载HS300成分股列表
    
    Args:
        components_file_path (str): HS300成分股文件路径
        
    Returns:
        pd.DataFrame: 包含股票代码和股票名称的DataFrame
    """
    print("正在加载HS300成分股列表...")
    
    # 读取成分股文件
    components_df = pd.read_csv(components_file_path)
    print(f"原始成分股记录数: {len(components_df)} 条")
    
    # 处理股票代码格式
    components_df['股票代码_str'] = components_df['股票代码'].astype(str).str.strip()
    
    # 显示重复的股票代码（去重前）
    duplicate_stocks = components_df[components_df.duplicated(subset=['股票代码_str'], keep=False)]
    if len(duplicate_stocks) > 0:
        duplicate_codes = duplicate_stocks['股票代码_str'].unique().tolist()
        print(f"发现重复股票代码: {duplicate_codes}")
        print(f"重复股票数量: {len(duplicate_codes)} 只")
    
    # 去重处理，保留每个股票代码的第一条记录
    components_df = components_df.drop_duplicates(subset=['股票代码_str'], keep='first')
    
    print(f"去重后成功加载 {len(components_df)} 只HS300成分股")
    return components_df

def get_all_daily_price_files(daily_prices_dir):
    """
    获取所有每日价格数据文件路径
    
    Args:
        daily_prices_dir (str): 每日价格数据目录路径
        
    Returns:
        list: 所有CSV文件路径列表
    """
    pattern = os.path.join(daily_prices_dir, "hs300_daily_prices_batch_*.csv")
    files = glob.glob(pattern)
    files.sort()  # 按文件名排序
    
    print(f"找到 {len(files)} 个每日价格数据文件")
    return files

def extract_unique_stocks_from_files(file_paths):
    """
    从所有价格数据文件中提取唯一的股票代码
    
    Args:
        file_paths (list): CSV文件路径列表
        
    Returns:
        set: 所有文件中出现的唯一股票代码集合
    """
    unique_stocks = set()
    
    print("正在从价格数据文件中提取股票代码...")
    
    for i, file_path in enumerate(file_paths, 1):
        print(f"处理文件 {i}/{len(file_paths)}: {os.path.basename(file_path)}")
        
        try:
            # 读取文件，只读取股票代码列
            df = pd.read_csv(file_path, usecols=['股票代码'])
            
            # 确保股票代码格式一致（转换为字符串并去除可能的空格）
            df['股票代码'] = df['股票代码'].astype(str).str.strip()
            
            # 提取唯一的股票代码
            file_stocks = set(df['股票代码'].unique())
            unique_stocks.update(file_stocks)
            
            print(f"  文件包含 {len(file_stocks)} 只股票，累计发现 {len(unique_stocks)} 只唯一股票")
            
            # 显示前几个股票代码作为示例
            if i == 1:
                sample_stocks = list(file_stocks)[:5]
                print(f"  示例股票代码: {sample_stocks}")
            
        except Exception as e:
            print(f"  处理文件 {file_path} 时出错: {e}")
    
    return unique_stocks

def check_missing_stocks(components_df, price_stocks_set):
    """
    检查缺失的股票
    
    Args:
        components_df (pd.DataFrame): HS300成分股DataFrame
        price_stocks_set (set): 价格数据中出现的股票代码集合
        
    Returns:
        pd.DataFrame: 缺失的股票信息DataFrame
    """
    print("\n正在检查缺失的股票...")
    
    # 确保成分股股票代码格式一致
    components_df['股票代码_str'] = components_df['股票代码'].astype(str).str.strip()
    
    # 获取成分股中的所有股票代码
    component_stocks = set(components_df['股票代码_str'])
    
    # 显示一些示例股票代码用于调试
    print(f"成分股示例股票代码: {list(component_stocks)[:5]}")
    print(f"价格数据示例股票代码: {list(price_stocks_set)[:5]}")
    
    # 找出在成分股中但不在价格数据中的股票
    missing_stocks = component_stocks - price_stocks_set
    
    # 创建缺失股票的报告
    missing_info = []
    for stock_code in missing_stocks:
        # 查找该股票在成分股中的信息
        stock_info = components_df[components_df['股票代码_str'] == stock_code].iloc[0]
        missing_info.append({
            '股票代码': stock_code,
            '股票名称': stock_info['股票名称'],
            '纳入日期': stock_info['纳入日期']
        })
    
    missing_df = pd.DataFrame(missing_info)
    
    print(f"HS300成分股总数: {len(component_stocks)}")
    print(f"价格数据中出现的股票数: {len(price_stocks_set)}")
    print(f"缺失的股票数: {len(missing_df)}")
    
    return missing_df

def generate_report(components_df, price_stocks_set, missing_df, output_dir):
    """
    生成详细的检查报告
    
    Args:
        components_df (pd.DataFrame): HS300成分股DataFrame
        price_stocks_set (set): 价格数据中出现的股票代码集合
        missing_df (pd.DataFrame): 缺失的股票信息DataFrame
        output_dir (str): 输出目录路径
    """
    print("\n正在生成检查报告...")
    
    # 确保输出目录存在
    os.makedirs(output_dir, exist_ok=True)
    
    # 获取唯一的成分股数量（使用处理后的股票代码）
    unique_component_stocks = len(set(components_df['股票代码_str']))
    
    # 生成汇总报告
    report_content = f"""
HS300成分股数据完整性检查报告
================================

检查时间: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}

统计信息:
- HS300成分股原始记录数: {len(components_df) + len(components_df[components_df.duplicated(subset=['股票代码_str'], keep=False)])}
- HS300成分股去重后总数: {unique_component_stocks}
- 价格数据中出现的股票数: {len(price_stocks_set)}
- 缺失的股票数: {len(missing_df)}
- 数据完整率: {(len(price_stocks_set) / unique_component_stocks * 100):.2f}%

缺失股票列表:
"""
    
    if len(missing_df) > 0:
        for _, row in missing_df.iterrows():
            report_content += f"- {row['股票代码']} ({row['股票名称']}), 纳入日期: {row['纳入日期']}\n"
    else:
        report_content += "无缺失股票，数据完整！\n"
    
    # 保存报告
    report_file = os.path.join(output_dir, "hs300_data_completeness_report.txt")
    with open(report_file, 'w', encoding='utf-8') as f:
        f.write(report_content)
    
    # 保存缺失股票的CSV文件
    if len(missing_df) > 0:
        missing_csv_file = os.path.join(output_dir, "missing_hs300_stocks.csv")
        missing_df.to_csv(missing_csv_file, index=False, encoding='utf-8-sig')
    
    print(f"报告已保存到: {report_file}")
    if len(missing_df) > 0:
        print(f"缺失股票列表已保存到: {missing_csv_file}")

def main():
    """主函数"""
    # 文件路径配置
    components_file_path = r"e:\Design\Graduation-Design\data_pipeline\data\components\hs300_components_cutoff_2024.csv"
    daily_prices_dir = r"e:\Design\Graduation-Design\data_pipeline\data\daily_prices"
    output_dir = r"e:\Design\Graduation-Design\data_pipeline\data\reports"
    
    print("开始检查HS300成分股数据完整性...")
    print("=" * 60)
    
    try:
        # 1. 加载HS300成分股列表
        components_df = load_hs300_components(components_file_path)
        
        # 2. 获取所有每日价格数据文件
        price_files = get_all_daily_price_files(daily_prices_dir)
        
        if not price_files:
            print("错误：未找到任何每日价格数据文件")
            return
        
        # 3. 从价格数据文件中提取唯一的股票代码
        price_stocks_set = extract_unique_stocks_from_files(price_files)
        
        # 4. 检查缺失的股票
        missing_df = check_missing_stocks(components_df, price_stocks_set)
        
        # 5. 生成报告
        generate_report(components_df, price_stocks_set, missing_df, output_dir)
        
        print("\n" + "=" * 60)
        print("检查完成！")
        
        # 显示简要结果
        if len(missing_df) == 0:
            print("✅ 所有HS300成分股数据完整，无缺失股票")
        else:
            print(f"⚠️  发现 {len(missing_df)} 只缺失股票，请查看详细报告")
            print("缺失股票:")
            for _, row in missing_df.iterrows():
                print(f"  - {row['股票代码']} ({row['股票名称']})")
        
    except Exception as e:
        print(f"检查过程中出现错误: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()