"""
测试AKshare接口调用功能
验证股票代码格式转换和日期格式转换的正确性
"""

import akshare as ak
import config
import pandas as pd
from datetime import datetime

def test_akshare_api_call():
    """测试AKshare接口调用"""
    print("=== AKshare接口调用测试 ===")
    
    # 测试用例：选择几只代表性的股票
    test_stocks = [
        ('600690', '青岛海尔'),  # 上海主板
        ('858', '五粮液'),      # 深圳主板（需要格式转换）
        ('300498', '温氏股份'), # 创业板
        ('688012', '中微公司'), # 科创板
    ]
    
    # 测试时间范围（使用包含交易日的时间段）
    start_date = "2024-01-02"  # 避开元旦假期
    end_date = "2024-01-12"
    start_date_ak = start_date.replace("-", "")
    end_date_ak = end_date.replace("-", "")
    
    print(f"测试时间范围: {start_date} 到 {end_date}")
    print(f"AKshare日期格式: {start_date_ak} 到 {end_date_ak}")
    print()
    
    for stock_code, stock_name in test_stocks:
        try:
            # 获取AKshare符号（6位数字代码）
            symbol = config.get_stock_symbol(stock_code)
            print(f"测试股票: {stock_code} {stock_name}")
            print(f"AKshare符号: {symbol}")
            
            # 调用AKshare接口
            stock_data = ak.stock_zh_a_hist(
                symbol=symbol, 
                period="daily", 
                start_date=start_date_ak, 
                end_date=end_date_ak,
                adjust="qfq"
            )
            
            if stock_data.empty:
                print(f"  ⚠️  无数据返回")
            else:
                print(f"  ✅ 成功获取 {len(stock_data)} 条数据")
                # 显示数据的基本信息
                print(f"    日期范围: {stock_data['日期'].min()} 到 {stock_data['日期'].max()}")
                print(f"    字段: {list(stock_data.columns)}")
                
                # 显示第一条数据
                first_row = stock_data.iloc[0]
                print(f"    示例数据: 日期={first_row['日期']}, 收盘价={first_row['收盘']}")
            
            print()
            
        except Exception as e:
            print(f"  ❌ 调用失败: {e}")
            print()

def test_date_format_conversion():
    """测试日期格式转换功能"""
    print("=== 日期格式转换测试 ===")
    
    test_dates = [
        "2024-01-01",
        "2023-12-31", 
        "2019-01-01",
        "2024-12-31"
    ]
    
    for date_str in test_dates:
        converted = date_str.replace("-", "")
        print(f"原始日期: {date_str} -> AKshare格式: {converted}")
    
    print()

def test_config_parameters():
    """测试配置参数"""
    print("=== 配置参数检查 ===")
    
    print(f"开始日期: {config.START_DATE}")
    print(f"结束日期: {config.END_DATE}")
    print(f"AKshare格式开始日期: {config.START_DATE.replace('-', '')}")
    print(f"AKshare格式结束日期: {config.END_DATE.replace('-', '')}")
    print(f"成分股文件: {config.COMPONENTS_FILE}")
    print(f"批次大小: {config.BATCH_SIZE}")
    print(f"请求延迟: {config.REQUEST_DELAY}秒")
    print(f"最大重试次数: {config.MAX_RETRIES}")
    print()

def main():
    """主测试函数"""
    print("开始测试AKshare接口调用功能\n")
    
    # 测试日期格式转换
    test_date_format_conversion()
    
    # 测试配置参数
    test_config_parameters()
    
    # 测试AKshare接口调用
    test_akshare_api_call()
    
    print("=== 测试完成 ===")
    print("如果所有测试都通过，说明AKshare接口调用功能正常。")
    print("现在可以运行数据采集器来获取完整的沪深300成分股历史数据。")

if __name__ == "__main__":
    main()