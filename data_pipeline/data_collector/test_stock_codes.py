"""
测试股票代码格式转换和数据采集功能
"""

import config
import pandas as pd

# 测试股票代码格式转换
def test_stock_symbol_conversion():
    """测试股票代码格式转换"""
    print("=== 股票代码格式转换测试 ===")
    
    # 测试用例：包含各种格式的股票代码
    test_cases = [
        ('2', '000002'),      # 深圳主板，1位代码
        ('858', '000858'),    # 深圳主板，3位代码
        ('651', '000651'),    # 深圳主板，3位代码
        ('600690', '600690'), # 上海主板，6位代码
        ('300498', '300498'), # 创业板，6位代码
        ('688012', '688012'), # 科创板，6位代码
        ('1', '000001'),      # 深圳主板，1位代码
        ('63', '000063'),     # 深圳主板，2位代码
        ('100', '000100'),    # 深圳主板，3位代码
        ('601319', '601319'), # 上海主板，6位代码
    ]
    
    all_passed = True
    for input_code, expected_output in test_cases:
        actual_output = config.get_stock_symbol(input_code)
        status = "✓" if actual_output == expected_output else "✗"
        print(f"{status} {input_code} -> {actual_output} (期望: {expected_output})")
        
        if actual_output != expected_output:
            all_passed = False
    
    print(f"\n测试结果: {'全部通过' if all_passed else '存在错误'}")
    return all_passed

def test_component_file_loading():
    """测试成分股文件加载和股票代码格式"""
    print("\n=== 成分股文件加载测试 ===")
    
    try:
        # 加载成分股文件
        components_df = pd.read_csv(config.COMPONENTS_FILE, dtype={'股票代码': str})
        print(f"成功加载 {len(components_df)} 只成分股")
        
        # 检查股票代码格式
        stock_codes = components_df['股票代码'].tolist()
        
        # 统计不同长度的股票代码
        code_lengths = {}
        for code in stock_codes:
            length = len(str(code))
            code_lengths[length] = code_lengths.get(length, 0) + 1
        
        print("股票代码长度分布:")
        for length, count in sorted(code_lengths.items()):
            print(f"  {length}位代码: {count}只")
        
        # 测试前10只股票的格式转换
        print("\n前10只股票的AKshare符号:")
        for i, row in components_df.head(10).iterrows():
            stock_code = row['股票代码']
            stock_name = row['股票名称']
            symbol = config.get_stock_symbol(stock_code)
            print(f"  {stock_code} {stock_name} -> {symbol}")
        
        return True
        
    except Exception as e:
        print(f"加载成分股文件失败: {e}")
        return False

def main():
    """主测试函数"""
    print("开始测试股票代码格式转换和数据采集功能\n")
    
    # 测试股票代码转换
    conversion_passed = test_stock_symbol_conversion()
    
    # 测试成分股文件加载
    loading_passed = test_component_file_loading()
    
    # 总结
    print("\n=== 测试总结 ===")
    if conversion_passed and loading_passed:
        print("✓ 所有测试通过！股票代码格式转换功能正常。")
        print("✓ 成分股文件加载功能正常。")
        print("\n建议：现在可以运行数据采集器来获取股票历史数据。")
    else:
        print("✗ 存在测试失败，请检查配置和文件格式。")

if __name__ == "__main__":
    main()