# 特征工程模块

股票数据分析核心流程的特征工程实现。

## 功能概述

本模块实现股票数据分析的核心技术指标计算流程，包括：

1. **数据读取与排序**：从CSV文件读取日线交易数据，按交易日升序排列
2. **技术指标计算**：
   - 过去N个交易日的收益率（支持1,5,10,20日周期）
   - 滚动窗口均值（可配置窗口大小，默认20日）
   - 滚动窗口标准差（可配置窗口大小，默认20日）
   - 动量指标（可指定计算周期，默认10日）
   - RSI相对强弱指数（默认14日周期）
   - 换手率的移动平均值（默认5日窗口）
3. **序列化处理**：确保数据处理过程的可追溯性和完整性
4. **结构化输出**：生成包含原始数据和技术因子的完整数据集

## 模块结构

```
feature_engineer/
├── __init__.py          # 模块初始化
├── technical_indicators.py  # 技术指标计算核心类
├── main.py              # 特征工程主程序
└── README.md           # 说明文档
```

## 使用方法

### 1. 快速使用

```python
from feature_engineer import TechnicalIndicators

# 创建计算器（使用默认配置）
calculator = TechnicalIndicators()

# 读取数据
df = calculator.load_data("data_pipeline/data/daily_prices/Merge/hs300_daily_prices_merged.csv")

# 计算所有技术指标
df_with_indicators = calculator.process_all_indicators(df)

# 保存结果
df_with_indicators.to_csv("output.csv", index=False)
```

### 2. 自定义配置

```python
config = {
    'return_periods': [1, 5, 10, 20],  # 收益率计算周期
    'rolling_window': 20,               # 滚动窗口大小
    'momentum_period': 10,              # 动量指标周期
    'rsi_period': 14,                   # RSI周期
    'turnover_window': 5                # 换手率移动平均窗口
}

calculator = TechnicalIndicators(config)
```

### 3. 运行完整管道

```bash
python data_pipeline/feature_engineer/main.py
```

## 技术指标说明

### 收益率 (Return)
- 计算过去N个交易日的价格收益率
- 公式: `return_Nd = (close_t / close_{t-N}) - 1`

### 滚动统计 (Rolling Statistics)
- **均值**: N日收盘价的移动平均
- **标准差**: N日收盘价的标准差，衡量价格波动性

### 动量指标 (Momentum)
- 衡量价格变动趋势的强度
- 公式: `momentum_Nd = (close_t / close_{t-N}) - 1`

### RSI相对强弱指数
- 衡量股票超买超卖状态
- 范围: 0-100，通常30以下为超卖，70以上为超买
- 公式: `RSI = 100 - (100 / (1 + RS))`，其中RS为平均收益/平均损失

### 换手率移动平均
- 换手率的N日移动平均，反映交易活跃度的趋势

## 数据要求

输入数据应包含以下列：
- `date`: 交易日期
- `股票代码`: 股票标识
- `open`: 开盘价
- `close`: 收盘价  
- `high`: 最高价
- `low`: 最低价
- `volume`: 成交量
- `turnover`: 换手率（可选，用于换手率移动平均计算）

## 输出数据

输出数据包含原始数据列和以下技术指标列：
- `return_1d`, `return_5d`, `return_10d`, `return_20d`
- `rolling_mean_20d`, `rolling_std_20d`
- `momentum_10d`
- `rsi_14d`
- `turnover_ma_5d`（如果输入数据包含turnover列）

## 序列化处理

模块自动记录处理过程，包括：
- 处理开始和结束时间
- 使用的配置参数
- 每个计算步骤的详细信息
- 最终结果统计

处理日志保存为JSON格式，便于后续分析和审计。

## 错误处理

模块包含完整的错误处理机制：
- 数据文件不存在或格式错误
- 必要列缺失检查
- 计算过程中的异常捕获
- 详细的日志记录