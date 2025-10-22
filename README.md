# 毕业设计项目：基于机器学习的量化选股数据管道

本仓库当前聚焦于“数据阶段”的生产化落地：沪深300成分股的历史数据采集与特征工程流水线，为后续模型训练、策略回测与部署打下数据基础。

注意：顶层目标是“基于机器学习的量化选股系统”，但目前仓库实现主要覆盖数据采集与特征工程模块；训练/回测/部署将在后续迭代中补全。

## 功能特性
- HS300 成分股日频数据采集（批次/限频/重试/恢复/详细日志）
- 特征工程主流程（技术指标、基本面、宏观、EPU 因子、因子选择）
- 标准化目录结构与产出文件命名规范
- 面向生产的可配置与可追溯处理日志

## 系统架构
```mermaid
graph TD
A[数据采集 data_collector] --> B[特征工程 feature_engineer]
B --> C[特征输出 features/*.csv|json]
C -.-> D[模型训练(待实现)]
D -.-> E[策略回测(待实现)]
E -.-> F[实盘部署(待实现)]
```

## 目录结构
```
Graduation-Design/
├─ data_pipeline/
│  ├─ data_collector/               # HS300 日频数据采集模块
│  │  ├─ config.py                  # 采集配置项（起止日期、批次、限频等）
│  │  ├─ data_collector.py          # 采集主程序
│  │  ├─ cli.py                     # 命令行入口
│  │  ├─ utils.py                   # 工具函数
│  │  └─ README.md                  # 模块说明
│  └─ feature_engineer/             # 特征工程模块
│     ├─ technical_indicators.py    # 技术指标计算
│     ├─ fundamental_factors.py     # 基本面因子
│     ├─ macro_factors.py           # 宏观因子
│     ├─ epu_factors.py             # EPU 因子
│     ├─ factor_selection.py        # 因子选择
│     ├─ main.py                    # 特征工程主流程入口
│     └─ README.md                  # 模块说明
├─ data/
│  ├─ components/                   # 成分股列表
│  ├─ daily_prices/                 # 原始日线数据（CSV 分批/合并）
│  ├─ features/                     # 各类因子与选择结果
│  ├─ EPU/                          # EPU 原始文件
│  └─ logs/                         # 运行日志
├─ requirements.txt                 # 依赖列表（数据阶段）
├─ LICENSE
└─ README.md                        # 本文件
```

## 环境与依赖
- 语言：Python 3.9+（推荐）
- 操作系统：Windows（PowerShell 示例）
- 依赖（数据阶段）：`akshare`、`tushare`、`baostock`、`pandas`、`sqlalchemy`、`aiosqlite`、`requests`、`loguru`、`tqdm`

安装依赖：
```powershell
# 在项目根目录执行
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

## 快速开始
1) 采集 HS300 日频数据（默认配置）
```powershell
cd data_pipeline\data_collector
python cli.py
```

2) 运行特征工程主流程
```powershell
cd ..\feature_engineer
python main.py
```

运行完成后，产物将写入 `data/` 子目录（见下文“输出产物”）。

## 数据采集使用说明（data_collector）
- 入口：`data_pipeline/data_collector/cli.py`
- 关键配置：`data_pipeline/data_collector/config.py`

常用参数示例：
```powershell
# 指定时间范围与批次大小
python cli.py --start-date 2020-01-01 --end-date 2023-12-31 --batch-size 10

# 恢复模式（跳过已处理股票）
python cli.py --resume

# 试运行（不实际拉取数据）
python cli.py --dry-run

# 提高日志详细度
python cli.py --verbose
```
参数说明（精选）：
- `--start-date/-s` 开始日期，默认取自 `config.py`
- `--end-date/-e` 结束日期，默认取自 `config.py`
- `--batch-size/-b` 每批处理股票数量，默认取自 `config.py`
- `--delay/-d` 请求间隔秒数（限频保护，默认 2.0）
- `--max-retries/-r` 最大重试次数（默认 3）
- `--resume` 恢复模式
- `--dry-run` 试运行
- `--verbose/-v` 详细日志

输出位置与命名：
- 原始分批数据：`data/daily_prices/`，如 `hs300_daily_prices_batch_001_YYYYMMDD_HHMMSS.csv`
- 合并文件（如有合并流程）：`data/daily_prices/Merge/hs300_daily_prices_merged.csv`

日志：`data/logs/data_collector.log`

更多细节请参见模块文档：`data_pipeline/data_collector/README.md`。

## 特征工程主流程（feature_engineer）
- 入口：`data_pipeline/feature_engineer/main.py`
- 功能步骤：
  1. 技术指标计算（收益率、滚动统计、动量、RSI、成交活跃度、MACD 等）
  2. 基本面因子计算（PE、PB、ROE、ROA、营收/利润增速等）
  3. 宏观因子计算（GDP、CPI、利率、M2 增速等，含日历映射）
  4. EPU 因子构建（滞后、增长等衍生）
  5. 因子选择（相关性阈值、PCA、特征重要性等可组合）

运行：
```powershell
cd data_pipeline\feature_engineer
python main.py
```

关键配置位置（示例，位于 `main.py` 顶部区域）：
- 技术指标参数：`tech_config`（周期、窗口、MACD 参数等）
- 基本面因子：`fundamental_config`（启用因子清单）
- 宏观因子：`macro_config`（启用因子清单）
- EPU 配置：`epu_config`（EPU 源文件路径、滞后与增长周期）
- 选择策略：`selection_config`（相关性阈值、PCA 方差阈、Top-K 与方法集）

输出产物：
- `data/features/technical_indicators.csv`
- `data/features/fundamental_factors.csv`
- `data/features/macro_factors.csv`
- `data/features/macro_factors_daily.csv`
- `data/features/epu_factors.csv`
- `data/features/epu_factors_daily.csv`
- `data/features/factor_selection_results.json`

日志：
- `data/logs/feature_engineering.log`
- 各子模块处理日志 JSON：如 `technical_indicators_processing_log.json`、`macro_factors_processing_log.json` 等

更多细节请参见模块文档：`data_pipeline/feature_engineer/README.md`。

## 常见问题与排查
1) API 限频或请求失败
- 现象：报错提示限频或网络错误
- 处理：提高 `--delay`（如 3-5 秒）、保持 `--max-retries`≥3、确保网络稳定

2) 数据为空或缺列
- 现象：特征工程报“必需列缺失”
- 处理：检查采集阶段是否成功；确认合并文件路径 `data/daily_prices/Merge/hs300_daily_prices_merged.csv` 是否存在并包含列：`date`、`股票代码`、`open`、`close`、`high`、`low`、`volume`

3) 文件路径/权限问题（Windows）
- 现象：无法写入 `data/*` 子目录
- 处理：以有写权限的用户运行；确保先创建 `data` 及子目录或让程序自动创建；避免路径中包含只读位置

4) PowerShell 执行策略
- 现象：虚拟环境激活脚本被拦截
- 处理：在有权限前提下执行 `Set-ExecutionPolicy -Scope CurrentUser RemoteSigned`

## 注意事项
- 合理设置采集批次与请求间隔，避免触发数据源限频
- 保持依赖版本与 `requirements.txt` 一致，减少兼容性问题
- 运行前确认 EPU 源文件路径是否正确（默认示例：`data/EPU/China_Mainland_Paper_EPU.xlsx`）
- 特征工程前建议先确认合并后的日线数据已就绪

## 后续计划（占位）
- 模型训练：XGBoost / LightGBM 融合与 LSTM 时序扩展（依赖与代码将后续补充）
- 策略回测：多维评价指标与参数敏感性分析
- 部署：实盘/仿真对接与稳定性保障

以上为规划目标，当前仓库尚未包含训练、回测与部署代码与依赖。

## 许可证
本项目采用开源许可，详见 `LICENSE`。

## 参考与致谢
- 数据源与生态：AkShare、TuShare、BaoStock 等
- 项目内参考文献条目请见先前文档说明（论文/参考文献目录）

---
如需问题定位，请优先查看 `data/logs/*.log` 与各 JSON 处理日志；若需扩展或定制，请在对应模块的配置段落中调整参数并重新运行。