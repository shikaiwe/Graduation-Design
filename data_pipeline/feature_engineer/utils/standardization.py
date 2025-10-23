#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
标准化与数据质量工具
- 列名映射与统一（中英文与别名 -> 统一规范）
- 空值/零值系统化处理（自动识别、标记、合理填充）
- 处理过程变更日志记录
- 数据质量报告生成
"""
from __future__ import annotations

import json
from dataclasses import dataclass, asdict
from typing import Dict, Any, Tuple, List, Optional
import pandas as pd
import numpy as np
from datetime import datetime
import os


# 统一后的关键标识列
CANONICAL_DATE = "date"
CANONICAL_CODE = "stock_code"


def load_default_mapping() -> Dict[str, str]:
    """提供默认的列名映射。未知列保持不变。
    仅覆盖常见/已知中文列到英文蛇形命名。
    """
    mapping = {
        # 关键标识
        "日期": CANONICAL_DATE,
        "date": CANONICAL_DATE,
        "股票代码": CANONICAL_CODE,
        "代码": CANONICAL_CODE,
        "stock_code": CANONICAL_CODE,
        # 基本面常见列
        "净资产收益率": "roe",
        "总资产报酬率": "roa",
        "营业总收入同比增长率": "revenue_growth_yoy",
        "净利润同比增长率": "profit_growth_yoy",
        "市盈率": "pe",
        "市盈率_TTM": "pe_ttm",
        "市净率": "pb",
        # 宏观/EPU（大多已英文，这里留冗余）
        "国内生产总值": "gdp",
        "国内生产总值同比增长": "gdp_yoy",
        "国内生产总值环比增长": "gdp_qoq",
        "全國": "cpi",
        "全國同比增长": "cpi_yoy",
        # 技术指标（多数已英文，保留原名）
    }
    return mapping


@dataclass
class TransformationEvent:
    step: str
    timestamp: str
    details: Dict[str, Any]


@dataclass
class TransformationLog:
    start_time: str
    steps: List[TransformationEvent]

    def add(self, step: str, details: Dict[str, Any]):
        self.steps.append(TransformationEvent(step=step, timestamp=datetime.now().isoformat(), details=details))

    def as_dict(self) -> Dict[str, Any]:
        return {
            "start_time": self.start_time,
            "steps": [asdict(s) for s in self.steps],
        }


def _is_numeric_series(s: pd.Series) -> bool:
    return pd.api.types.is_numeric_dtype(s)


def _safe_to_numeric(df: pd.DataFrame) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """尽力将可数值列转为数值，记录转换信息。"""
    info = {"converted_columns": []}
    for col in df.columns:
        if df[col].dtype == object:
            # 尝试数值化（错误设为 NaN）
            converted = pd.to_numeric(df[col], errors="ignore")
            if pd.api.types.is_numeric_dtype(converted):
                df[col] = pd.to_numeric(df[col], errors="coerce")
                info["converted_columns"].append(col)
    return df, info


def standardize_columns(
    df: pd.DataFrame, mapping: Dict[str, str]
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """应用列名映射，避免冲突，记录映射前后关系。"""
    original_cols = list(df.columns)
    rename_map: Dict[str, str] = {}
    for col in original_cols:
        target = mapping.get(col, col)
        # 避免重名：若冲突，追加后缀
        if target in rename_map.values() and target != col:
            k = 1
            new_target = f"{target}__dup{k}"
            while new_target in rename_map.values():
                k += 1
                new_target = f"{target}__dup{k}"
            target = new_target
        rename_map[col] = target
    df = df.rename(columns=rename_map)
    return df, {"original_columns": original_cols, "applied_mapping": rename_map}


def handle_missing_and_zero_values(
    df: pd.DataFrame,
    id_columns: Optional[List[str]] = None,
    zero_to_nan_columns: Optional[List[str]] = None,
    group_fill_by: Optional[List[str]] = None,
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """
    系统化处理空值与零值：
    - 将 inf/-inf 统一设为 NaN
    - 针对部分列(如估值类)将0视为缺失
    - 分组前向/后向填充
    返回变更统计日志。
    """
    id_columns = id_columns or []
    zero_to_nan_columns = zero_to_nan_columns or []
    group_fill_by = group_fill_by or []

    log: Dict[str, Any] = {
        "replaced_inf_to_nan": {},
        "zero_to_nan": {},
        "ffill_bfill": {},
    }

    # 1) 将 inf/-inf 置为 NaN
    before_inf = {}
    for col in df.columns:
        if _is_numeric_series(df[col]):
            cnt = np.isinf(df[col].astype(float)).sum()
            if cnt:
                before_inf[col] = int(cnt)
    df.replace([np.inf, -np.inf], np.nan, inplace=True)
    log["replaced_inf_to_nan"] = before_inf

    # 2) 对指定列的0值置为 NaN（如估值类/比率类）
    zero_change = {}
    for col in zero_to_nan_columns:
        if col in df.columns and _is_numeric_series(df[col]):
            cnt0 = int((df[col] == 0).sum())
            if cnt0:
                df.loc[df[col] == 0, col] = np.nan
            zero_change[col] = cnt0
    log["zero_to_nan"] = zero_change

    # 3) 分组填充：优先按 group_fill_by 分组，其次全局
    filled = {}
    target_cols = [c for c in df.columns if c not in set(id_columns)]

    if group_fill_by and all(col in df.columns for col in group_fill_by):
        # 统计填充前缺失
        before_missing = df[target_cols].isna().sum().to_dict()
        df[target_cols] = df.groupby(group_fill_by)[target_cols].apply(lambda x: x.ffill().bfill())
        after_missing = df[target_cols].isna().sum().to_dict()
        filled = {col: int(before_missing.get(col, 0) - after_missing.get(col, 0)) for col in target_cols}
    else:
        before_missing = df[target_cols].isna().sum().to_dict()
        df[target_cols] = df[target_cols].ffill().bfill()
        after_missing = df[target_cols].isna().sum().to_dict()
        filled = {col: int(before_missing.get(col, 0) - after_missing.get(col, 0)) for col in target_cols}

    log["ffill_bfill"] = filled
    return df, log


def standardize_dataframe(
    df: pd.DataFrame,
    mapping: Dict[str, str],
    id_columns_aliases: Optional[List[Tuple[str, str]]] = None,
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """整体标准化流程：列映射 -> 数值化 -> 零/空处理 -> 记录日志。"""
    tlog = TransformationLog(start_time=datetime.now().isoformat(), steps=[])

    # 列名映射
    df, map_info = standardize_columns(df, mapping)
    tlog.add("standardize_columns", map_info)

    # 尝试数值化
    df, conv_info = _safe_to_numeric(df)
    tlog.add("safe_to_numeric", conv_info)

    # 关键ID列解析：支持别名回落
    date_col = CANONICAL_DATE
    code_col = CANONICAL_CODE

    # 零视为缺失的典型列（估值/比率类）：可按需扩展
    zero_as_nan = [c for c in ["pe", "pe_ttm", "pb"] if c in df.columns]

    # 分组填充按股票代码；若没有则全局
    group_cols = [code_col] if code_col in df.columns else []

    df, miss_log = handle_missing_and_zero_values(
        df,
        id_columns=[date_col, code_col],
        zero_to_nan_columns=zero_as_nan,
        group_fill_by=group_cols,
    )
    tlog.add("handle_missing_and_zero_values", miss_log)

    return df, tlog.as_dict()


def generate_data_quality_report(df: pd.DataFrame) -> Dict[str, Any]:
    """生成数据质量报告：缺失率、零值率、基本统计、异常值（zscore>5）。"""
    report: Dict[str, Any] = {
        "generated_at": datetime.now().isoformat(),
        "shape": list(df.shape),
        "columns": [],
    }

    for col in df.columns:
        col_info: Dict[str, Any] = {"column": col}
        s = df[col]
        col_info["dtype"] = str(s.dtype)
        col_info["missing_count"] = int(s.isna().sum())
        col_info["missing_rate"] = float(s.isna().mean())
        if _is_numeric_series(s):
            s_num = pd.to_numeric(s, errors="coerce")
            col_info["zero_count"] = int((s_num == 0).sum())
            col_info["zero_rate"] = float(((s_num == 0).mean()))
            desc = s_num.describe().to_dict()
            for k, v in desc.items():
                if isinstance(v, (np.floating, float)) and (np.isnan(v) or np.isinf(v)):
                    desc[k] = None
            col_info["stats"] = desc
            # 粗略异常：|z|>5
            try:
                z = (s_num - s_num.mean()) / (s_num.std(ddof=0) if s_num.std(ddof=0) != 0 else 1)
                col_info["outlier_count_absz_gt_5"] = int((z.abs() > 5).sum())
            except Exception:
                col_info["outlier_count_absz_gt_5"] = None
        else:
            col_info["zero_count"] = None
            col_info["zero_rate"] = None
            col_info["stats"] = None
            col_info["outlier_count_absz_gt_5"] = None
        report["columns"].append(col_info)
    return report


def save_json(obj: Dict[str, Any], file_path: str):
    os.makedirs(os.path.dirname(file_path), exist_ok=True)
    with open(file_path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)
