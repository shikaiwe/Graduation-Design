"""时间序列插值工具"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Optional, Sequence

import pandas as pd


@dataclass(frozen=True)
class InterpolationConfig:
    """控制月度到日度插值行为的配置项"""

    method: str = "linear"
    limit_direction: str = "both"
    fill_strategy: str = "ffill"  # 可选: ffill | bfill | both | none
    preserve_monthly: bool = True


class MonthlyToDailyInterpolator:
    """将月度时间序列转换为日度时间序列的插值器"""

    def __init__(self, config: Optional[InterpolationConfig] = None) -> None:
        self.config = config or InterpolationConfig()

    def convert(
        self,
        monthly_df: pd.DataFrame,
        date_column: str,
        value_columns: Sequence[str],
        daily_calendar: Optional[Sequence[pd.Timestamp]] = None
    ) -> pd.DataFrame:
        """执行插值转换"""

        if monthly_df.empty:
            return pd.DataFrame(columns=[date_column, *value_columns])

        df = monthly_df.copy()
        df[date_column] = pd.to_datetime(df[date_column])
        df = df.sort_values(date_column).drop_duplicates(subset=[date_column])
        df = df.set_index(date_column)

        missing_columns = [col for col in value_columns if col not in df.columns]
        if missing_columns:
            raise ValueError(f"Monthly dataframe missing columns: {missing_columns}")

        for column in value_columns:
            df[column] = pd.to_numeric(df[column], errors="coerce")

        if daily_calendar is not None:
            target_index = pd.DatetimeIndex(pd.to_datetime(daily_calendar))
            target_index = target_index.unique().sort_values()
        else:
            start = df.index.min()
            end = df.index.max()
            target_index = pd.date_range(start=start, end=end, freq="D")

        # 先对所有日度日期进行扩展
        reindexed_df = df.reindex(target_index)

        interpolated_df = reindexed_df[list(value_columns)].interpolate(
            method=self.config.method,
            limit_direction=self.config.limit_direction
        )

        interpolated_df = self._fill_endpoints(interpolated_df)

        if self.config.preserve_monthly:
            for column in value_columns:
                monthly_column = f"{column}_monthly_ref"
                interpolated_df[monthly_column] = df[column]

        interpolated_df.index.name = date_column
        interpolated_df = interpolated_df.reset_index()

        return interpolated_df

    def _fill_endpoints(self, df: pd.DataFrame) -> pd.DataFrame:
        strategy = self.config.fill_strategy.lower()

        if strategy == "ffill":
            return df.ffill()
        if strategy == "bfill":
            return df.bfill()
        if strategy == "both":
            return df.ffill().bfill()
        if strategy == "none":
            return df

        raise ValueError(f"Unknown fill strategy: {self.config.fill_strategy}")


def convert_monthly_to_daily(
    monthly_df: pd.DataFrame,
    date_column: str,
    value_columns: Iterable[str],
    daily_calendar: Optional[Sequence[pd.Timestamp]] = None,
    config: Optional[InterpolationConfig] = None
) -> pd.DataFrame:
    """便捷函数：将月度数据转换为日度数据"""

    interpolator = MonthlyToDailyInterpolator(config)
    return interpolator.convert(
        monthly_df=monthly_df,
        date_column=date_column,
        value_columns=list(value_columns),
        daily_calendar=daily_calendar
    )
