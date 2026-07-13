import numpy as np
import pandas as pd


def load_eswf32(filepath: str, new_col_name: str = "eswf3_2") -> pd.DataFrame:
    df = pd.read_csv(filepath, parse_dates=["DATE"], na_values=["", " "])
    df = df.rename(columns={"DATE": "datetime", "V": new_col_name})
    df = df.dropna(subset=["datetime"])
    df["datetime"] = pd.to_datetime(df["datetime"])

    # 정각 기준 ±30분 이내만 유지
    rounded = df["datetime"].dt.round("1h")
    diff_minutes = (df["datetime"] - rounded).abs().dt.total_seconds() / 60
    df = df[diff_minutes <= 30].copy()
    df["rounded"] = rounded[diff_minutes <= 30].values
    df["diff_min"] = diff_minutes[diff_minutes <= 30].values

    # 정각에 가까울수록 높은 가중치 (역수 가중평균)
    # diff_min == 0인 경우 ZeroDivisionError 방지
    df["weight"] = 1.0 / (df["diff_min"] + 0.5)

    def weighted_mean(group):
        w = group["weight"]
        v = group[new_col_name]
        return (w * v).sum() / w.sum()

    result = (
        df.groupby("rounded")
          .apply(weighted_mean)
          .reset_index()
    )
    result.columns = ["datetime", new_col_name]

    return result.reset_index(drop=True)


def eswf32_from_file(df: pd.DataFrame,
                     filepath: str,
                     new_col_name: str = "eswf3_2") -> pd.DataFrame:
    base = df[["datetime", "speed"]].copy()
    base["datetime"] = pd.to_datetime(base["datetime"])

    eswf32 = load_eswf32(filepath, new_col_name=new_col_name)

    merged = pd.merge(base, eswf32, on="datetime", how="left")

    return merged[["datetime", "speed", new_col_name]].copy()

