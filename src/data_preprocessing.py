# Imports
from pathlib import Path
import pandas as pd
import numpy as np
from typing import Tuple, Dict
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, StandardScaler, PowerTransformer

# Making a working copy for peprocessing and feature engineering
def preprocess(df: pd.DataFrame) -> pd.DataFrame:
    # making a temporary file to work on data wrangling
    df = df.copy()  
    return df

# Cleaning outliers
def cap_outliers(df: pd.DataFrame, cols: list, q: float = 0.99) -> pd.DataFrame:
    # Cap numeric columns at given upper quantile.
    for c in cols:
        if c in df.columns:
            upper = df[c].quantile(q)
            df[c] = np.where(df[c] > upper, upper, df[c])
    return df

# Encoding columns
def col_encode(df, label_cols, onehot_cols, encoders=None):
    encoders = {} if encoders is None else encoders

    for col in label_cols:
        if col in df:
            le = LabelEncoder()
            df[col] = df[col].astype(str)
            df[f"{col}_enc"] = le.fit_transform(df[col])
            encoders[col] = {"type": "label", "encoder": le}

    for col in onehot_cols:
        if col in df:
            ohe = OneHotEncoder(sparse_output=False, handle_unknown="ignore")
            arr = ohe.fit_transform(df[[col]])
            ohe_df = pd.DataFrame(arr, index=df.index,
                                  columns=[f"{col}_{v}" for v in ohe.categories_[0]])
            df = pd.concat([df.drop(columns=col), ohe_df], axis=1)
            encoders[col] = {"type": "onehot", "encoder": ohe}

    return df, encoders

# Transforming columns, based on skewness
def col_transform(df, skew_threshold=0.75, fitted_power=None):

    cols = df.columns
    skew_vals = df[cols].skew()
    log_cols, power_cols = {}, []

    for col in cols:
        s = skew_vals.get(col, 0)
        if s > skew_threshold:
            shift = max(0, -df[col].min() + 1e-6)
            df[f"{col}_log"] = np.log1p(df[col] + shift)
            log_cols[col] = True
        elif s < -skew_threshold:
            power_cols.append(col)

    pt = fitted_power
    if power_cols:
        if pt is None:
            pt = PowerTransformer(method="yeo-johnson", standardize=True)
            arr = pt.fit_transform(df[power_cols])
        else:
            arr = pt.transform(df[power_cols])

        for i, col in enumerate(power_cols):
            df[f"{col}_pow"] = arr[:, i]

    return df, log_cols, pt

# Scaling the data
def col_scaling(df: pd.DataFrame, scaler: StandardScaler | None = None) -> Tuple[pd.DataFrame, StandardScaler]:
    
    cols = df.columns
    scaler = scaler or StandardScaler()

    df[[c + "_sc" for c in cols]] = scaler.fit_transform(df[cols].fillna(0))

    return df, scaler

# flight data preprocessing
def preprocess_flights(df: pd.DataFrame | None = None) -> pd.DataFrame:

    df = preprocess(df) # copy of the data

    # date handling
    if "date" in df.columns:
        df["date"] = pd.to_datetime(df["date"], errors="coerce")
        df["month"] = df["date"].dt.month
        df["weekday"] = df["date"].dt.dayofweek
        df["is_weekend"] = (df["weekday"] >= 5).astype(int)
        df["quarter"] = df["date"].dt.quarter

    # season
    if "month" in df.columns:
        df["season"] = pd.cut(
            df["month"],
            bins=[0, 3, 6, 9, 12],
            labels=["winter", "spring", "summer", "fall"],
            right=False,
        )

    # route feature
    df["route"] = df["from"] + "_" + df["to"]

    # price per km derived numeric
    df["price_per_km"] = df["price"] / (df["distance"] + 1)

    cols            = df.columns.tolist()
    label_cols      = ["flighttype"]
    onehot_cols     = ["from", "to", "agency", "season", "route"]

    df = cap_outliers(df, cols)
    df, encoder = col_encode(df, label_cols, onehot_cols)
    df, log_info, power_t = col_transform(df)
    df, scaler = col_scaling(df)

    return  df, encoder, log_info, power_t, scaler
    