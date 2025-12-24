import os
from pathlib import Path
import pandas as pd

def wrangle(df: pd.DataFrame) -> pd.DataFrame:
    # making a temporary file to work on data wrangling
    df = df.copy()  
    return df

def drop_null_values(df: pd.DataFrame, max_null_ratio: float = 0.05) -> pd.DataFrame:
    # drop null values, if the data to null value ratio is less than 5%

    threshold = int(df.shape[1] * (1 - max_null_ratio))
    return df.dropna(thresh=threshold)

def drop_duplicates(df: pd.DataFrame, subset=None, keep="first") -> pd.DataFrame:
   # drop the duplicates if any

    return df.drop_duplicates(subset=subset, keep=keep).reset_index(drop=True)

def standardize_column_names(df: pd.DataFrame) -> pd.DataFrame:
    # column names will be saved in a standard format

    df.columns = (df.columns.str.strip().str.lower().str.replace(" ", "_"))
    return df

def strip_whitespace(df: pd.DataFrame) -> pd.DataFrame:
    # clean columns with striping white spaces for better processing

    str_cols = df.select_dtypes(include="object").columns
    for col in str_cols:
        df[col] = df[col].astype(str).str.strip()
    return df

def enforce_dtypes(df: pd.DataFrame, dtype_map: dict) -> pd.DataFrame:
    # Cast columns to given dtypes where possible, dtype_map = {"price": "float", "distance": "float", "days": "int"}

    for col, dt in dtype_map.items():
        if col in df.columns:
            if dt in ("int", "float"):
                df[col] = pd.to_numeric(df[col], errors="coerce")
            else:
                df[col] = df[col].astype(dt)
    return df

def convert_dates(df: pd.DataFrame, date_cols) -> pd.DataFrame:
    # datatype correction for dates

    for col in date_cols:
        if col in df.columns:
            df[col] = pd.to_datetime(df[col], errors="coerce")
    return df

def wrangle_flights(df: pd.DataFrame) -> pd.DataFrame:
    # Clean and standardize flights table, Returns a new wrangled DataFrame, original is untouched.
    
    df = wrangle(df)               # work on a copy
    df = drop_null_values(df)
    df = drop_duplicates(df)
    df = standardize_column_names(df)
    df = strip_whitespace(df)

    dtype_map = {
        "travelCode"    : "int64", 
        "userCode"      : "int64",
        "from"          : "string",
        "to"            : "string", 
        "flightType"    : "string", 
        "price"         : "float64",
        "time"          : "float64",
        "distance"      : "float64",
        "agency"        : "string"
        }
    df = enforce_dtypes(df, dtype_map)
    df = convert_dates(df, ["date"])

    return df

    