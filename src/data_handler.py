from pathlib import Path
import pandas as pd
import joblib

def load_csv(path: str | Path) -> pd.DataFrame:
    # Generic CSV loader

    path = Path(path)
    return pd.read_csv(path)


def save_csv(df: pd.DataFrame, path: str | Path, index: bool = False) -> None:
    # Generic CSV saver; creates folders if needed

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=index)

def save_features(X: pd.DataFrame | None, name: str):
    """
    Save feature matrix X and optional target y as PKL files.

    name example: 'flights', 'hotels'
    """
    FEATURE_DIR = Path(__file__).resolve().parent.parent / "data" / "features"
    FEATURE_DIR.mkdir(parents=True, exist_ok=True)

    joblib.dump(X, FEATURE_DIR / f"{name}_features.pkl")

def load_features(name: str):
    FEATURE_DIR = Path(__file__).resolve().parent.parent / "data" / "features"
    X = joblib.load(FEATURE_DIR / f"{name}_features.pkl")
    return X 
