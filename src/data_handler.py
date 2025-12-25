from pathlib import Path
import pandas as pd


def load_csv(path: str | Path) -> pd.DataFrame:
    # Generic CSV loader

    path = Path(path)
    return pd.read_csv(path)


def save_csv(df: pd.DataFrame, path: str | Path, index: bool = False) -> None:
    # Generic CSV saver; creates folders if needed

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=index)