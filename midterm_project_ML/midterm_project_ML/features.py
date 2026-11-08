"""Feature preparation shared by train.py and predict.py.

Keeping this in one place guarantees the model sees exactly the same
features at training time and at serving time.
"""

import pandas as pd

# Raw CSV column names -> clean snake_case names
COLUMN_MAP = {
    "Type": "type",
    "Air temperature [K]": "air_temperature",
    "Process temperature [K]": "process_temperature",
    "Rotational speed [rpm]": "rotational_speed",
    "Torque [Nm]": "torque",
    "Tool wear [min]": "tool_wear",
    "Machine failure": "machine_failure",
}

# Columns that must NOT be used as features:
# - IDs carry no signal
# - TWF/HDF/PWF/OSF/RNF are the failure causes -> they leak the target
DROP_COLUMNS = ["UDI", "UID", "Product ID", "TWF", "HDF", "PWF", "OSF", "RNF"]

TARGET = "machine_failure"


def load_raw(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    df = df.drop(columns=[c for c in DROP_COLUMNS if c in df.columns])
    df = df.rename(columns=COLUMN_MAP)
    return df


def add_features(df: pd.DataFrame) -> pd.DataFrame:
    """Engineered features. Works on a DataFrame of one or many rows."""
    df = df.copy()
    # TODO(EDA): justify these in the notebook (plot failure rate vs each one),
    # and add/remove features based on what you find.
    df["temp_diff"] = df["process_temperature"] - df["air_temperature"]
    df["power"] = df["torque"] * df["rotational_speed"] * 2 * 3.141592653589793 / 60  # watts
    df["strain"] = df["tool_wear"] * df["torque"]
    return df
