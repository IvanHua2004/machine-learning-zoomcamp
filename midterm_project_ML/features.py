import pandas as pd

COLUMN_MAP = {
    "Type": "type",
    "Air temperature [K]": "air_temperature",
    "Process temperature [K]": "process_temperature",
    "Rotational speed [rpm]": "rotational_speed",
    "Torque [Nm]": "torque",
    "Tool wear [min]": "tool_wear",
    "Machine failure": "machine_failure",
}

DROP_COLUMNS = ["UDI", "UID", "Product ID", "TWF", "HDF", "PWF", "OSF", "RNF"]

TARGET = "machine_failure"


def load_raw(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    df = df.drop(columns=[c for c in DROP_COLUMNS if c in df.columns])
    df = df.rename(columns=COLUMN_MAP)
    return df


def add_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["temp_diff"] = df["process_temperature"] - df["air_temperature"]
    df["power"] = df["torque"] * df["rotational_speed"] * 2 * 3.141592653589793 / 60
    df["strain"] = df["tool_wear"] * df["torque"]
    return df
