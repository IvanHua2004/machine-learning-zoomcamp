import io
import zipfile
from pathlib import Path

import requests

URL = "https://archive.ics.uci.edu/static/public/601/ai4i+2020+predictive+maintenance+dataset.zip"
OUT = Path("data/ai4i2020.csv")


def main():
    OUT.parent.mkdir(exist_ok=True)
    resp = requests.get(URL, timeout=60)
    resp.raise_for_status()
    with zipfile.ZipFile(io.BytesIO(resp.content)) as zf:
        name = next(n for n in zf.namelist() if n.endswith(".csv"))
        OUT.write_bytes(zf.read(name))
    print(f"saved {OUT}")


if __name__ == "__main__":
    main()
