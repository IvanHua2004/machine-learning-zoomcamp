import sys

import requests

url = (sys.argv[1] if len(sys.argv) > 1 else "http://localhost:9696").rstrip("/")

reading = {
    "type": "L",
    "air_temperature": 298.9,
    "process_temperature": 309.1,
    "rotational_speed": 1350,
    "torque": 62.5,
    "tool_wear": 220,
}

response = requests.post(f"{url}/predict", json=reading, timeout=10)
response.raise_for_status()
print(response.json())
