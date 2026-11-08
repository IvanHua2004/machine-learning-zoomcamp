# Machine Failure Prediction (ML Zoomcamp Midterm Project)

Predict whether a milling machine is about to fail, based on its process readings, and serve the model as a web API.

## Problem description

> TODO: Write 2–3 paragraphs covering the points below. This section is worth 2 points.
> - **Context:** unplanned machine downtime is expensive, and predictive maintenance aims to catch failures before they happen.
> - **The data:** what each row represents and which sensor readings are included.
> - **How the model is used:** a monitoring system sends the current readings to `/predict` and gets back a failure probability. If the probability is above a threshold, a maintenance alert is raised.
> - **Why the metric matters:** failures are rare (~3.4%), so explain the cost of a missed failure compared with a false alarm, and why you chose your metric.

## Dataset

[AI4I 2020 Predictive Maintenance Dataset](https://archive.ics.uci.edu/dataset/601/ai4i+2020+predictive+maintenance+dataset) from the UCI Machine Learning Repository, licensed CC BY 4.0. It has 10,000 rows. The data is synthetic but modelled on real industrial predictive-maintenance data.

| Feature | Description |
|---|---|
| `type` | Product quality variant: L / M / H |
| `air_temperature` | Air temperature [K] |
| `process_temperature` | Process temperature [K] |
| `rotational_speed` | Spindle speed [rpm] |
| `torque` | Torque [Nm] |
| `tool_wear` | Tool wear [min] |
| **`machine_failure`** | **Target (0/1)** |

The ID columns and the five failure-mode columns (`TWF`, `HDF`, `PWF`, `OSF`, `RNF`) are dropped. The failure modes describe *why* a machine failed, so using them as features would leak the target.

The CSV is committed at `data/ai4i2020.csv`. To re-download it:

```bash
pipenv run python download_data.py
```

## Project structure

```
├── data/ai4i2020.csv      # dataset
├── notebook.ipynb         # cleaning, EDA, feature importance, model selection & tuning
├── features.py            # feature prep shared by training and serving
├── train.py               # trains the final model -> model.bin
├── predict.py             # FastAPI web service
├── test_predict.py        # sends a sample request to the service
├── download_data.py       # downloads the dataset
├── Pipfile / Pipfile.lock # dependencies
├── Dockerfile
├── k8s/                   # Kubernetes manifests
└── images/                # screenshots of the deployed service
```

## Results

> TODO: Add a table with each model's best validation score, then the final test score and the threshold you chose.

| Model | Val PR AUC | Val ROC AUC |
|---|---|---|
| Logistic regression | | |
| Decision tree | | |
| Random forest | | |
| XGBoost | | |

## How to run

### 1. Install dependencies (Pipenv)

```bash
pip install pipenv
pipenv install --dev        # creates the virtualenv + Pipfile.lock
pipenv shell                # activate the environment
```

### 2. Notebook

```bash
jupyter notebook notebook.ipynb
```

### 3. Train the model

```bash
python train.py             # writes model.bin
```

### 4. Run the service locally

```bash
uvicorn predict:app --host 0.0.0.0 --port 9696
python test_predict.py      # in another terminal
```

Interactive API docs are available at http://localhost:9696/docs.

### 5. Docker

```bash
docker build -t machine-failure:v1 .
docker run -it --rm -p 9696:9696 machine-failure:v1
python test_predict.py
```

### 6. Kubernetes (local cluster with kind)

```bash
kind create cluster
kind load docker-image machine-failure:v1
kubectl apply -f k8s/deployment.yaml
kubectl apply -f k8s/service.yaml
kubectl get pods                                   # wait for READY 1/1
kubectl port-forward service/machine-failure 9696:80
python test_predict.py
```

> TODO: Add a screenshot to `images/`, then embed it here.

Example request:

```bash
curl -X POST http://localhost:9696/predict \
  -H "Content-Type: application/json" \
  -d '{"type":"L","air_temperature":298.9,"process_temperature":309.1,"rotational_speed":1350,"torque":62.5,"tool_wear":220}'
```

Example response:

```json
{"failure_probability": 0.93, "failure": true}
```
