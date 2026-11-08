FROM python:3.12-slim

RUN pip install --no-cache-dir pipenv

WORKDIR /app

# Install only runtime (non-dev) dependencies into the system Python
COPY Pipfile Pipfile.lock ./
RUN pipenv install --system --deploy

COPY features.py predict.py model.bin ./

EXPOSE 9696

ENTRYPOINT ["uvicorn", "predict:app", "--host", "0.0.0.0", "--port", "9696"]
