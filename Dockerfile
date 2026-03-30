FROM python:3.10-slim
WORKDIR /app
RUN pip install fastapi uvicorn scikit-learn joblib numpy pydantic
COPY train_model.py .
RUN python train_model.py
COPY app/main.py .
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8080"]
