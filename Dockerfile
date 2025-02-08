# ใช้ Python 3.8 image
FROM python:3.8-slim

# ติดตั้ง dependencies
RUN pip install mlflow scikit-learn

# ตั้ง working directory
WORKDIR /app

# คัดลอกโค้ดของคุณไปยัง container
COPY . /app

# รันคำสั่งเพื่อเริ่ม MLflow UI
CMD ["mlflow", "ui", "--port", "5000", "--host", "0.0.0.0"]
