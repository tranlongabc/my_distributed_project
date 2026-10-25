FROM python:3.9-slim

WORKDIR /app

# Cập nhật và cài đặt các thư viện thiết yếu (dùng --no-install-recommends để giảm lỗi rác mạng)
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1 \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . /app