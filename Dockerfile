# ======================================================
# STAGE: DEVELOP (Development Environment)
# ======================================================
FROM python:3.11-slim AS develop

# ติดตั้ง system dependencies สำหรับ sentence-transformers
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgomp1 \
    libstdc++6 \
    libc6 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Copy requirements.txt ก่อนเพื่อใช้ Docker cache
COPY requirements.txt /app/

# ติดตั้ง Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy code และ data
COPY code/ /app/code/
COPY local_chroma_v2/ /app/local_chroma_v2/

# ตั้งค่า environment variables
ENV PYTHONPATH=/app/code
ENV PYTHONUNBUFFERED=1

# Expose port
EXPOSE 3000

# รัน uvicorn ด้วย auto-reload สำหรับ development
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "3000", "--reload"]

# ======================================================
# STAGE: STAGING (Pre-production Testing)
# ======================================================
FROM python:3.11-slim AS staging

RUN apt-get update && apt-get install -y --no-install-recommends \
    libgomp1 \
    libstdc++6 \
    libc6 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt /app/
RUN pip install --no-cache-dir -r requirements.txt

COPY code/ /app/code/
COPY local_chroma_v2/ /app/local_chroma_v2/

ENV PYTHONPATH=/app/code
ENV PYTHONUNBUFFERED=1

EXPOSE 3000

# ใช้ --reload เหมือน dev เพื่อทดสอบ
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "3000", "--reload"]

# ======================================================
# STAGE: PRODUCTION (Optimized for Performance)
# ======================================================
FROM python:3.11-slim AS prod

RUN apt-get update && apt-get install -y --no-install-recommends \
    libgomp1 \
    libstdc++6 \
    libc6 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt /app/
RUN pip install --no-cache-dir -r requirements.txt

COPY code/ /app/code/
COPY local_chroma_v2/ /app/local_chroma_v2/

ENV PYTHONPATH=/app/code
ENV PYTHONUNBUFFERED=1

EXPOSE 3000

# Production: ไม่มี --reload, เพิ่ม workers สำหรับ performance
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "3000", "--workers", "2"]