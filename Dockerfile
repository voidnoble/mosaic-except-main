# MOFY Phase 1 — Python 3.10, CUDA 미사용(CPU) 기본 이미지
# GPU 사용 시 nvidia/cuda 베이스로 교체할 것 (Phase 2)
FROM python:3.10-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

RUN apt-get update && apt-get install -y --no-install-recommends \
        libgl1 libglib2.0-0 ffmpeg \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY pyproject.toml .
RUN pip install --upgrade pip && pip install -e .

# baselinewithfacenet/ 와 tracking/ 을 /app 에 복사해 사용
# COPY baselinewithfacenet/ ./baselinewithfacenet/
# COPY tracking/ ./tracking/
# COPY pipeline.py eval_baseline.py ./

# ENTRYPOINT ["python", "pipeline.py"]
