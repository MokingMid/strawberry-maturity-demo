# 草莓成熟度检测 Demo —— Docker 镜像
#
# 构建：docker build -t strawberry-maturity-demo .
# 运行：docker run -p 8000:8000 strawberry-maturity-demo
#
# 镜像基于 CPU 版 torch，体积约 2GB。首次启动加载模型约 10~20 秒。

FROM python:3.11-slim

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    PORT=8000

# OpenCV 需要 libGL 等运行库
RUN apt-get update && apt-get install -y --no-install-recommends \
        libgl1 libglib2.0-0 libsm6 libxext6 libxrender1 \
        fonts-wqy-zenhei fonts-noto-cjk \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# 先装依赖，利用 Docker 层缓存
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt \
    && apt-get clean

COPY . .

# 预热模型：构建时就加载一次，避免容器首次请求时卡顿
RUN python -c "import sys; sys.path.insert(0,'demo'); \
from paths import find_weights; \
print('detector  ->', find_weights('detector')); \
print('classifier->', find_weights('classifier'))"

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=5s --start-period=40s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8000/api/samples?size=1').read()" || exit 1

CMD ["gunicorn", "--bind", "0.0.0.0:8000", "--workers", "1", "--timeout", "120", \
     "--access-logfile", "-", "app:app"]
