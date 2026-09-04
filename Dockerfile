# ---- 构建阶段：安装依赖 ----
FROM python:3.11-slim AS builder
WORKDIR /app
COPY requirements.txt .

# PIP_INDEX_URL 可选：处于内网 / 防火墙后时用国内镜像加速或绕过封锁
# 例：docker build --build-arg PIP_INDEX_URL=https://mirrors.tencent.com/pypi/simple/ .
ARG PIP_INDEX_URL=""
RUN if [ -n "$PIP_INDEX_URL" ]; then \
      pip install --no-cache-dir --prefix=/install -r requirements.txt -i "$PIP_INDEX_URL"; \
    else \
      pip install --no-cache-dir --prefix=/install -r requirements.txt; \
    fi

# ---- 运行阶段：非 root + 健康检查 ----
FROM python:3.11-slim
WORKDIR /app
COPY --from=builder /install /usr/local
COPY . .

# 持久化向量库存放目录，赋予非 root 写权限
RUN useradd -m appuser \
 && mkdir -p /app/store && chown -R appuser:appuser /app
USER appuser

EXPOSE 8080
ENV PORT=8080
HEALTHCHECK --interval=30s --timeout=5s --start-period=10s --retries=3 \
  CMD python -c "import urllib.request,sys; sys.exit(0 if urllib.request.urlopen('http://localhost:8080/healthz').status==200 else 1)"

CMD ["python", "app.py"]
