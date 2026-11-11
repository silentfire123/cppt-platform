FROM python:3.11-slim-bookworm
ENV PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1 PIP_DISABLE_PIP_VERSION_CHECK=1 TZ=Asia/Shanghai
WORKDIR /app
COPY requirements-storage.txt ./
RUN pip install --no-cache-dir -r requirements-storage.txt && pip check
COPY backend ./backend
COPY scripts ./scripts
COPY data ./data
RUN mkdir -p /workspace/runtime /workspace/artifacts /workspace/datasets /workspace/uploads /workspace/model-cache /workspace/training-work && chown -R 1000:1000 /workspace
EXPOSE 8010 8011
HEALTHCHECK --interval=30s --timeout=5s --start-period=30s --retries=3 CMD python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8010/api/health', timeout=3).read()"
CMD ["python", "scripts/run_backend_linux.py"]
