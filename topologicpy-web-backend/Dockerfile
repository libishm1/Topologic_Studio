FROM python:3.12-slim

# scipy and shapely need a toolchain only if a wheel is unavailable; the slim
# image plus these headers covers the platforms Render builds on.
RUN apt-get update \
 && apt-get install -y --no-install-recommends build-essential \
 && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# requirements.txt is now the single source of truth: it pins topologicpy AND
# topologic-core (0.9 split the native backend into its own distribution) and
# declares ifcopenshell, which Classic installed only inside the Dockerfile.
COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

COPY app ./app

ENV PORT=8000
EXPOSE 8000

# Fail the build rather than the first request if the native backend is missing.
RUN python -c "from app.graphs.topologic_engine import version_info; \
info = version_info(); print(info); \
raise SystemExit(0 if info['usable'] else 'topologicpy backend unusable')"

HEALTHCHECK --interval=30s --timeout=5s --start-period=20s \
  CMD python -c "import urllib.request,os; urllib.request.urlopen(f\"http://127.0.0.1:{os.getenv('PORT','8000')}/health\")"

CMD ["sh", "-c", "uvicorn app.main:app --host 0.0.0.0 --port ${PORT:-8000}"]
