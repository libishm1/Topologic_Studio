FROM python:3.12-slim

# No build-essential: every dependency, topologic-core and ifcopenshell
# included, ships a manylinux wheel for CPython 3.12. Dropping the toolchain
# takes a few hundred MB off the image, which is what a cold start on Render
# has to pull. If a wheel ever goes missing the build fails here, loudly,
# rather than at the first request.
#
# TOPOLOGICPY_CORE_BACKEND: TopologicPy picks its geometry kernel at import
# ("auto"), and 0.9.80+ prefers PythonOCC whenever it is importable. Pin the
# kernel this image is tested with, so an unrelated dependency cannot switch it.
ENV PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    TOPOLOGICPY_CORE_BACKEND=topologic_core

WORKDIR /app

# requirements.txt is the single source of truth: it pins topologicpy AND
# topologic-core (0.9 split the native backend into its own distribution) and
# declares ifcopenshell, which Classic installed only inside the Dockerfile.
COPY requirements.txt ./
RUN pip install -r requirements.txt

COPY app ./app

# Compile the app's bytecode now so the first boot after a spin-down does not.
RUN python -m compileall -q app

ENV PORT=8000
EXPOSE 8000

# Fail the build rather than the first request if the native backend is missing.
RUN python -c "from app.graphs.topologic_engine import version_info; \
info = version_info(); print(info); \
raise SystemExit(0 if info['usable'] else 'topologicpy backend unusable')"

# Render probes healthCheckPath itself and ignores this; it is for local
# `docker run`. Kept infrequent so it does not compete with requests on a
# small instance.
HEALTHCHECK --interval=120s --timeout=5s --start-period=20s \
  CMD python -c "import urllib.request,os; urllib.request.urlopen(f\"http://127.0.0.1:{os.getenv('PORT','8000')}/health\")"

# One worker on purpose: graphs live in an in-process store, so a second
# worker would not see the first one's graph ids.
CMD ["sh", "-c", "uvicorn app.main:app --host 0.0.0.0 --port ${PORT:-8000}"]
