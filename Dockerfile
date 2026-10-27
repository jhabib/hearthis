FROM python:3.12-slim-bookworm
ENV PYTHONUNBUFFERED=1 PIP_NO_CACHE_DIR=1 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
RUN apt-get update && apt-get install -y --no-install-recommends ffmpeg libsndfile1 \
    && rm -rf /var/lib/apt/lists/* \
    && useradd --uid 1000 --create-home hearthis \
    && mkdir /data && chown hearthis:hearthis /data
WORKDIR /app
COPY pyproject.toml /tmp/pyproject.toml
RUN pip install torch==2.5.1 torchaudio==2.5.1 --index-url https://download.pytorch.org/whl/cpu \
    && python -c "import tomllib,subprocess; p=tomllib.load(open('/tmp/pyproject.toml','rb'))['project']; subprocess.check_call(['pip','install',*p['dependencies'],*p['optional-dependencies']['serve'],*p['optional-dependencies']['analysis']])"
COPY pyproject.toml README.md ./
COPY src ./src
RUN pip install --no-deps .
USER 1000:1000
EXPOSE 8080
CMD ["python", "-m", "hearthis.service", "api"]
