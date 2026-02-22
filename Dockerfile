FROM python:3.12-slim-bookworm AS buildstage
LABEL maintainer="dxf209@student.bham.ac.uk"

ENV PIP_NO_CACHE_DIR=1

WORKDIR /build

# For captum
RUN apt-get update && \
    apt-get install -y --no-install-recommends g++ && \
    rm -rf /var/lib/apt/lists/*

COPY requirements.txt .

RUN pip install --upgrade pip && \
    pip wheel --no-cache-dir --no-deps --wheel-dir /wheels -r requirements.txt

WORKDIR /app
COPY letsfool/static/* /app/static/
COPY letsfool/main.py /app
COPY letsfool/mnist_model.py /app
COPY letsfool/mnist_model.pth /app


# ----------- Runtime container ---------------
FROM python:3.12-slim-bookworm
LABEL maintainer="dxf209@student.bham.ac.uk"

EXPOSE 8080

# Create a non-privileged user
RUN groupadd -r user && useradd -r -s /bin/false -g user user

WORKDIR /app

# Install packages from pre-built wheel
COPY --from=buildstage /wheels /wheels

RUN pip install --upgrade pip && \
    pip install --no-cache-dir /wheels/* && \
    rm -rf /wheels

# Copy application code
COPY . .
RUN chown -R user:user /app

# Copy files to runtime
COPY --from=buildstage /app /app

# Set up a dir for matplotlib to write to
ENV MPLBACKEND=Agg
ENV MPLCONFIGDIR=/home/user/.config/matplotlib

RUN mkdir -p /home/user/.config/matplotlib && \
    chown -R user:user /home/user

USER user

HEALTHCHECK CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8080/health')"
CMD ["gunicorn", "main:app", "--worker-class", "uvicorn.workers.UvicornWorker", "--bind", "0.0.0.0:8080", "--workers", "1", "--timeout", "60", "--access-logfile", "-", "--error-logfile", "-"]
