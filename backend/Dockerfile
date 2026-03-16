FROM python:3.11-slim

RUN apt-get update && apt-get install -y --no-install-recommends \
    gdal-bin libgdal-dev gcc g++ \
    && rm -rf /var/lib/apt/lists/*

ENV GDAL_VERSION=3.6.2
ENV CPLUS_INCLUDE_PATH=/usr/include/gdal
ENV C_INCLUDE_PATH=/usr/include/gdal

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

RUN mkdir -p /app/media

EXPOSE 8000

CMD ["gunicorn", "core.wsgi:application", "--bind", "0.0.0.0:8000", "--workers", "2", "--timeout", "300"]
