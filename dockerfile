FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE 1
ENV PYTHONUNBUFFERED 1

RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        gcc \
        libpq-dev \
        postgresql-server-dev-all \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY . /app/

COPY requirements.txt /app/
RUN pip install --upgrade pip \
    && pip install -r requirements.txt

#EXPOSE 8000

#CMD ["python", "manage.py", "runserver", "0.0.0.0:8000"]

CMD ["sh", "-c", "python manage.py migrate && python manage.py collectstatic --noinput && gunicorn NurtrDjango.wsgi:application --bind 0.0.0.0:8000 --timeout 300"]

