import os
import ssl
from celery import Celery

REDIS_URL = os.environ.get("CELERY_BROKER_URL")

if not REDIS_URL:
    raise ValueError("CELERY_BROKER_URL not set")

celery = Celery(
    "worker",
    broker=REDIS_URL,
    backend=REDIS_URL
)

celery.conf.update(
    broker_use_ssl={"ssl_cert_reqs": ssl.CERT_NONE},
    redis_backend_use_ssl={"ssl_cert_reqs": ssl.CERT_NONE},
    task_serializer="json",
    accept_content=["json"],
    result_serializer="json",
    timezone="UTC",
    enable_utc=True
)
