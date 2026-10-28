"""Unsigned S3 access to OpenNeuro public datasets.

OpenNeuro mirrors every dataset to the public, read-only S3 bucket
``s3://openneuro.org/<dataset_id>/``. The bucket allows anonymous (unsigned)
requests, so no AWS credentials are required. Passing UNSIGNED is mandatory:
without it boto3 searches for credentials and raises NoCredentialsError even
though the objects are world-readable.

This module is transport-only. Policy decisions (whether a download is large
enough to require explicit confirmation) live in ``server.py`` so they stay
visible at the tool boundary.
"""

from __future__ import annotations

import os
from dataclasses import dataclass

import boto3
from botocore import UNSIGNED
from botocore.config import Config

BUCKET = "openneuro.org"
REGION = "us-east-1"


def _client():
    """Return a boto3 S3 client configured for anonymous access."""
    return boto3.client(
        "s3",
        region_name=REGION,
        config=Config(signature_version=UNSIGNED),
    )


@dataclass(frozen=True)
class S3Object:
    key: str
    size: int  # bytes


def list_objects(dataset_id: str, subprefix: str = "") -> list[S3Object]:
    """List every object under ``<dataset_id>/<subprefix>`` (paginated)."""
    prefix = f"{dataset_id}/{subprefix}".rstrip("/")
    if subprefix:
        prefix += "/" if not prefix.endswith("/") else ""
    else:
        prefix = f"{dataset_id}/"
    client = _client()
    paginator = client.get_paginator("list_objects_v2")
    out: list[S3Object] = []
    for page in paginator.paginate(Bucket=BUCKET, Prefix=prefix):
        for obj in page.get("Contents", []):
            out.append(S3Object(key=obj["Key"], size=obj["Size"]))
    return out


def total_size(objects: list[S3Object]) -> int:
    """Sum of object sizes in bytes."""
    return sum(o.size for o in objects)


def download_object(key: str, dest_path: str) -> int:
    """Download a single object to ``dest_path``. Returns bytes written."""
    os.makedirs(os.path.dirname(os.path.abspath(dest_path)), exist_ok=True)
    client = _client()
    client.download_file(BUCKET, key, dest_path)
    return os.path.getsize(dest_path)
