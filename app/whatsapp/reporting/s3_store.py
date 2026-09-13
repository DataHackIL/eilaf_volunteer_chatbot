"""Report store backed by an S3 bucket — the production destination.

Credentials are **not** configured here. On the EC2 host boto3 picks up the
instance role from the metadata service, which is why no AWS key appears in
``.env``: an instance role cannot leak through a stolen env file and rotates
itself. The role needs exactly one permission, ``s3:PutObject`` on the
configured prefix (see ``app/whatsapp/SETUP.md``).

Note this is the first thing in the repo that speaks the S3 *API*. The corpus
store reaches S3 the other way, as a filesystem path (``EILAF_STATIC_DIR``), and
that remains true — the two are unrelated.
"""

from __future__ import annotations

import json
from typing import Any

from app.whatsapp.reporting.base import ReportStore, report_path


class S3ReportStore(ReportStore):
    def __init__(self, bucket: str, prefix: str = "", region: str = "", client=None):
        self.bucket = bucket
        # Tolerate a prefix given with or without its trailing slash; joining
        # blindly would otherwise produce either "conversations2026/..." or a
        # double slash, both of which read as a different folder in the console.
        self.prefix = f"{prefix.strip('/')}/" if prefix.strip("/") else ""
        self.region = region
        self._client = client

    @property
    def client(self):
        """The boto3 client, built on first use.

        Lazily, so importing this module (and constructing the store) costs
        nothing on a machine with no AWS credentials at all — which is every dev
        machine and the whole test suite.
        """
        if self._client is None:
            import boto3

            self._client = boto3.client(
                "s3", **({"region_name": self.region} if self.region else {})
            )
        return self._client

    def save(self, report: dict[str, Any]) -> str:
        key = f"{self.prefix}{report_path(report)}"
        self.client.put_object(
            Bucket=self.bucket,
            Key=key,
            # ensure_ascii=False keeps Hebrew and Arabic readable in the console
            # and in anything that reads the object back, rather than storing
            # every letter as a \uXXXX escape.
            Body=json.dumps(report, ensure_ascii=False, indent=2).encode(),
            ContentType="application/json; charset=utf-8",
        )
        return f"s3://{self.bucket}/{key}"
