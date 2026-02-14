import argparse
import concurrent.futures as futures
import fnmatch
import mimetypes
import os
import sys
from dataclasses import dataclass
from typing import Optional

import boto3
from botocore.config import Config
from botocore.exceptions import ClientError
from boto3.s3.transfer import TransferConfig
from tqdm import tqdm


@dataclass
class R2Config:
    endpoint_url: str
    access_key: str
    secret_key: str
    region_name: str = "auto"


def make_s3_client(cfg: R2Config):
    session = boto3.session.Session()
    return session.client(
        "s3",
        endpoint_url=cfg.endpoint_url,
        aws_access_key_id=cfg.access_key,
        aws_secret_access_key=cfg.secret_key,
        region_name=cfg.region_name,
        config=Config(
            retries={"max_attempts": 10, "mode": "standard"},
            signature_version="s3v4",
            read_timeout=120,
            connect_timeout=30,
        ),
    )


def iter_files(root: str, include: Optional[str] = None, exclude: Optional[str] = None):
    root = os.path.abspath(root)
    for dirpath, _, filenames in os.walk(root):
        for name in filenames:
            abs_path = os.path.join(dirpath, name)
            rel_path = os.path.relpath(abs_path, root)
            if exclude and fnmatch.fnmatch(rel_path, exclude):
                continue
            if include and not fnmatch.fnmatch(rel_path, include):
                continue
            yield abs_path, rel_path


def object_exists_with_same_size(s3, bucket: str, key: str, size: int) -> bool:
    try:
        resp = s3.head_object(Bucket=bucket, Key=key)
        return int(resp.get("ContentLength", -1)) == int(size)
    except ClientError as e:
        if e.response["Error"]["Code"] in ("404", "NoSuchKey", "NotFound"):
            return False
        raise


def content_type_for(path: str) -> str:
    ct, _ = mimetypes.guess_type(path)
    return ct or "application/octet-stream"


def upload_one(
    s3,
    bucket: str,
    local_path: str,
    key: str,
    cfg: TransferConfig,
    overwrite: bool,
    pbar: tqdm,
):
    size = os.path.getsize(local_path)
    if not overwrite and object_exists_with_same_size(s3, bucket, key, size):
        pbar.update(1)
        return "skipped"

    extra_args = {"ContentType": content_type_for(local_path)}
    s3.upload_file(local_path, bucket, key, ExtraArgs=extra_args, Config=cffg)
    pbar.update(1)
    return "uploaded"


def main():
    parser = argparse.ArgumentParser(description="Upload a folder to Cloudflare R2")
    parser.add_argument("--account-id", required=True, help="R2 account ID")
    parser.add_argument("--access-key", required=True, help="R2 access key ID")
    parser.add_argument("--secret-key", required=True, help="R2 secret access key")
    parser.add_argument("--bucket", required=True, help="Target bucket name")
    parser.add_argument("--local-dir", required=True, help="Local directory to upload")
    parser.add_argument("--prefix", default="", help="Key prefix inside bucket (e.g., 'hyspecnet/')")
    parser.add_argument("--include", default=None, help="Glob to include (optional)")
    parser.add_argument("--exclude", default=None, help="Glob to exclude (optional)")
    parser.add_argument("--max-workers", type=int, default=16, help="Parallel uploads")
    parser.add_argument("--multipart-chunk-mb", type=int, default=64, help="Multipart chunk size MB")
    parser.add_argument("--overwrite", type=str, default="false", choices=["true", "false"], help="Overwrite existing")
    args = parser.parse_args()

    local_dir = os.path.abspath(args.local_dir)
    if not os.path.isdir(local_dir):
        print(f"Local dir not found: {local_dir}", file=sys.stderr)
        sys.exit(1)

    endpoint = f"https://{args.account_id}.r2.cloudflarestorage.com"
    r2cfg = R2Config(endpoint_url=endpoint, access_key=args.access_key, secret_key=args.secret_key)
    s3 = make_s3_client(r2cfg)

    # Ensure bucket exists (won't create if already present)
    try:
        s3.head_bucket(Bucket=args.bucket)
    except ClientError as e:
        if e.response["Error"]["Code"] in ("404", "NoSuchBucket"):
            print(f"Bucket {args.bucket} not found.", file=sys.stderr)
            sys.exit(2)
        else:
            raise

    files = list(iter_files(local_dir, include=args.include, exclude=args.exclude))
    if not files:
        print("No files to upload.", file=sys.stderr)
        sys.exit(0)

    # Transfer config (multipart + concurrency)
    chunk = args.multipart_chunk_mb * 1024 * 1024
    global cffg
    cffg = TransferConfig(
        multipart_threshold=chunk,      # files >= this use multipart
        multipart_chunksize=chunk,      # part size
        max_concurrency=args.max_workers,
        use_threads=True,
    )

    total = len(files)
    uploaded = 0
    skipped = 0

    print(f"Uploading {total} files from {local_dir} to s3://{args.bucket}/{args.prefix}")
    with tqdm(total=total, unit="file") as pbar:
        with futures.ThreadPoolExecutor(max_workers=args.max_workers) as ex:
            tasks = []
            for abs_path, rel_path in files:
                key = os.path.join(args.prefix, rel_path).replace("\\", "/")
                tasks.append(ex.submit(upload_one, s3, args.bucket, abs_path, key, cffg, args.overwrite == "true", pbar))

            for t in futures.as_completed(tasks):
                status = t.result()
                if status == "uploaded":
                    uploaded += 1
                elif status == "skipped":
                    skipped += 1

    print(f"Done. Uploaded: {uploaded}, Skipped: {skipped}, Total: {total}")


if __name__ == "__main__":
    main()
