import argparse
import concurrent.futures as futures
import logging
import os
import sys
import time
from dataclasses import dataclass
from typing import Optional

import boto3
from botocore.config import Config
from botocore.exceptions import ClientError
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


def list_objects(s3, bucket: str, prefix: str):
    paginator = s3.get_paginator("list_objects_v2")
    for page in paginator.paginate(Bucket=bucket, Prefix=prefix):
        for obj in page.get("Contents", []):
            yield obj["Key"], obj["Size"]


def download_one(s3, bucket: str, key: str, local_path: str, overwrite: bool, size: int, pbar: tqdm):
    if os.path.exists(local_path) and not overwrite and os.path.getsize(local_path) == size:
        pbar.update(1)
        return "skipped"
    
    os.makedirs(os.path.dirname(local_path), exist_ok=True)
    
    try:
        s3.download_file(bucket, key, local_path)
        pbar.update(1)
        return "downloaded"
    except Exception as e:
        pbar.update(1)
        print(f"Error downloading {key}: {e}", file=sys.stderr)
        return "error"


def main():
    parser = argparse.ArgumentParser(description="Download a folder from Cloudflare R2")
    parser.add_argument("--account-id", required=True, help="R2 account ID")
    parser.add_argument("--access-key", required=True, help="R2 access key ID")
    parser.add_argument("--secret-key", required=True, help="R2 secret access key")
    parser.add_argument("--bucket", required=True, help="Source bucket name")
    parser.add_argument("--remote-prefix", default="", help="Key prefix inside bucket (e.g., 'hyspecnet/')")
    parser.add_argument("--local-dir", required=True, help="Local directory to save files")
    parser.add_argument("--max-workers", type=int, default=16, help="Parallel downloads")
    parser.add_argument("--overwrite", type=str, default="false", choices=["true", "false"], help="Overwrite existing")
    parser.add_argument("--log-file", type=str, default=None, help="Path to a file to log progress")
    
    args = parser.parse_args()

    if args.log_file:
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(message)s',
            filename=args.log_file,
            filemode='w'
        )
        print(f"Logging progress to {args.log_file}")

    local_dir = os.path.abspath(args.local_dir)
    os.makedirs(local_dir, exist_ok=True)

    endpoint = f"https://{args.account_id}.r2.cloudflarestorage.com"
    r2cfg = R2Config(endpoint_url=endpoint, access_key=args.access_key, secret_key=args.secret_key)
    s3 = make_s3_client(r2cfg)

    try:
        s3.head_bucket(Bucket=args.bucket)
    except ClientError as e:
        if e.response["Error"]["Code"] in ("404", "NoSuchBucket"):
            print(f"Bucket {args.bucket} not found.", file=sys.stderr)
            sys.exit(2)
        else:
            raise

    files = list(list_objects(s3, args.bucket, args.remote_prefix))
    if not files:
        print("No files to download.", file=sys.stderr)
        sys.exit(0)

    total = len(files)
    downloaded = 0
    skipped = 0
    errors = 0
    log_interval = 100

    start_time = time.time()

    print(f"Downloading {total} files from s3://{args.bucket}/{args.remote_prefix} to {local_dir}")
    with tqdm(total=total, unit="file") as pbar:
        with futures.ThreadPoolExecutor(max_workers=args.max_workers) as ex:
            tasks = []
            for key, size in files:
                rel_path = os.path.relpath(key, args.remote_prefix) if args.remote_prefix else key
                local_path = os.path.join(local_dir, rel_path)
                tasks.append(ex.submit(
                    download_one, s3, args.bucket, key, local_path, args.overwrite == "true", size, pbar
                ))

            completed_count = 0
            for t in futures.as_completed(tasks):
                status = t.result()
                if status == "downloaded":
                    downloaded += 1
                elif status == "skipped":
                    skipped += 1
                elif status == "error":
                    errors += 1
                
                completed_count += 1
                
                if args.log_file and (completed_count % log_interval == 0 or completed_count == total):
                    progress_percent = (completed_count / total) * 100
                    
                    elapsed_time = time.time() - start_time
                    files_per_second = completed_count / elapsed_time if elapsed_time > 0 else 0
                    remaining_files = total - completed_count
                    
                    if files_per_second > 0:
                        eta_seconds = remaining_files / files_per_second
                        eta_formatted = time.strftime('%H:%M:%S', time.gmtime(eta_seconds))
                    else:
                        eta_formatted = "N/A"

                    log_message = (
                        f"Progress: {completed_count}/{total} files "
                        f"({progress_percent:.2f}%) processed. "
                        f"ETA: {eta_formatted}. "
                        f"(D:{downloaded}, S:{skipped}, E:{errors})"
                    )
                    logging.info(log_message)

    if args.log_file:
        total_time = time.time() - start_time
        total_time_formatted = time.strftime('%H:%M:%S', time.gmtime(total_time))
        logging.info(f"Download process finished in {total_time_formatted}.")

    print(f"Done. Downloaded: {downloaded}, Skipped: {skipped}, Errors: {errors}, Total: {total}")


if __name__ == "__main__":
    main()