import logging
import os
import tomllib
from datetime import datetime
from pathlib import Path

import boto3
import yfinance as yf
from boto3.s3.transfer import TransferConfig
from botocore.exceptions import ClientError
from dotenv import load_dotenv
from mypy_boto3_s3 import S3Client
from pandas import DataFrame
from tenacity import retry, stop_after_attempt, wait_exponential

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
log = logging.getLogger(__name__)


# -------------------------- DOWNLOAD DATA FROM YFINANCE ------------------
@retry(stop=stop_after_attempt(3), wait=wait_exponential(min=2, max=10))
def _download(
    tickers: list[str],
    start_date: datetime | None,
    end_date: datetime | None,
    period: str | None,
) -> DataFrame | None:
    if period:
        return yf.download(tickers=tickers, period=period, group_by="Ticker", progress=False)
    return yf.download(
        tickers=tickers,
        start=start_date,
        end=end_date,
        group_by="Ticker",
        progress=False,
    )


def get_storage_path() -> Path:
    base_dir = Path(__file__).resolve().parent.parent
    data_dir = base_dir / "data" / "raw"
    data_dir.mkdir(parents=True, exist_ok=True)
    return data_dir


def download_data(
    tickers: list[str],
    start_date: datetime | None = None,
    end_date: datetime | None = None,
    period: str | None = "1y",
    upload: bool = False,
) -> Path:
    file_timestamp = datetime.now().strftime("%y%m%d")
    if period is None:
        start_date = start_date or datetime.now().replace(month=1, day=1)
        end_date = end_date or datetime.now()
        log.info(f"Fetching {len(tickers)} tickers | Period : {start_date:%Y-%m-%d} to {end_date:%Y-%m-%d}")
        file_name = f"Prices_{start_date:%Y%m%d}_{end_date:%Y%m%d}_{file_timestamp}.parquet"
    else:
        log.info(f"Fetching {len(tickers)} tickers | Period : {period}")
        file_name = f"Prices_period_{period}_{file_timestamp}.parquet"

    data = _download(tickers, start_date, end_date, period)

    if data is None or data.empty:
        log.error(f"No data returned for tickers : {tickers}")
        raise ValueError("No data returned from API!")

    data = data.stack(level=0).rename_axis(["Date", "Ticker"]).reset_index(level=1).reset_index()

    file_path = get_storage_path() / file_name
    data.to_parquet(file_path, compression="zstd", engine="pyarrow", index=False)
    log.info(f"Successfully saved {len(data):,} rows to {file_path}")

    if upload:
        upload_to_r2(file_path)

    return file_path


# ------------------------- UPLOAD DATA TO R2 -----------------------------
def get_r2_client() -> S3Client:
    load_dotenv()

    required_vars = ["R2_ENDPOINT", "R2_KEY", "R2_SECRET"]
    missing = [k for k in required_vars if not os.getenv(k)]
    if missing:
        raise OSError(f"Missing required env vars: {missing}")

    return boto3.client(
        "s3",
        endpoint_url=os.environ["R2_ENDPOINT"],
        aws_access_key_id=os.environ["R2_KEY"],
        aws_secret_access_key=os.environ["R2_SECRET"],
    )


def upload_to_r2(local_file_path: Path, bucket: str = "stockpulse") -> str:
    client = get_r2_client()
    key = f"raw/{local_file_path.name}"

    # Avoid billiing
    # Check 1: Set storage class default to STANDARD
    extra_args = {"StorageClass": "STANDARD"}

    # Check 2: Disable multipart load - boto3 by default splits file > 8MB into parts
    # resulting in many Class A ops
    transfer_config = TransferConfig(
        multipart_threshold=500 * 1024 * 1024,  # Split after 500 MB
        use_threads=False,
    )

    # Check 3: Avoid uploading if file already present
    # 1 Class B op to check if file exists -> Avoids expensive Class A op of re-upload
    try:
        client.head_object(Bucket=bucket, Key=key)
        # if we reach here, object exists — filename match is enough
        log.info(f"Skipping upload — {key} already exists in R2")
        return key

    except ClientError as e:
        error_code = e.response.get("Error", {}).get("Code")
        if error_code not in ("404", "403", "NoSuchKey"):
            raise

    client.upload_file(str(local_file_path), bucket, key, ExtraArgs=extra_args, Config=transfer_config)

    log.info(f"Uploaded to R2: s3://{bucket}/{key}")

    return key


if __name__ == "__main__":
    config_path = Path(__file__).resolve().parent.parent / "config.toml"
    log.info(f"Reading config : {config_path}")
    with open(config_path, "rb") as f:
        config = tomllib.load(f)

    download_data(
        tickers=config["finance_api_params"]["tickers"],
        period=config["finance_api_params"].get("period", "1y"),
        upload=True,
    )
