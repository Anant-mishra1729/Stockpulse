import logging
import tomllib
from datetime import datetime
from pathlib import Path

import yfinance as yf
from pandas import DataFrame
from tenacity import retry, stop_after_attempt, wait_exponential

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
log = logging.getLogger(__name__)


@retry(stop=stop_after_attempt(3), wait=wait_exponential(min=2, max=10))
def _download(
    tickers: list[str],
    start_date: datetime | None,
    end_date: datetime | None,
    period: str | None,
) -> DataFrame | None:
    if period:
        return yf.download(
            tickers=tickers, period=period, group_by="Ticker", progress=False
        )
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
) -> Path:
    file_timestamp = datetime.now().strftime("%y%m%d")
    if period is None:
        start_date = start_date or datetime.now().replace(month=1, day=1)
        end_date = end_date or datetime.now()
        log.info(
            f"Fetching {len(tickers)} tickers | "
            "Period : {start_date:%Y-%m-%d} to {end_date:%Y-%m-%d}"
        )
        file_name = (
            f"Prices_{start_date:%Y%m%d}_{end_date:%Y%m%d}_{file_timestamp}.parquet"
        )
    else:
        log.info(f"Fetching {len(tickers)} tickers | Period : {period}")
        file_name = f"Prices_period_{period}_{file_timestamp}.parquet"

    data = _download(tickers, start_date, end_date, period)

    if data is None or data.empty:
        log.error(f"No data returned for tickers : {tickers}")
        raise ValueError("No data returned from API!")

    data = (
        data.stack(level=0)
        .rename_axis(["Date", "Ticker"])
        .reset_index(level=1)
        .reset_index()
    )

    file_path = get_storage_path() / file_name
    data.to_parquet(file_path, compression="zstd", engine="pyarrow", index=False)
    log.info(f"Successfully saved {len(data):,} rows to {file_path}")
    return file_path


if __name__ == "__main__":
    config_path = Path(__file__).resolve().parent.parent / "config.toml"
    log.info(f"Reading config : {config_path}")
    with open(config_path, "rb") as f:
        config = tomllib.load(f)

    download_data(
        tickers=config["finance_api_params"]["tickers"],
        period=config["finance_api_params"].get("period", "1y"),
    )
