import yfinance as yf
from datetime import datetime
from pathlib import Path


def get_storage_path() -> Path:
    BASE_DIR = Path(__file__).resolve().parent.parent
    data_dir = BASE_DIR / "data" / "raw"
    data_dir.mkdir(parents=True, exist_ok=True)
    return data_dir


def download_data(
    tickers: list,
    start_date: datetime | None = None,
    end_date: datetime | None = None,
    period: str | None = "1y",
):
    FILE_TIMESTAMP = datetime.now().strftime("%y%m%d")
    try:
        if period is None:
            if start_date is None:
                start_date = datetime.now().replace(month=1, day=1)

            if end_date is None:
                end_date = datetime.now()
            print(
                f"Fetching data for tickers {tickers} from {start_date:%Y-%m-%d} to {end_date:%Y-%m-%d}"
            )
            data = yf.download(
                tickers=tickers, start=start_date, end=end_date, group_by="Ticker"
            )
            FILE_NAME = f"Tickers_{start_date:%Y%m%d}_{end_date:%Y%m%d}_{FILE_TIMESTAMP}.parquet"
        else:
            print(f"Fetching data for tickers {tickers} for period of {period}")
            data = yf.download(tickers=tickers, period=period, group_by="Ticker")
            FILE_NAME = f"Tickers_period_{period}_{FILE_TIMESTAMP}.parquet"

        if data is not None and not data.empty:
            data = (
                data.stack(level=0).rename_axis(["Date", "Ticker"]).reset_index(level=1)
            )
            FILE_PATH = get_storage_path() / FILE_NAME
            data.to_parquet(FILE_PATH, compression="zstd", engine="pyarrow")
            print(f"Successfully saved to {FILE_PATH}")
        else:
            print(f"Couldn't find data for {tickers}")
    except Exception as e:
        print(f"Error downloading tickers {tickers} data : {e}")


if __name__ == "__main__":
    download_data(
        [
            "AAPL",
            "MSFT",
            "GOOGL",
            "^GSPC",
            "^IXIC",
            "SPY",
            "BTC-USD",
            "GLD",
            "TSLA",
            "NVDA",
        ]
    )
