"""
plot_btc_price_volatility_context.py
====================================

Creates a publication-style BTC closing price and 30-day rolling-volatility context figure.
"""

from __future__ import annotations

from pathlib import Path
import argparse
import ast
import logging
from typing import Optional

import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib.ticker import FuncFormatter


SCRIPT_PATH = Path(__file__).resolve()


def candidate_roots() -> list[Path]:
    """Return candidate filesystem roots to search for the project directory."""
    starts = [SCRIPT_PATH.parent, Path.cwd()]
    roots: list[Path] = []
    for start in starts:
        roots.append(start)
        roots.extend(list(start.parents))

    out, seen = [], set()
    for root in roots:
        if root not in seen:
            out.append(root)
            seen.add(root)
    return out


def find_project_root() -> Path:
    """Locate the repository root by checking for expected project folders."""
    for root in candidate_roots():
        # Prefer a root that contains both the data and results folders.
        if (root / "data").exists() and (root / "results").exists():
            return root

    for root in candidate_roots():
        if root.name == "btc_llm_forecasting":
            return root

    for root in candidate_roots():
        if (root / "results").exists() and root.name != "scripts":
            return root

    return Path.cwd()


PROJECT_ROOT = find_project_root()
OUTPUT_DIR = PROJECT_ROOT / "figures" / "results"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

EXPORT_FORMATS = ("png", "pdf", "svg")
PNG_DPI = 500

logger = logging.getLogger(__name__)

FIGURE_TITLE_SIZE = 22
PANEL_TITLE_SIZE = 19
AXIS_LABEL_SIZE = 17
TICK_LABEL_SIZE = 17
CAPTION_NOTE_SIZE = 15

PALETTE = {
    "blue": "#0072B2",
    "orange": "#E69F00",
    "grid": "#999999",
}


def apply_publication_style() -> None:
    """Apply publication-ready Matplotlib styling for fonts, grids, and figure aesthetics."""
    plt.rcParams.update({
        "font.family": "serif",
        "font.serif": ["Times New Roman", "Times", "DejaVu Serif"],
        "mathtext.fontset": "dejavuserif",
        "font.size": 17.0,
        "axes.titlesize": 17.0,
        "axes.titleweight": "bold",
        "axes.labelsize": AXIS_LABEL_SIZE,
        "xtick.labelsize": TICK_LABEL_SIZE,
        "ytick.labelsize": TICK_LABEL_SIZE,
        "figure.titlesize": FIGURE_TITLE_SIZE,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.linewidth": 1.0,
        "grid.linewidth": 0.8,
        "grid.alpha": 0.30,
        "figure.facecolor": "white",
        "axes.facecolor": "white",
        "savefig.facecolor": "white",
        "savefig.edgecolor": "white",
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
    })


def flatten_columns(columns) -> list[str]:
    """Normalize multi-index or stringified tuple column labels into flat strings."""
    def normalize_col(col) -> str:
        # Handle actual tuples and stringified tuple-like column labels.
        if isinstance(col, tuple):
            parts = [str(part) for part in col if str(part) not in ("", "None")]
            return "_".join(parts)

        if isinstance(col, str):
            try:
                parsed = ast.literal_eval(col)
                if isinstance(parsed, tuple):
                    parts = [str(part) for part in parsed if str(part) not in ("", "None")]
                    return "_".join(parts)
            except (ValueError, SyntaxError):
                pass

        return str(col)

    return [normalize_col(c) for c in columns]


def normalise_market_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Standardize market data columns and return a clean Date/Close DataFrame."""
    df = df.copy()
    df.columns = flatten_columns(df.columns)

    lower_map = {str(c).lower(): c for c in df.columns}

    date_col = None
    for cand in ["date", "datetime", "timestamp"]:
        if cand in lower_map:
            date_col = lower_map[cand]
            break

    close_col = None
    for cand in ["close", "adj close", "adj_close"]:
        if cand in lower_map:
            close_col = lower_map[cand]
            break

    if close_col is None:
        for col in df.columns:
            if str(col).lower().startswith("close"):
                close_col = col
                break

    if date_col is None:
        if isinstance(df.index, pd.DatetimeIndex):
            df = df.reset_index()
            date_col = df.columns[0]
        else:
            raise KeyError("Could not find a Date column or DatetimeIndex.")

    if close_col is None:
        raise KeyError(f"Could not find a Close column. Columns found: {list(df.columns)}")

    out = pd.DataFrame({
        "Date": pd.to_datetime(df[date_col]),
        "Close": pd.to_numeric(df[close_col], errors="coerce"),
    })

    return out.dropna(subset=["Date", "Close"]).sort_values("Date").reset_index(drop=True)


def load_project_market_data() -> pd.DataFrame | None:
    """Attempt to load market data from local project parquet files."""
    candidates = [
        PROJECT_ROOT / "data" / "raw" / "btc_market_data.parquet",
        PROJECT_ROOT / "data" / "processed" / "btc_market_trends_merged.parquet",
        PROJECT_ROOT / "data" / "processed" / "btc_features.parquet",
    ]

    for path in candidates:
        if not path.exists():
            continue

        print(f"Attempting project market data: {path}")
        try:
            df = pd.read_parquet(path)
            normalized = normalise_market_columns(df)
            print(f"Loaded project market data from: {path}")
            return normalized
        except Exception as exc:
            print(f"Warning: failed to load {path}: {exc}")
            continue

    return None


def load_yfinance_market_data() -> pd.DataFrame:
    """Fetch BTC-USD daily market data from yfinance as a fallback data source."""
    try:
        import yfinance as yf
    except ImportError as exc:
        raise ImportError(
            "Project data was not found and yfinance is not installed. "
            "Install it with: pip install yfinance"
        ) from exc

    start = "2025-01-01"
    end = pd.Timestamp.today().strftime("%Y-%m-%d")

    print(f"Project market data not found. Fetching BTC-USD from yfinance: {start} to {end}")
    raw = yf.download("BTC-USD", start=start, end=end, auto_adjust=False, progress=False)

    if raw.empty:
        raise ValueError("yfinance returned no BTC-USD data.")

    return normalise_market_columns(raw.reset_index())


def load_market_data() -> pd.DataFrame:
    """Load market data from project files or fall back to yfinance if needed."""
    df = load_project_market_data()
    return df if df is not None else load_yfinance_market_data()


def currency_formatter(x: float, _pos: int) -> str:
    """Format the y-axis tick labels as USD values with k-scale notation."""
    return f"${x / 1000:.0f}k" if abs(x) >= 1000 else f"${x:.0f}"


def percent_formatter(x: float, _pos: int) -> str:
    """Format the y-axis tick labels as percentages for volatility values."""
    return f"{x:.1%}"


def save_figure(fig: plt.Figure, stem: str, output_dir: Path = OUTPUT_DIR) -> None:
    """Save the generated figure in PNG, PDF, and SVG formats to the output directory."""
    output_dir.mkdir(parents=True, exist_ok=True)

    for fmt in EXPORT_FORMATS:
        out = output_dir / f"{stem}.{fmt}"
        if out.exists():
            out.unlink()

        kwargs = {"bbox_inches": "tight", "facecolor": "white"}
        if fmt == "png":
            kwargs["dpi"] = PNG_DPI

        fig.savefig(out, **kwargs)
        logger.info("Saved: %s", out)


def format_date_axis(ax: plt.Axes) -> None:
    """Configure the x-axis to show quarterly tick labels and rotated date text."""
    ax.xaxis.set_major_locator(mdates.MonthLocator(interval=3))
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%b %Y"))
    ax.tick_params(axis="x", labelbottom=True, rotation=35, labelsize=TICK_LABEL_SIZE)
    ax.tick_params(axis="y", labelsize=TICK_LABEL_SIZE)


def plot_btc_context(df: pd.DataFrame, output_dir: Path) -> None:
    """Create the BTC closing price and 30-day rolling volatility figure."""
    df = df.copy()
    df["Return"] = df["Close"].pct_change()
    df["Rolling_Volatility_30d"] = df["Return"].rolling(30).std()

    fig, axes = plt.subplots(
        2,
        1,
        figsize=(12.5, 8.8),
        constrained_layout=False,
        sharex=False,
    )

    ax_price, ax_vol = axes

    ax_price.plot(df["Date"], df["Close"], linewidth=2.4, color=PALETTE["blue"])
    ax_price.set_title("BTC Closing Price Over the Study Period", pad=12)
    ax_price.set_ylabel("Closing Price (USD)")
    ax_price.set_xlabel("Date", labelpad=12)
    ax_price.yaxis.set_major_formatter(FuncFormatter(currency_formatter))
    ax_price.grid(axis="y", linestyle="--", color=PALETTE["grid"])
    ax_price.set_axisbelow(True)
    format_date_axis(ax_price)

    ax_vol.plot(df["Date"], df["Rolling_Volatility_30d"], linewidth=2.4, color=PALETTE["orange"])
    ax_vol.set_title("30-Day Rolling Return Volatility", pad=18)
    ax_vol.set_ylabel("Rolling Volatility")
    ax_vol.set_xlabel("Date", labelpad=12)
    ax_vol.yaxis.set_major_formatter(FuncFormatter(percent_formatter))
    ax_vol.grid(axis="y", linestyle="--", color=PALETTE["grid"])
    ax_vol.set_axisbelow(True)
    format_date_axis(ax_vol)

    fig.suptitle(
        "BTC Price and Volatility Context",
        fontsize=FIGURE_TITLE_SIZE,
        fontweight="bold",
        y=0.985,
    )

    fig.text(
        0.5,
        0.006,
        "Rolling volatility is computed as the 30-day rolling standard deviation of daily BTC returns.",
        ha="center",
        fontsize=16.0,
        color="#333333",
    )

    fig.subplots_adjust(
        left=0.10,
        right=0.98,
        top=0.88,
        bottom=0.17,
        hspace=0.73,
    )

    save_figure(fig, "btc_price_volatility_context", args.output_dir)
    plt.close(fig)


def parse_args(argv: Optional[list[str]] = None) -> argparse.Namespace:
    """Parse command-line arguments for generating and saving the BTC context figure."""
    parser = argparse.ArgumentParser(
        prog="plot_btc_price_volatility_context",
        description="Generate a BTC price and volatility context figure for the project.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=OUTPUT_DIR,
        help="Folder where generated figure files will be saved",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable debug logging",
    )
    return parser.parse_args(argv)


def main(args: argparse.Namespace) -> None:
    """Load market data, apply styling, render the chart, and save output files."""
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
    )

    apply_publication_style()
    logger.info("Project root: %s", PROJECT_ROOT)
    logger.info("Output directory: %s", args.output_dir)

    df = load_market_data()
    plot_btc_context(df, args.output_dir)

    logger.info("BTC context figure generated successfully.")


if __name__ == "__main__":
    main(parse_args())