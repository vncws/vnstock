import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from vnstock import Vnstock


# ==============================
# CONFIG
# ==============================

SYMBOL = "VCB"
START_DATE = "2015-01-01"
END_DATE = "2025-12-31"


# ==============================
# INIT VNSTOCK
# ==============================

stock = Vnstock().stock(symbol=SYMBOL, source="VCI")


# ==============================
# PRICE DATA
# ==============================

print("Loading price data...")

price = stock.quote.history(
    start=START_DATE,
    end=END_DATE,
    interval="1D"
)

price["time"] = pd.to_datetime(price["time"])
price = price.set_index("time")

# resample quarterly price (mean)
price_q = price["close"].resample("Q").mean().reset_index()

price_q["year"] = price_q["time"].dt.year
price_q["quarter"] = price_q["time"].dt.quarter


# ==============================
# BALANCE SHEET
# ==============================

print("Loading balance sheet...")

bs = stock.finance.balance_sheet(period="year", lang="vi")

# normalize column names
bs.columns = [c.lower() for c in bs.columns]

year_col = [c for c in bs.columns if "năm" in c][0]
equity_col = [c for c in bs.columns if "vốn chủ" in c][0]

bs = bs[[year_col, equity_col]]

bs.columns = ["year", "equity"]

bs = bs.sort_values("year")


# ==============================
# SHARES OUTSTANDING
# ==============================

print("Loading company overview...")

info = stock.company.overview()

# detect share column automatically
share_key = [c for c in info.columns if "share" in c.lower()][0]

shares = float(info.iloc[0][share_key])

print("Shares outstanding:", shares)


# ==============================
# BVPS CALCULATION
# ==============================

bs["bvps"] = bs["equity"] / shares


# ==============================
# MERGE DATA
# ==============================

df = price_q.merge(bs, on="year", how="left")

# forward fill BVPS
df["bvps"] = df["bvps"].ffill()


# ==============================
# PB CALCULATION
# ==============================

# price from vnstock is in thousand VND
df["price_vnd"] = df["close"] * 1000

df["PB"] = df["price_vnd"] / df["bvps"]


# ==============================
# VALUATION STATS
# ==============================

pb_mean = df["PB"].mean()
pb_std = df["PB"].std()

pb_p25 = df["PB"].quantile(0.25)
pb_p75 = df["PB"].quantile(0.75)


print("PB mean:", round(pb_mean,2))
print("PB std:", round(pb_std,2))


# ==============================
# PLOT
# ==============================

plt.style.use("seaborn-v0_8-whitegrid")

fig, ax = plt.subplots(figsize=(14,7))

# PB line
ax.plot(
    df["time"],
    df["PB"],
    color="#1f3c88",
    linewidth=2.5,
    label="VCB P/B"
)

# mean
ax.axhline(
    pb_mean,
    linestyle="--",
    linewidth=2,
    color="black",
    label="Mean"
)

# std bands
ax.axhline(
    pb_mean + pb_std,
    linestyle=":",
    linewidth=2,
    color="#d62728",
    label="+1σ"
)

ax.axhline(
    pb_mean - pb_std,
    linestyle=":",
    linewidth=2,
    color="#2ca02c",
    label="-1σ"
)

# percentile band shading
ax.fill_between(
    df["time"],
    pb_p25,
    pb_p75,
    color="#ffcc00",
    alpha=0.15,
    label="25–75 percentile"
)

# highlight current PB
current_pb = df["PB"].iloc[-1]

ax.scatter(
    df["time"].iloc[-1],
    current_pb,
    color="red",
    s=80,
    zorder=5
)

ax.text(
    df["time"].iloc[-1],
    current_pb + 0.05,
    f"{current_pb:.2f}x",
    fontsize=11
)

# titles
ax.set_title(
    "VCB Valuation – P/B (Quarterly, 10Y)",
    fontsize=16,
    weight="bold"
)

ax.set_ylabel("P/B")
ax.set_xlabel("")

ax.legend(frameon=False)

plt.tight_layout()

plt.savefig("vcb_pb_10y_quarterly.png", dpi=300)