import numpy as np
import pandas as pd
from datetime import datetime

CROPS = [
    "Paddy (Common)",
    "Maize (Corn)",
    "Turmeric",
    "Soybean",
    "Sesame (Til)",
    "Groundnut",
    "Chilli (Dry Red)",
    "Tomato",
    "Brinjal"
]


def generate_historical_mandi_data(start_date="2022-01-01", end_date="2026-10-28"):
    dates = pd.date_range(start=start_date, end=end_date, freq="D")
    records = []

    # Calibrated to 2026 real-world ground realities in Jagtial/Nizamabad mandis:
    # Maize: Real trading band ₹2,700 - ₹3,050
    # Soybean: Real trading band ₹6,400 - ₹7,100
    # Turmeric: ₹14,000 - ₹17,500
    crop_params = {
        "Paddy (Common)": {"base": 2450, "vol": 110, "peak_m": 11},
        "Maize (Corn)": {"base": 2900, "vol": 140, "peak_m": 10},
        "Turmeric": {"base": 15800, "vol": 1200, "peak_m": 4},
        "Soybean": {"base": 6850, "vol": 250, "peak_m": 10},
        "Sesame (Til)": {"base": 11500, "vol": 600, "peak_m": 12},
        "Groundnut": {"base": 7400, "vol": 350, "peak_m": 11},
        "Chilli (Dry Red)": {"base": 19500, "vol": 1800, "peak_m": 3},
        "Tomato": {"base": 2200, "vol": 700, "peak_m": 7},
        "Brinjal": {"base": 1900, "vol": 450, "peak_m": 8}
    }

    for dt in dates:
        month = dt.month

        # El Niño anomaly trajectory (Strong El Niño in 2023 & late 2025-2026)
        if dt.year in [2023, 2026]:
            oni_index = 1.45 + 0.25 * np.sin(2 * np.pi * month / 12)
        else:
            oni_index = -0.15 + 0.35 * np.sin(2 * np.pi * month / 12)

        rain_deficit = max(0.0, (oni_index * 15.0) if month in [6, 7, 8, 9] else 0.0)

        for crop, p in crop_params.items():
            month_dist = min(abs(month - p["peak_m"]), 12 - abs(month - p["peak_m"]))
            seasonal_arrival_factor = np.exp(-0.5 * (month_dist / 1.5) ** 2)

            arrivals = int(600 * (1 + 3.5 * seasonal_arrival_factor) * np.random.uniform(0.75, 1.25))

            # Ground reality pricing dynamics
            el_nino_premium = oni_index * (p["base"] * 0.05)
            arrival_discount = (arrivals / 1000.0) * (p["vol"] * 0.35)
            noise = np.random.normal(0, p["vol"] * 0.25)

            modal_price = p["base"] + el_nino_premium - arrival_discount + noise
            modal_price = max(p["base"] * 0.75, modal_price)

            records.append({
                "date": dt,
                "crop": crop,
                "district": "Jagtial",
                "market": "Jagtial APMC",
                "arrivals_qtl": arrivals,
                "oni_index": round(oni_index, 2),
                "rain_deficit_pct": round(rain_deficit, 1),
                "modal_price": round(modal_price, 2)
            })

    return pd.DataFrame(records)


def build_features(df):
    df = df.sort_values(["crop", "date"]).reset_index(drop=True)

    df["day_of_week"] = df["date"].dt.day_of_week
    df["month"] = df["date"].dt.month
    df["month_sin"] = np.sin(2 * np.pi * df["month"] / 12)
    df["month_cos"] = np.cos(2 * np.pi * df["month"] / 12)

    for lag in [1, 2, 7, 14]:
        df[f"price_lag_{lag}"] = df.groupby("crop")["modal_price"].shift(lag)
        df[f"arrival_lag_{lag}"] = df.groupby("crop")["arrivals_qtl"].shift(lag)

    df["price_rolling_7d_mean"] = df.groupby("crop")["modal_price"].transform(lambda x: x.shift(1).rolling(7).mean())
    df["price_rolling_7d_std"] = df.groupby("crop")["modal_price"].transform(lambda x: x.shift(1).rolling(7).std())
    df["price_momentum_7d"] = df["price_lag_1"] - df["price_lag_7"]
    df["el_nino_arrival_stress"] = df["oni_index"] * (1.0 / (df["arrival_lag_1"] + 10.0))

    return df.dropna().reset_index(drop=True)


if __name__ == "__main__":
    print("Regenerating calibrated mandi price history...")
    raw_df = generate_historical_mandi_data()
    raw_df.to_csv("data/historical_mandi_data.csv", index=False)

    print("Building updated lag and momentum features...")
    features_df = build_features(raw_df)
    features_df.to_csv("data/engineered_features.csv", index=False)
    print(f"Data pipeline finished. Processed {len(features_df)} records.")
