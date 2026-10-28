import os
import joblib
import pandas as pd
import lightgbm as lgb
from sklearn.metrics import mean_absolute_error

FEATURE_COLS = [
    "arrivals_qtl", "oni_index", "rain_deficit_pct",
    "month_sin", "month_cos", "day_of_week",
    "price_lag_1", "price_lag_2", "price_lag_7", "price_lag_14",
    "arrival_lag_1", "arrival_lag_7",
    "price_rolling_7d_mean", "price_rolling_7d_std", "price_momentum_7d",
    "el_nino_arrival_stress"
]


def train_crop_models():
    os.makedirs("models", exist_ok=True)
    df = pd.read_csv("data/engineered_features.csv")
    df["date"] = pd.to_datetime(df["date"])

    # Temporal Train/Test split: Train on 2022-2025, evaluate on 2026 data
    train_mask = df["date"] < "2026-01-01"
    crops = df["crop"].unique()

    summary = {}

    for crop in crops:
        print(f"\nTraining Multi-Quantile Engine for: {crop}")
        crop_df = df[df["crop"] == crop]

        X_train = crop_df[train_mask][FEATURE_COLS]
        y_train = crop_df[train_mask]["modal_price"]

        X_test = crop_df[~train_mask][FEATURE_COLS]
        y_test = crop_df[~train_mask]["modal_price"]

        crop_models = {}
        for alpha in [0.10, 0.50, 0.90]:
            params = {
                "objective": "quantile",
                "alpha": alpha,
                "metric": "quantile",
                "learning_rate": 0.04,
                "num_leaves": 31,
                "min_child_samples": 20,
                "verbosity": -1,
                "random_state": 42
            }
            dtrain = lgb.Dataset(X_train, label=y_train)
            model = lgb.train(params, dtrain, num_boost_round=250)
            crop_models[f"p{int(alpha * 100)}"] = model

        # Evaluate P50 (Expected Equilibrium Price)
        p50_preds = crop_models["p50"].predict(X_test)
        mae = mean_absolute_error(y_test, p50_preds)
        print(f"-> [{crop}] Test MAE: ₹{mae:.2f}/quintal")

        # Save bundle per crop
        safe_name = crop.replace(" ", "_").replace("(", "").replace(")", "")
        joblib.dump(crop_models, f"models/{safe_name}_models.pkl")
        summary[crop] = mae

    print("\nTraining completed successfully for all 9 crops.")
    return summary


if __name__ == "__main__":
    train_crop_models()
