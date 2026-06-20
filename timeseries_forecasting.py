import numpy as np
import pandas as pd
from pathlib import Path
import matplotlib.pyplot as plt

from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


class TimeSeriesForecaster:
    def __init__(self, output_dir="outputs/timeseries"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.model = None

    def create_sample_series(self, periods=1000):
        np.random.seed(42)

        time_index = pd.date_range(
            start="2025-01-01",
            periods=periods,
            freq="h"
        )

        trend = np.linspace(10, 25, periods)
        daily_pattern = 5 * np.sin(2 * np.pi * np.arange(periods) / 24)
        weekly_pattern = 3 * np.sin(2 * np.pi * np.arange(periods) / (24 * 7))
        noise = np.random.normal(0, 1.5, periods)

        value = trend + daily_pattern + weekly_pattern + noise

        df = pd.DataFrame({
            "timestamp": time_index,
            "value": value
        })

        return df

    def create_features(self, df, max_lag=24):
        data = df.copy()

        data["hour"] = data["timestamp"].dt.hour
        data["dayofweek"] = data["timestamp"].dt.dayofweek
        data["month"] = data["timestamp"].dt.month

        data["hour_sin"] = np.sin(2 * np.pi * data["hour"] / 24)
        data["hour_cos"] = np.cos(2 * np.pi * data["hour"] / 24)

        for lag in range(1, max_lag + 1):
            data[f"lag_{lag}"] = data["value"].shift(lag)

        for window in [3, 6, 12, 24]:
            data[f"rolling_mean_{window}"] = data["value"].shift(1).rolling(window).mean()
            data[f"rolling_std_{window}"] = data["value"].shift(1).rolling(window).std()
            data[f"rolling_min_{window}"] = data["value"].shift(1).rolling(window).min()
            data[f"rolling_max_{window}"] = data["value"].shift(1).rolling(window).max()

        data["target"] = data["value"].shift(-1)
        data = data.dropna().reset_index(drop=True)

        return data

    def split_data(self, data):
        split_index = int(len(data) * 0.8)

        train = data.iloc[:split_index]
        test = data.iloc[split_index:]

        feature_columns = [
            col for col in data.columns
            if col not in ["timestamp", "value", "target"]
        ]

        X_train = train[feature_columns]
        y_train = train["target"]

        X_test = test[feature_columns]
        y_test = test["target"]

        return X_train, X_test, y_train, y_test, test

    def train(self, X_train, y_train):
        self.model = HistGradientBoostingRegressor(
            max_iter=300,
            learning_rate=0.04,
            max_leaf_nodes=31,
            l2_regularization=0.1,
            random_state=42
        )

        self.model.fit(X_train, y_train)

    def evaluate(self, X_test, y_test, test_df):
        pred = self.model.predict(X_test)

        mae = mean_absolute_error(y_test, pred)
        rmse = mean_squared_error(y_test, pred) ** 0.5
        r2 = r2_score(y_test, pred)

        result = pd.DataFrame({
            "timestamp": test_df["timestamp"].values,
            "actual": y_test.values,
            "predicted": pred,
            "error": y_test.values - pred
        })

        result.to_csv(
            self.output_dir / "forecast_result.csv",
            index=False,
            encoding="utf-8-sig"
        )

        metrics = pd.DataFrame([{
            "mae": mae,
            "rmse": rmse,
            "r2": r2
        }])

        metrics.to_csv(
            self.output_dir / "forecast_metrics.csv",
            index=False
        )

        print("MAE:", round(mae, 4))
        print("RMSE:", round(rmse, 4))
        print("R2:", round(r2, 4))

        self.plot_result(result)

    def plot_result(self, result):
        plt.figure(figsize=(12, 5))
        plt.plot(result["timestamp"], result["actual"], label="Actual")
        plt.plot(result["timestamp"], result["predicted"], label="Predicted")
        plt.title("Time Series Forecasting Result")
        plt.xlabel("Timestamp")
        plt.ylabel("Value")
        plt.legend()
        plt.tight_layout()
        plt.savefig(self.output_dir / "forecast_plot.png", dpi=150)
        plt.close()

    def run(self):
        df = self.create_sample_series()
        data = self.create_features(df)
        X_train, X_test, y_train, y_test, test_df = self.split_data(data)

        self.train(X_train, y_train)
        self.evaluate(X_test, y_test, test_df)

        print("시계열 예측 완료")


if __name__ == "__main__":
    forecaster = TimeSeriesForecaster()
    forecaster.run()
