"""以 RFE 與多元線性迴歸分析 50 Startups 資料集。"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import requests
from sklearn.feature_selection import RFE
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler


BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
OUTPUT_DIR = BASE_DIR / "outputs"
DATA_PATH = DATA_DIR / "50_Startups.csv"
DATA_URL = (
    "https://gist.githubusercontent.com/GaneshSparkz/"
    "b5662effbdae8746f7f7d8ed70c42b2d/raw/"
    "faf8b1a0d58e251f48a647d3881e7a960c3f0925/50_Startups.csv"
)
NUMERIC_COLUMNS = ["R&D Spend", "Administration", "Marketing Spend"]


def download_data() -> None:
    """資料不存在時下載 CSV。"""
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    if DATA_PATH.exists():
        print(f"資料已存在：{DATA_PATH}")
        return

    print(f"正在下載資料：{DATA_URL}")
    response = requests.get(DATA_URL, timeout=30)
    response.raise_for_status()
    DATA_PATH.write_bytes(response.content)
    print(f"已下載資料至：{DATA_PATH}")


def load_and_explore() -> pd.DataFrame:
    """載入資料並列印資料概況、型態、缺失值與統計摘要。"""
    data = pd.read_csv(DATA_PATH)
    expected_columns = {*NUMERIC_COLUMNS, "State", "Profit"}
    missing_columns = expected_columns.difference(data.columns)
    if missing_columns:
        raise ValueError(f"資料缺少必要欄位：{sorted(missing_columns)}")

    print("\n=== 資料概況 ===")
    print(f"資料筆數：{len(data)}；欄位數：{len(data.columns)}")
    print(f"欄位：{data.columns.tolist()}")
    print("\n資料型態：")
    print(data.dtypes.to_string())
    print("\n缺失值：")
    print(data.isna().sum().to_string())
    print("\n數值欄位描述統計：")
    print(data.describe().to_string())
    print("\nState 分布：")
    print(data["State"].value_counts().to_string())
    return data


def preprocess(
    data: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series, list[str]]:
    """先切分資料，再只用訓練集 fit One-Hot Encoder。"""
    if data[[*NUMERIC_COLUMNS, "State", "Profit"]].isna().any().any():
        raise ValueError("資料含缺失值，請先決定合適的補值策略。")

    features = data[[*NUMERIC_COLUMNS, "State"]].copy()
    target = pd.to_numeric(data["Profit"], errors="raise").astype(float)
    for column in NUMERIC_COLUMNS:
        features[column] = pd.to_numeric(features[column], errors="raise").astype(float)
    features["State"] = features["State"].astype(str)

    X_train, X_test, y_train, y_test = train_test_split(
        features, target, test_size=0.2, random_state=42
    )
    encoder = OneHotEncoder(
        drop="first", handle_unknown="ignore", sparse_output=False
    )
    encoder.fit(X_train[["State"]])

    dummy_columns = encoder.get_feature_names_out(["State"]).tolist()
    feature_names = [*NUMERIC_COLUMNS, *dummy_columns]

    def encode(frame: pd.DataFrame) -> pd.DataFrame:
        encoded_states = encoder.transform(frame[["State"]])
        state_frame = pd.DataFrame(encoded_states, columns=dummy_columns)
        numeric_frame = frame[NUMERIC_COLUMNS].reset_index(drop=True)
        return pd.concat([numeric_frame, state_frame], axis=1).astype(float)

    X_train_encoded = encode(X_train)
    X_test_encoded = encode(X_test)
    print("\n=== 前處理 ===")
    print(f"訓練集：{len(X_train_encoded)} 筆；測試集：{len(X_test_encoded)} 筆")
    print(f"State dummy variables：{dummy_columns}")
    print("California 為 drop='first' 的基準類別；模型候選欄位：")
    print(feature_names)
    return (
        X_train_encoded,
        X_test_encoded,
        y_train.reset_index(drop=True),
        y_test.reset_index(drop=True),
        feature_names,
    )


def calculate_metrics(actual: pd.Series, predicted: np.ndarray) -> dict[str, float]:
    mse = mean_squared_error(actual, predicted)
    return {
        "MSE": float(mse),
        "RMSE": float(np.sqrt(mse)),
        "MAE": float(mean_absolute_error(actual, predicted)),
        "R-squared": float(r2_score(actual, predicted)),
    }


def evaluate_feature_counts(
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
    y_train: pd.Series,
    y_test: pd.Series,
    feature_names: list[str],
) -> tuple[pd.DataFrame, dict[str, list[dict[str, object]]]]:
    """僅在訓練資料執行 RFE，對固定測試集比較 1 至 5 個特徵。"""
    if len(feature_names) < 5:
        raise ValueError(f"RFE 需要至少 5 個候選特徵，目前只有 {len(feature_names)}。")

    selection_scaler = StandardScaler().fit(X_train)
    X_train_scaled = selection_scaler.transform(X_train)
    rows = []
    model_details = []
    for feature_count in range(1, 6):
        selector = RFE(
            estimator=LinearRegression(), n_features_to_select=feature_count
        )
        selector.fit(X_train_scaled, y_train)
        selected = [
            name for name, keep in zip(feature_names, selector.support_) if keep
        ]
        model = LinearRegression().fit(X_train[selected], y_train)
        train_metrics = calculate_metrics(y_train, model.predict(X_train[selected]))
        test_predictions = model.predict(X_test[selected])
        test_metrics = calculate_metrics(y_test, test_predictions)

        rows.append(
            {
                "Number of Features": feature_count,
                "Selected Features": selected,
                "Train MSE": train_metrics["MSE"],
                "Train RMSE": train_metrics["RMSE"],
                "Train MAE": train_metrics["MAE"],
                "Train R-squared": train_metrics["R-squared"],
                "Test MSE": test_metrics["MSE"],
                "Test RMSE": test_metrics["RMSE"],
                "Test MAE": test_metrics["MAE"],
                "Test R-squared": test_metrics["R-squared"],
            }
        )
        model_details.append(
            {
                "number_of_features": feature_count,
                "selected_features": selected,
                "intercept": float(model.intercept_),
                "coefficients": {
                    name: float(coefficient)
                    for name, coefficient in zip(selected, model.coef_)
                },
                "test_rmse": test_metrics["RMSE"],
                "test_r_squared": test_metrics["R-squared"],
                "test_cases": [
                    {
                        "case": index + 1,
                        "actual": float(actual),
                        "predicted": float(predicted),
                        "residual": float(actual - predicted),
                    }
                    for index, (actual, predicted) in enumerate(
                        zip(y_test, test_predictions)
                    )
                ],
            }
        )
        print(
            f"\n特徵數 {feature_count}：{selected}\n"
            f"  訓練 RMSE={train_metrics['RMSE']:,.2f}，"
            f"測試 RMSE={test_metrics['RMSE']:,.2f}，"
            f"測試 R²={test_metrics['R-squared']:.4f}，"
            f"測試 MAE={test_metrics['MAE']:,.2f}"
        )

    return pd.DataFrame(rows), {"models": model_details}


def plot_feature_selection(results: pd.DataFrame) -> None:
    feature_counts = results["Number of Features"]
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.8))
    fig.patch.set_facecolor("#f7f7f2")
    for axis in axes:
        axis.set_facecolor("#ffffff")
        axis.spines[["top", "right"]].set_visible(False)
        axis.grid(axis="y", color="#e4e8e5", linewidth=0.8)
        axis.set_xticks(feature_counts)
        axis.set_xlabel("Number of Features")

    axes[0].plot(
        feature_counts,
        results["Test RMSE"],
        color="#2563eb",
        linewidth=2.5,
        marker="o",
        markersize=8,
    )
    axes[0].set_title("RMSE by Number of Features", loc="left", weight="bold")
    axes[0].set_ylabel("RMSE")

    axes[1].plot(
        feature_counts,
        results["Test R-squared"],
        color="#2563eb",
        linewidth=2.5,
        marker="o",
        markersize=8,
    )
    axes[1].set_title("R-squared by Number of Features", loc="left", weight="bold")
    axes[1].set_ylabel("R-squared")
    axes[1].set_ylim(min(-0.05, results["Test R-squared"].min() - 0.08), 1.05)

    fig.suptitle("50 Startups / RFE feature selection", x=0.08, ha="left", weight="bold")
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "feature_selection.png", dpi=170, bbox_inches="tight")
    plt.close(fig)


def plot_train_test_rmse(results: pd.DataFrame) -> None:
    feature_counts = results["Number of Features"]
    fig, axis = plt.subplots(figsize=(9, 4.8))
    fig.patch.set_facecolor("#f7f7f2")
    axis.set_facecolor("#ffffff")
    axis.plot(
        feature_counts,
        results["Train RMSE"],
        color="#198754",
        linewidth=2.5,
        marker="o",
        markersize=8,
        label="Training RMSE",
    )
    axis.plot(
        feature_counts,
        results["Test RMSE"],
        color="#e76f51",
        linewidth=2.5,
        marker="o",
        markersize=8,
        label="Testing RMSE",
    )
    axis.set_title("Training vs Testing RMSE", loc="left", weight="bold")
    axis.set_xlabel("Number of Features")
    axis.set_ylabel("RMSE")
    axis.set_xticks(feature_counts)
    axis.grid(axis="y", color="#e4e8e5", linewidth=0.8)
    axis.spines[["top", "right"]].set_visible(False)
    axis.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "train_test_rmse.png", dpi=170, bbox_inches="tight")
    plt.close(fig)


def print_overfit_notes(results: pd.DataFrame) -> None:
    best = results.loc[results["Test RMSE"].idxmin()]
    print("\n=== 訓練與測試差異 ===")
    print(
        f"測試 RMSE 最低的是 {int(best['Number of Features'])} 個特徵"
        f"（{best['Test RMSE']:,.2f}）。"
    )
    for _, row in results.iterrows():
        gap = row["Test RMSE"] - row["Train RMSE"]
        ratio = row["Test RMSE"] / row["Train RMSE"] if row["Train RMSE"] else np.inf
        if ratio > 1.5:
            note = "測試誤差明顯較高，需留意過度擬合或切分波動"
        elif row["Train RMSE"] > 15000 and row["Test RMSE"] > 15000:
            note = "訓練與測試誤差都偏高，可能欠擬合"
        else:
            note = "訓練與測試誤差差距有限"
        print(
            f"{int(row['Number of Features'])} 個特徵："
            f"Train RMSE {row['Train RMSE']:,.2f}；"
            f"Test RMSE {row['Test RMSE']:,.2f}；"
            f"差值 {gap:,.2f}。{note}。"
        )


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    download_data()
    data = load_and_explore()
    X_train, X_test, y_train, y_test, feature_names = preprocess(data)
    results, model_details = evaluate_feature_counts(
        X_train, X_test, y_train, y_test, feature_names
    )

    results.to_csv(
        OUTPUT_DIR / "feature_selection_results.csv",
        index=False,
        encoding="utf-8-sig",
        float_format="%.8f",
    )
    with (OUTPUT_DIR / "model_details.json").open("w", encoding="utf-8") as output_file:
        json.dump(model_details, output_file, ensure_ascii=False, indent=2)
    plot_feature_selection(results)
    plot_train_test_rmse(results)
    print_overfit_notes(results)

    display = results[
        ["Number of Features", "Selected Features", "Test MSE", "Test RMSE", "Test MAE", "Test R-squared"]
    ].rename(
        columns={
            "Test MSE": "MSE",
            "Test RMSE": "RMSE",
            "Test MAE": "MAE",
            "Test R-squared": "R-squared",
        }
    )
    print("\n=== 測試集結果表 ===")
    print(display.to_string(index=False, formatters={
        "MSE": "{:,.2f}".format,
        "RMSE": "{:,.2f}".format,
        "MAE": "{:,.2f}".format,
        "R-squared": "{:.4f}".format,
    }))
    print(f"\n所有圖表與結果已儲存至：{OUTPUT_DIR}")


if __name__ == "__main__":
    main()