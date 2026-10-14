"""以 RFE 與多元線性迴歸分析 50 Startups 資料集。"""

from __future__ import annotations

import json
from itertools import combinations
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import requests
from sklearn.feature_selection import RFE, SequentialFeatureSelector
from sklearn.linear_model import ElasticNetCV, LassoCV, LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import KFold, cross_val_score, train_test_split
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
SELECTION_METHODS = {
    "RFE": "RFE",
    "Lasso": "Lasso L1 排名",
    "Forward SFS": "Forward SFS",
    "Elastic Net": "Elastic Net",
    "Exhaustive Search": "Exhaustive Search",
}


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


def select_features(
    method: str,
    X_fit: pd.DataFrame,
    y_fit: pd.Series,
    feature_count: int,
) -> list[str]:
    """只根據傳入訓練 fold 選特徵，內層 CV 不接觸外層驗證 fold。"""
    feature_names = X_fit.columns.tolist()
    if feature_count == len(feature_names):
        return feature_names

    inner_cv = KFold(n_splits=min(4, len(X_fit)), shuffle=True, random_state=42)
    if method == "RFE":
        scaler = StandardScaler().fit(X_fit)
        selector = RFE(
            estimator=LinearRegression(), n_features_to_select=feature_count
        ).fit(scaler.transform(X_fit), y_fit)
        return [name for name, keep in zip(feature_names, selector.support_) if keep]

    if method in {"Lasso", "Elastic Net"}:
        scaler = StandardScaler().fit(X_fit)
        X_scaled = scaler.transform(X_fit)
        if method == "Lasso":
            selector = LassoCV(cv=inner_cv, max_iter=50000, random_state=42).fit(
                X_scaled, y_fit
            )
        else:
            selector = ElasticNetCV(
                l1_ratio=[0.2, 0.5, 0.8, 0.95],
                cv=inner_cv,
                max_iter=50000,
            ).fit(X_scaled, y_fit)
        ranked_indices = np.argsort(np.abs(selector.coef_))[::-1][:feature_count]
        selected_set = {feature_names[index] for index in ranked_indices}
        return [name for name in feature_names if name in selected_set]

    if method == "Forward SFS":
        selector = SequentialFeatureSelector(
            LinearRegression(),
            n_features_to_select=feature_count,
            direction="forward",
            scoring="neg_root_mean_squared_error",
            cv=inner_cv,
        ).fit(X_fit, y_fit)
        return [name for name, keep in zip(feature_names, selector.support_) if keep]

    if method == "Exhaustive Search":
        best_score = -np.inf
        best_features: tuple[str, ...] | None = None
        for candidate in combinations(feature_names, feature_count):
            score = cross_val_score(
                LinearRegression(),
                X_fit[list(candidate)],
                y_fit,
                scoring="neg_root_mean_squared_error",
                cv=inner_cv,
            ).mean()
            if score > best_score:
                best_score = score
                best_features = candidate
        if best_features is None:
            raise RuntimeError("Exhaustive Search 找不到有效特徵組合。")
        return list(best_features)

    raise ValueError(f"不支援的特徵選擇方法：{method}")


def evaluate_feature_counts(
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
    y_train: pd.Series,
    y_test: pd.Series,
    feature_names: list[str],
) -> tuple[
    pd.DataFrame,
    dict[str, list[dict[str, object]]],
    pd.DataFrame,
]:
    """巢狀 CV 比較選擇器；外層測試集只用於最終一次評估。"""
    if len(feature_names) < 5:
        raise ValueError(f"需要至少 5 個候選特徵，目前只有 {len(feature_names)}。")

    comparison_rows = []
    model_details = []
    outer_cv = KFold(n_splits=5, shuffle=True, random_state=42)
    for method, method_label in SELECTION_METHODS.items():
        for feature_count in range(1, 6):
            fold_rmse = []
            fold_r2 = []
            for fit_indices, validation_indices in outer_cv.split(X_train):
                X_fit = X_train.iloc[fit_indices]
                X_validation = X_train.iloc[validation_indices]
                y_fit = y_train.iloc[fit_indices]
                y_validation = y_train.iloc[validation_indices]
                fold_features = select_features(method, X_fit, y_fit, feature_count)
                fold_model = LinearRegression().fit(X_fit[fold_features], y_fit)
                fold_prediction = fold_model.predict(X_validation[fold_features])
                fold_metrics = calculate_metrics(y_validation, fold_prediction)
                fold_rmse.append(fold_metrics["RMSE"])
                fold_r2.append(fold_metrics["R-squared"])

            selected = select_features(method, X_train, y_train, feature_count)
            model = LinearRegression().fit(X_train[selected], y_train)
            train_metrics = calculate_metrics(y_train, model.predict(X_train[selected]))
            test_predictions = model.predict(X_test[selected])
            test_metrics = calculate_metrics(y_test, test_predictions)
            comparison_rows.append(
                {
                    "Method": method,
                    "Method Label": method_label,
                    "Number of Features": feature_count,
                    "Selected Features": selected,
                    "CV RMSE": float(np.mean(fold_rmse)),
                    "CV RMSE Std": float(np.std(fold_rmse, ddof=1)),
                    "CV R-squared": float(np.mean(fold_r2)),
                    "Train RMSE": train_metrics["RMSE"],
                    "Test MSE": test_metrics["MSE"],
                    "Test RMSE": test_metrics["RMSE"],
                    "Test MAE": test_metrics["MAE"],
                    "Test R-squared": test_metrics["R-squared"],
                }
            )

            if method == "RFE":
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
                f"{method_label} | k={feature_count} | "
                f"CV RMSE={np.mean(fold_rmse):,.2f} ± {np.std(fold_rmse, ddof=1):,.2f} | "
                f"Test RMSE={test_metrics['RMSE']:,.2f}"
            )

    comparison = pd.DataFrame(comparison_rows)
    rfe_results = comparison.loc[comparison["Method"] == "RFE"].copy()
    return rfe_results, {"models": model_details}, comparison


def plot_feature_selection(comparison: pd.DataFrame) -> None:
    colors = {
        "RFE": "#2563eb",
        "Lasso": "#e76f51",
        "Forward SFS": "#16866d",
        "Elastic Net": "#b04b83",
        "Exhaustive Search": "#c18115",
    }
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.8))
    fig.patch.set_facecolor("#f4f5ef")
    for axis in axes:
        axis.set_facecolor("#ffffff")
        axis.spines[["top", "right"]].set_visible(False)
        axis.grid(axis="y", color="#e4e8e5", linewidth=0.8)
        axis.set_xticks(range(1, 6))
        axis.set_xlabel("Number of Features")

    for method, group in comparison.groupby("Method", sort=False):
        group = group.sort_values("Number of Features")
        axes[0].plot(
            group["Number of Features"],
            group["CV RMSE"],
            color=colors[method],
            linewidth=2,
            marker="o",
            markersize=6,
            label=method,
        )
        axes[1].plot(
            group["Number of Features"],
            group["CV R-squared"],
            color=colors[method],
            linewidth=2,
            marker="o",
            markersize=6,
            label=method,
        )

    axes[0].set_title("5-Fold CV RMSE by Number of Features", loc="left", weight="bold")
    axes[0].set_ylabel("Mean CV RMSE (USD; zoomed axis)")
    axes[0].set_ylim(8800, 9900)
    axes[1].set_title("5-Fold CV R-squared by Number of Features", loc="left", weight="bold")
    axes[1].set_ylabel("Mean CV R-squared (zoomed axis)")
    axes[1].set_ylim(0.92, 0.95)
    axes[1].axhline(0, color="#89958e", linewidth=0.8)
    axes[1].legend(frameon=False, fontsize=8, loc="best")
    axes[0].legend(frameon=False, fontsize=8, loc="best")

    fig.suptitle("Feature selection comparison / nested CV on training data", x=0.08, ha="left", weight="bold")
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
    results, model_details, comparison = evaluate_feature_counts(
        X_train, X_test, y_train, y_test, feature_names
    )

    results.to_csv(
        OUTPUT_DIR / "feature_selection_results.csv",
        index=False,
        encoding="utf-8-sig",
        float_format="%.8f",
    )
    comparison.to_csv(
        OUTPUT_DIR / "method_comparison_results.csv",
        index=False,
        encoding="utf-8-sig",
        float_format="%.8f",
    )
    with (OUTPUT_DIR / "model_details.json").open("w", encoding="utf-8") as output_file:
        json.dump(model_details, output_file, ensure_ascii=False, indent=2)
    plot_feature_selection(comparison)
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