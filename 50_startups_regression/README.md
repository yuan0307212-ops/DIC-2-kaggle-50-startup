# 50 Startups 多元線性迴歸

以 50 Startups 資料集示範多元線性迴歸、One-Hot Encoding 與遞迴特徵消除（RFE）。執行程式會下載資料、以固定測試集比較 1 至 5 個特徵，並產生圖表、CSV 與可部署到 GitHub Pages 的繁中分析頁。

## 專案結構

```text
50_startups_regression/
├── data/50_Startups.csv
├── outputs/
│   ├── feature_selection.png
│   ├── train_test_rmse.png
│   ├── feature_selection_results.csv
│   ├── method_comparison_results.csv
│   └── model_details.json
├── index.html
├── main.py
├── requirements.txt
└── README.md
```

## 安裝與執行

```bash
cd 50_startups_regression
python3 -m venv .venv
source .venv/bin/activate       # Windows：.venv\Scripts\activate
python -m pip install -r requirements.txt
python main.py
```

程式會在資料不存在時下載 CSV 至 `data/`，並在 `outputs/` 產生兩張圖、RFE 指標 CSV、五種方法比較 CSV，以及供儀表板互動預測和殘差診斷使用的模型明細 JSON。若要在瀏覽器預覽分析頁，請從此資料夾啟動靜態伺服器：

```bash
python -m http.server 8000
```

開啟 `http://localhost:8000`。網頁會讀取程式輸出的 CSV 和圖表，因此執行 `main.py` 後即可呈現最新結果。

## 分析流程

1. 使用 pandas 讀取資料，檢查資料型態、缺失值、統計摘要及 State 分布。
2. 以 `test_size=0.2`、`random_state=42` 切分訓練集與測試集。
3. One-Hot Encoder 僅在訓練集 fit。為避免完全共線性，`drop="first"` 以 California 為基準，產生 `State_Florida` 與 `State_New York` 兩個 dummy variables；模型候選特徵共五個。
4. 比較 RFE、Lasso L1 排名、Forward SFS、Elastic Net 與 Exhaustive Search，在每個外層 fold 內只用該 fold 的訓練資料選特徵，使用巢狀 5-fold CV 估計 1 至 5 個特徵的泛化表現。
5. 以原始 40 筆訓練集重新選出最終特徵，訓練線性模型，最後只對固定的 10 筆 holdout 測試集評估一次；RFE 的 MSE、RMSE、MAE、R² 保留於 `feature_selection_results.csv`，五種方法的 CV 與測試分數放在 `method_comparison_results.csv`。

所有特徵都是數值型態；不含需要從資料估計的補值、縮放或其他全資料 fit 步驟。資料若出現缺失值，程式會明確停止，避免默默採用不合適的補值方式。

## 評估指標

- **MSE（均方誤差）**：實際值與預測值差值平方的平均；越低越好，但會放大較大的誤差。
- **RMSE（均方根誤差）**：MSE 開平方，單位與 Profit 相同；越低越好。
- **MAE（平均絕對誤差）**：預測誤差絕對值的平均，單位與 Profit 相同；越低越好。
- **R²（決定係數）**：模型相較於以平均值預測所解釋的變異比例；越接近 1 通常越好，也可能小於 0。

## Overfitting 與 Underfitting

`outputs/feature_selection.png` 顯示五種方法的訓練集外層 5-fold CV 平均 RMSE 與 R²；RMSE 圖使用縮放 y 軸呈現相近模型的差異，CV RMSE 標準差在比較表中另列。`outputs/train_test_rmse.png` 保留 RFE 訓練／holdout RMSE。兩者都高可能表示欠擬合；訓練 RMSE 低很多、測試 RMSE 卻高，可能表示過度擬合或小樣本切分波動。本資料只有 50 筆，應把差距視為診斷線索，不宜只依單次切分下定論。

## GitHub Pages

工作區根目錄的 `.github/workflows/deploy-pages.yml` 會將本資料夾發布為靜態網站。Repository 的 **Settings → Pages → Build and deployment** 需將 Source 設為 **GitHub Actions**；後續推送至 `main` 會自動重新部署。此專案目前部署網址為 `https://yuan0307212-ops.github.io/DIC-2-kaggle-50-startup/`。