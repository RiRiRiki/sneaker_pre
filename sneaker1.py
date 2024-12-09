import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error
import joblib

# CSVファイルを読み込む
data = pd.read_csv("sneaker_data2.csv")  # 同じディレクトリに配置

# 日付型変換と特徴量生成
data["日時"] = pd.to_datetime(data["日時"], errors="coerce")  # 日付型変換（エラー処理付き）
data.dropna(subset=["日時"], inplace=True)  # 日付が欠損している行を削除
data["month"] = data["日時"].dt.month
data["weekday"] = data["日時"].dt.weekday

# 特徴量とターゲットの設定
X = data[["model", "color", "定価", "month", "weekday", "コラボ"]]
y = data["平均取引額"]

# 訓練データとテストデータに分割
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# カテゴリ変数と数値変数の処理
categorical_features = ["model", "color", "コラボ"]
numeric_features = ["定価", "month", "weekday"]

preprocessor = ColumnTransformer(
    transformers=[
        ("num", StandardScaler(), numeric_features),
        ("cat", OneHotEncoder(handle_unknown="ignore"), categorical_features)
    ]
)

# ランダムフォレストモデルのパイプライン
model = Pipeline(steps=[
    ("preprocessor", preprocessor),
    ("regressor", RandomForestRegressor(random_state=42))
])

# モデル訓練
model.fit(X_train, y_train)

# テストデータでの予測
y_pred = model.predict(X_test)

# モデル評価
mae = mean_absolute_error(y_test, y_pred)
print(f"平均絶対誤差 (MAE): {mae:.2f}")

# 訓練済みモデルを保存
joblib.dump(model, "sneaker_price_model.pkl")
print("モデルを sneaker_price_model.pkl として保存しました！")
