import streamlit as st
import pandas as pd
import joblib
from datetime import datetime

# モデルの読み込み
model = joblib.load("sneaker_price_model.pkl")

# カスタムCSSでデザインを変更
st.markdown(
    """
    <style>
    .main {
        background-color: #f9f9f9;
    }
    h1 {
        font-family: 'Arial', sans-serif;
        color: #2c3e50;
        text-align: center;
    }
    .sidebar .sidebar-content {
        background-color: #ecf0f1;
    }
    .stButton>button {
        background-color: #2ecc71;
        color: white;
        border-radius: 5px;
        padding: 10px;
    }
    .stButton>button:hover {
        background-color: #27ae60;
    }
    </style>
    """,
    unsafe_allow_html=True
)

# タイトルと説明
st.title("👟 スニーカー価格予測アプリ")
st.write("""
このアプリは、過去のスニーカー取引データを基にして、指定したスニーカーの平均取引額を予測します。
現時点では、データが限られているため、予測結果はあくまで参考値としてご利用ください！
""")

# 注意書き
st.info("⚠️ 注意: モデルはまだ発展途上であり、データが不足しているため精度に限界があります。予測結果を参考にしつつ、実際の市場情報も確認してください。")

# サイドバーに入力フォームを表示
st.sidebar.header("スニーカー情報の入力")

model_name = st.sidebar.selectbox("モデル名", ["dunk sb", "AJ1low", "AJ1HIGH", "AJ3", "AJ4", "AJ11"])
color = st.sidebar.selectbox("カラー", ["gray", "green", "blue", "black", "white", "red", "yellow", "brown"])
price = st.sidebar.number_input("定価 (円)", min_value=0, step=500)
release_date = st.sidebar.date_input("販売日", value=datetime.now())
collaboration = st.sidebar.selectbox("コラボ", ["yes", "no"])

# 入力データを作成
month = release_date.month
weekday = release_date.weekday()

input_data = pd.DataFrame({
    "model": [model_name],
    "color": [color],
    "定価": [price],
    "month": [month],
    "weekday": [weekday],
    "コラボ": [collaboration]
})

# 予測ボタン
if st.sidebar.button("予測する"):
    prediction = model.predict(input_data)[0]
    st.subheader("🔮 予測結果")
    st.write(f"予測された平均取引額: **{prediction:.2f} 円**")
