# -*- coding: utf-8 -*-
"""
聚丙烯极限氧指数预测与逆向设计系统
"""

import streamlit as st
import pandas as pd
import numpy as np
import joblib
from scipy.optimize import minimize

# 页面设置
st.set_page_config(page_title="聚丙烯LOI模型", layout="wide")
st.title("🧪 聚丙烯极限氧指数模型：性能预测 与 逆向设计")

# 选择功能
page = st.sidebar.selectbox("🔧 选择功能", ["性能预测", "逆向设计"])

# 加载模型和数据
data = joblib.load("model_and_scaler_loi.pkl")
model = data["model"]
scaler = data["scaler"]

df = pd.read_excel("trainrg3.xlsx")
feature_names = df.columns.tolist()
if "LOI" in feature_names:
    feature_names.remove("LOI")

# 添加单位选择
unit_type = st.radio("📏 请选择配方输入单位", ["质量 (g)", "质量分数 (wt%)", "体积分数 (vol%)"], horizontal=True)

# =========================== 性能预测 ===========================
if page == "性能预测":
    st.subheader("🔬 正向预测：配方 → LOI")

    with st.expander("📦 输入配方参数"):
        user_input = {}
        total = 0
        cols = st.columns(3)
        for i, name in enumerate(feature_names):
            unit_label = {
                "质量 (g)": "g",
                "质量分数 (wt%)": "wt%",
                "体积分数 (vol%)": "vol%"
            }[unit_type]
            val = cols[i % 3].number_input(f"{name} ({unit_label})", value=0.0, step=0.1 if "质量" in unit_type else 0.01)
            user_input[name] = val
            total += val

        if unit_type != "质量 (g)" and abs(total - 100) > 1e-3:
            st.warning("⚠️ 当前输入为分数单位，总和必须为 100。请检查输入是否正确。")

    if st.button("📊 开始预测"):
        if unit_type != "质量 (g)" and abs(total - 100) > 1e-3:
            st.error("❌ 输入的总和不为100，无法预测。")
        else:
            # 如果是分数单位，归一化成 100
            if unit_type != "质量 (g)" and total > 0:
                user_input = {k: v / total * 100 for k, v in user_input.items()}

            input_array = np.array([list(user_input.values())])
            input_scaled = scaler.transform(input_array)
            prediction = model.predict(input_scaled)[0]

            st.markdown("### 🎯 预测结果")
            st.metric(label="极限氧指数 (LOI)", value=f"{prediction:.2f} %")

# =========================== 逆向设计 ===========================
elif page == "逆向设计":
    st.subheader("🎯 逆向设计：LOI → 配方")

    target_loi = st.number_input("🎯 请输入目标 LOI 值 (%)", value=50.0, step=0.1)

    if st.button("🔄 开始逆向设计"):
        with st.spinner("正在反推出最优配方，请稍候..."):

            x0 = np.random.rand(len(feature_names))
            pp_index = feature_names.index("PP")
            x0[pp_index] = 0.7  # 初始PP较高

            bounds = [(0, 1)] * len(feature_names)
            bounds[pp_index] = (0.5, 1.0)

            def objective(x):
                x_norm = x / np.sum(x) * 100
                x_scaled = scaler.transform([x_norm])
                pred = model.predict(x_scaled)[0]
                return abs(pred - target_loi)

            cons = {'type': 'eq', 'fun': lambda x: np.sum(x) - 1}

            result = minimize(objective, x0, bounds=bounds, constraints=cons, method='SLSQP')

            if result.success:
                best_x = result.x / np.sum(result.x) * 100
                pred_loi = model.predict(scaler.transform([best_x]))[0]

                st.success("🎉 成功反推配方！")
                st.metric("预测 LOI", f"{pred_loi:.2f} %")

                unit_suffix = "wt%" if "质量" in unit_type else "vol%"
                df_result = pd.DataFrame([best_x], columns=feature_names)
                df_result.columns = [f"{col} ({unit_suffix})" for col in df_result.columns]

                st.markdown("### 📋 最优配方参数")
                st.dataframe(df_result.style.background_gradient(cmap='Blues').format("{:.2f}"))
            else:
                st.error("❌ 优化失败，请尝试更改目标 LOI 或检查模型")
