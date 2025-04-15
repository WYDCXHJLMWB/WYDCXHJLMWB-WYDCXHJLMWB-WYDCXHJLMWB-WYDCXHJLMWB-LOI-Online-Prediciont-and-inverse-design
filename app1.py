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

# 正向预测模块
if page == "性能预测":
    st.subheader("🔬 配方 → 预测 LOI")

    st.markdown(f"📎 当前单位：**{unit_type}**")

    user_input = {}
    total = 0

    cols = st.columns(3)
    for i, name in enumerate(feature_names):
        unit_label = {
            "质量 (g)": "g",
            "质量分数 (wt%)": "wt%",
            "体积分数 (vol%)": "vol%"
        }[unit_type]
        value = cols[i % 3].number_input(f"{name} ({unit_label})", value=0.0, step=0.1 if "质量" in unit_type else 0.01)
        user_input[name] = value
        total += value

    # 如果是分数类单位，进行归一化为总和100
    if "分数" in unit_type and total > 0:
        user_input = {k: v / total * 100 for k, v in user_input.items()}

    if st.button("开始预测"):
        input_array = np.array([list(user_input.values())])
        input_scaled = scaler.transform(input_array)
        prediction = model.predict(input_scaled)[0]
        st.success(f"🎯 预测结果：LOI = **{prediction:.3f}%**")

# 逆向设计模块
elif page == "逆向设计":
    st.subheader("🎯 LOI → 反推出配方")

    st.markdown(f"📎 当前配方单位：**{unit_type}**（仅支持质量分数 wt% 或体积分数 vol%）")

    target_loi = st.number_input("🎯 目标 LOI 值 (%)", value=50.0, step=0.1)

    if st.button("开始逆向设计"):
        with st.spinner("正在反推配方中，请稍候..."):

            x0 = np.random.rand(len(feature_names))
            pp_index = feature_names.index("PP")
            x0[pp_index] = 0.7  # 初始PP占比较高

            bounds = [(0, 1)] * len(feature_names)
            bounds[pp_index] = (0.5, 1.0)

            def objective(x):
                x_norm = x / np.sum(x) * 100  # 转为百分比
                x_scaled = scaler.transform([x_norm])
                pred = model.predict(x_scaled)[0]
                return abs(pred - target_loi)

            cons = {'type': 'eq', 'fun': lambda x: np.sum(x) - 1}

            result = minimize(objective, x0, bounds=bounds, constraints=cons, method='SLSQP')

            if result.success:
                best_x = result.x / np.sum(result.x) * 100  # 转为百分比
                pred_loi = model.predict(scaler.transform([best_x]))[0]

                st.success(f"✅ 成功反推配方，预测 LOI = **{pred_loi:.3f}%**")

                # 显示结果表格
                df_result = pd.DataFrame([best_x], columns=feature_names)
                unit_suffix = "wt%" if "质量" in unit_type else "vol%"
                df_result.columns = [f"{col} ({unit_suffix})" for col in df_result.columns]
                st.dataframe(df_result.style.format("{:.2f}"))
            else:
                st.error("❌ 优化失败，请尝试调整目标值或模型参数")
