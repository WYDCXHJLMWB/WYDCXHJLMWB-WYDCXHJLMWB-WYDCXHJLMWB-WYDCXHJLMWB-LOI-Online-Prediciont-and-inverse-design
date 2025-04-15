import streamlit as st
import pandas as pd
import numpy as np
import joblib
from scipy.optimize import minimize

# 页面设置
st.set_page_config(page_title="性能预测与逆向设计", layout="wide")
st.title("聚丙烯极限氧指数岭回归模型：性能预测 与 逆向设计")

# 选择功能
page = st.sidebar.selectbox("🔧 选择功能", ["性能预测", "逆向设计"])

# 加载模型和 scaler
data = joblib.load("model_and_scaler_loi.pkl")
model = data["model"]
scaler = data["scaler"]

# 加载特征名（已删除 LOI 列）
df = pd.read_excel("trainrg3.xlsx")
feature_names = df.columns.tolist()

if "LOI" in feature_names:
    feature_names.remove("LOI")

# 公共部分：单位选择
unit_type = st.radio("🧪 请选择填料单位", ["质量 (g)", "质量分数", "体积分数"], horizontal=True)

# 显示说明
if unit_type != "质量 (g)":
    st.markdown("📌 **注意：输入值总和将自动归一化为 1**（以确保比例有效）")

# 正向预测
if page == "性能预测":
    st.subheader("🔬 根据配方预测性能（LOI）")
    
    user_input = {}
    total = 0

    for name in feature_names:
        val = st.number_input(f"{name}", value=0.0, step=0.01 if unit_type != "质量 (g)" else 0.1)
        user_input[name] = val
        total += val

    # 归一化处理（如果是比例）
    if unit_type in ["质量分数", "体积分数"] and total > 0:
        user_input = {k: v / total * 100 for k, v in user_input.items()}  # 归一化为总和100

    if st.button("开始预测"):
        input_array = np.array([list(user_input.values())])
        input_scaled = scaler.transform(input_array)
        prediction = model.predict(input_scaled)[0]
        st.success(f"预测结果：LOI = **{prediction:.3f}**")

# 逆向设计
elif page == "逆向设计":
    st.subheader("🎯 逆向设计：根据目标性能反推配方")

    target_loi = st.number_input("目标 LOI 值", value=50.0, step=0.1)

    if st.button("开始逆向设计"):
        with st.spinner("正在反推出最优配方，请稍候..."):

            x0 = np.random.uniform(0.01, 1.0, len(feature_names))  # 初始化为比例
            pp_index = feature_names.index("PP")
            x0[pp_index] = 0.7  # PP 初始值偏大

            # 边界（比例范围）
            bounds = [(0, 1)] * len(feature_names)
            bounds[pp_index] = (0.5, 1)  # PP 范围更大

            # 目标函数
            def objective(x):
                x = x / np.sum(x) * 100  # 归一化为 100
                x_scaled = scaler.transform([x])
                pred = model.predict(x_scaled)[0]
                return abs(pred - target_loi)

            # 约束：比例总和为 1
            cons = ({'type': 'eq', 'fun': lambda x: np.sum(x) - 1})

            result = minimize(objective, x0, bounds=bounds, constraints=cons, method='SLSQP')

            if result.success:
                best_x = result.x / np.sum(result.x) * 100  # 转换为百分比
                pred_loi = model.predict(scaler.transform([best_x]))[0]
                st.success(f"✅ 找到配方！预测 LOI = {pred_loi:.3f}")
                df_result = pd.DataFrame([best_x], columns=feature_names)
                st.dataframe(df_result.style.format("{:.2f}"))
            else:
                st.error("❌ 优化失败，请检查目标值或模型设置")
