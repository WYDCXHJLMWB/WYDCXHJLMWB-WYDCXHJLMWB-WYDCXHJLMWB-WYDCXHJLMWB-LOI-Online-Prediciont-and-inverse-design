# -*- coding: utf-8 -*-
"""
Created on Wed Apr  9 18:36:18 2025

@author: ma'wei'bin
"""

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

# 保险处理，剔除 LOI
if "LOI" in feature_names:
    feature_names.remove("LOI")

# 性能预测页面
if page == "性能预测":
    st.subheader("🔬 根据配方预测性能（LOI）")
    
    user_input = {}
    for name in feature_names:
        # 显示配方特征及其单位
        user_input[name] = st.number_input(f"{name} (wt%)", value=0.0, step=0.1)
    
    if st.button("开始预测"):
        input_array = np.array([list(user_input.values())])
        input_scaled = scaler.transform(input_array)
        prediction = model.predict(input_scaled)[0]
        st.success(f"预测结果：LOI = **{prediction:.3f}%**")

# 逆向设计页面
elif page == "逆向设计":
    st.subheader("🎯 逆向设计：根据目标性能反推配方")

    target_loi = st.number_input("目标 LOI 值 (%)", value=50.0, step=0.1)

    if st.button("开始逆向设计"):
        with st.spinner("正在反推出最优配方，请稍候..."):

            # 初始猜测：随机生成各个特征的初始值，确保 PP 的初始值合理
            x0 = np.random.uniform(0, 100, len(feature_names))  # 随机初始化配方比例
            pp_index = feature_names.index("PP")  # 找到 PP 在特征中的索引
            x0[pp_index] = np.random.uniform(50, 100)  # 设置 PP 初始值为 50 到 100 之间的随机值

            # 设置边界，PP 的范围是 50 到 100 之间，其他特征为 0 到 100 之间
            bounds = [(0, 100)] * len(feature_names)
            bounds[pp_index] = (70, 100)  # PP 的比例范围是 50 到 100

            # 目标函数：最小化预测 LOI 与目标 LOI 之间的差异
            def objective(x):
                # 将配方比例归一化，使其总和为 100
                x_sum = np.sum(x)
                if x_sum != 0:
                    x = x / x_sum * 100  # 归一化

                x_scaled = scaler.transform([x])  # 对配方进行标准化
                pred = model.predict(x_scaled)[0]  # 使用模型预测 LOI
                return abs(pred - target_loi)  # 目标是最小化 LOI 与目标值的差距

            # 约束：配方总和为 100
            def constraint(x):
                return np.sum(x) - 100  # 配方比例和应该等于 100

            # 将约束加入到优化过程中
            cons = ({'type': 'eq', 'fun': constraint})  # 使用eq约束确保总和为100

            # 执行优化
            result = minimize(objective, x0, bounds=bounds, constraints=cons, method='SLSQP')

            if result.success:
                best_x = result.x
                # 反推的最佳配方
                pred_loi = model.predict(scaler.transform([best_x]))[0]  # 使用最佳配方预测 LOI

                # 显示结果
                st.success(f"✅ 找到配方！预测 LOI = {pred_loi:.3f}%")
                df_result = pd.DataFrame([best_x], columns=feature_names)
                # 为每个配方成分添加单位 wt%
                df_result = df_result.applymap(lambda x: f"{x:.2f} wt%")
                st.dataframe(df_result)
            else:
                st.error("❌ 优化失败，请检查模型或目标值是否合理")
