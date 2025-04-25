import streamlit as st
import pandas as pd
import numpy as np
import joblib
from hyperopt import hp, fmin, tpe, STATUS_OK, Trials
from sklearn.preprocessing import StandardScaler
from PIL import Image
import base64

# Function to convert image to base64
def image_to_base64(image_path):
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode()

# 设置页面配置（保持原样，图标依然是显示在浏览器标签页中）
image_path = "图片1.png"  # 使用上传的图片路径
icon_base64 = image_to_base64(image_path)  # 转换为 base64

# 设置页面标题和图标
st.set_page_config(page_title="聚丙烯LOI模型", layout="wide", page_icon=f"data:image/png;base64,{icon_base64}")

# 图标原始尺寸：507x158，计算出比例
width = 200  # 设置图标的宽度为100px
height = int(158 * (width / 507))  # 计算保持比例后的高度

# 在页面上插入图标与标题
st.markdown(
    f"""
    <h1 style="display: flex; align-items: center;">
        <img src="data:image/png;base64,{icon_base64}" style="width: {width}px; height: {height}px; margin-right: 15px;" />
        阻燃聚合物复合材料智能设计平台
    </h1>
    """, 
    unsafe_allow_html=True
)

page = st.sidebar.selectbox("🔧 选择功能", ["性能预测", "配方建议"])

# 加载模型与缩放器
data = joblib.load("model_and_scaler_loi.pkl")
model = data["model"]
scaler = data["scaler"]

df = pd.read_excel("trainrg3.xlsx")
feature_names = df.columns.tolist()
if "LOI" in feature_names:
    feature_names.remove("LOI")

unit_type = st.radio("📏 请选择配方输入单位", ["质量 (g)", "质量分数 (wt%)", "体积分数 (vol%)"], horizontal=True)

if page == "性能预测":
    st.subheader("🔬 正向预测：配方 → LOI")

    with st.form("input_form"):
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

        submitted = st.form_submit_button("📊 开始预测")

    if submitted:
        # 判断总和是否满足为100
        if unit_type != "质量 (g)" and abs(total - 100) > 1e-3:
            st.warning("⚠️ 配方加和不为100，无法预测。请确保总和为100后再进行预测。")
        else:
            # 若是分数单位，则再归一化一遍
            if unit_type == "质量 (g)" and total > 0:  # 判断是否为质量单位
                # 将每个成分的质量转换为质量分数
                user_input = {k: (v / total) * 100 for k, v in user_input.items()}  # 归一化为质量分数

            elif unit_type != "质量 (g)" and total > 0:
                user_input = {k: v / total * 100 for k, v in user_input.items()}  # 确保总和为100

            # 检查是否仅输入了PP，并且PP为100
            if np.all([user_input.get(name, 0) == 0 for name in feature_names if name != "PP"]) and user_input.get("PP", 0) == 100:
                # 如果只输入了PP且PP为100，强制返回LOI=17.5
                st.markdown("### 🎯 预测结果")
                st.metric(label="极限氧指数 (LOI)", value="17.5 %")
            else:
                input_array = np.array([list(user_input.values())])
                input_scaled = scaler.transform(input_array)
                prediction = model.predict(input_scaled)[0]

                st.markdown("### 🎯 预测结果")
                st.metric(label="极限氧指数 (LOI)", value=f"{prediction:.2f} %")

elif page == "逆向设计":

    target_loi = st.number_input("🎯 请输入目标 LOI 值 (%)", value=50.0, step=0.1)

    if st.button("🔄 开始逆向设计"):
        with st.spinner("正在反推出最优配方，请稍候..."):

            # 配方范围
            bounds = {
                feature: (0, 100) for feature in feature_names
            }
            pp_index = feature_names.index("PP")
            bounds["PP"] = (50, 100)  # 设定PP的范围更高一些

            # 目标函数：最小化预测 LOI 与目标 LOI 之间的差异
            def objective(params):
                # 将参数归一化，使总和为100
                params_norm = np.array(params) / sum(params) * 100
                params_norm = params_norm.round(3)
                
                # 标准化输入并预测
                params_scaled = scaler.transform([params_norm])
                prediction = model.predict(params_scaled)[0]

                # 返回误差
                return abs(prediction - target_loi)

            # 配方搜索空间
            space = [hp.uniform(feature, bounds[feature][0], bounds[feature][1]) for feature in feature_names]
            
            # 运行Hyperopt优化
            trials = Trials()
            best = fmin(
                fn=objective,
                space=space,
                algo=tpe.suggest,
                max_evals=100,
                trials=trials
            )
            
            # 显示结果
            best_params = np.array([best[feature] for feature in feature_names])
            best_params_norm = best_params / sum(best_params) * 100
            best_prediction = model.predict(scaler.transform([best_params_norm]))[0]

            st.success("🎉 成功反推配方！")
            st.metric("预测 LOI", f"{best_prediction:.2f} %")

            # 显示最优配方
            df_result = pd.DataFrame([best_params_norm], columns=feature_names)
            df_result.columns = [f"{col} (wt%)" for col in df_result.columns]

            st.markdown("### 📋 最优配方参数")
            st.dataframe(df_result.round(2))

