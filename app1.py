import streamlit as st
import pandas as pd
import numpy as np
import joblib
from scipy.optimize import minimize

# 设置页面配置（保持原样，图标依然是显示在浏览器标签页中）
image_url = "https://raw.githubusercontent.com/your-username/your-repo/main/assets/图片1.png"  # 更新为你的图片 URL
st.set_page_config(page_title="聚丙烯LOI模型", layout="wide", page_icon=image_url)

# 在页面上插入图标与标题
st.markdown(
    f"""
    <h1 style="display: flex; align-items: center;">
        <img src="{image_url}" style="width: 40px; height: 40px; margin-right: 10px;" />
        聚丙烯极限氧指数模型：性能预测 与 逆向设计
    </h1>
    """, 
    unsafe_allow_html=True
)

page = st.sidebar.selectbox("🔧 选择功能", ["性能预测", "逆向设计"])

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
            if unit_type != "质量 (g)" and total > 0:
                user_input = {k: v / total * 100 for k, v in user_input.items()}

            input_array = np.array([list(user_input.values())])
            input_scaled = scaler.transform(input_array)
            prediction = model.predict(input_scaled)[0]

            st.markdown("### 🎯 预测结果")
            st.metric(label="极限氧指数 (LOI)", value=f"{prediction:.2f} %")

elif page == "逆向设计":
    st.subheader("🎯 逆向设计：LOI → 配方")

    target_loi = st.number_input("🎯 请输入目标 LOI 值 (%)", value=50.0, step=0.1)

    if st.button("🔄 开始逆向设计"):
        with st.spinner("正在反推出最优配方，请稍候..."):

            # 初始猜测：随机生成各个特征的初始值，确保 PP 的初始值合理
            x0 = np.random.rand(len(feature_names))
            pp_index = feature_names.index("PP")
            x0[pp_index] = 0.7  # 初始PP较高

            bounds = [(0, 1)] * len(feature_names)
            bounds[pp_index] = (0.5, 1.0)

            # 目标函数：最小化预测 LOI 与目标 LOI 之间的差异
            def objective(x):
                x_norm = x / np.sum(x) * 100
                x_scaled = scaler.transform([x_norm])
                pred = model.predict(x_scaled)[0]
                return abs(pred - target_loi)

            # 约束：配方总和为 100
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
                st.dataframe(df_result.round(2))
            else:
                st.error("❌ 优化失败，请尝试更改目标 LOI 或检查模型")
