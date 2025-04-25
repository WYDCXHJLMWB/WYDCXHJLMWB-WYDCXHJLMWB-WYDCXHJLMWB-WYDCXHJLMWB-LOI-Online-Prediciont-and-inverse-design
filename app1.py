import streamlit as st
import pandas as pd
import numpy as np
import joblib
from sklearn.preprocessing import StandardScaler
from deap import base, creator, tools, algorithms
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

elif page == "配方建议":
    target_loi = st.number_input("🎯 请输入目标 LOI 值 (%)", value=20.0, step=0.1, min_value=10.0, max_value=40.0)
    output_mode = st.selectbox("📦 请选择输出形式", ["质量分数（wt%）", "质量（g）", "体积分数（vol%）"])

    if target_loi < 10 or target_loi > 40:
        st.warning("⚠️ 目标 LOI 值必须在 10 到 40 之间，请重新输入。")
    else:
        st.write("🔄 正在进行逆向设计，请稍等...")

        pp_index = feature_names.index("PP")
        num_features = len(feature_names)

        creator.create("FitnessMin", base.Fitness, weights=(-1.0,))
        creator.create("Individual", list, fitness=creator.FitnessMin)

        def make_valid_individual():
            ind = np.random.uniform(0.1, 1, num_features)
            ind[pp_index] = max(ind) + 0.1
            ind = np.clip(ind, 0, None)
            return creator.Individual(ind)

        def evaluate(ind):
            ind = np.clip(ind, 0, None)
            if ind[pp_index] <= max([x for i, x in enumerate(ind) if i != pp_index]):
                return 1e6,
            norm = ind / np.sum(ind) * 100  # 确保加和为100
            X_scaled = scaler.transform([norm])
            y_pred = model.predict(X_scaled)[0]
            return abs(y_pred - target_loi),

        toolbox = base.Toolbox()
        toolbox.register("individual", make_valid_individual)
        toolbox.register("population", tools.initRepeat, list, toolbox.individual)
        toolbox.register("evaluate", evaluate)
        toolbox.register("mate", tools.cxBlend, alpha=0.5)
        toolbox.register("mutate", tools.mutGaussian, mu=0.0, sigma=0.1, indpb=0.3)
        toolbox.register("select", tools.selTournament, tournsize=3)

        pop = toolbox.population(n=100)
        hof = tools.HallOfFame(20)

        algorithms.eaSimple(pop, toolbox, cxpb=0.7, mutpb=0.3, ngen=60, halloffame=hof, verbose=False)

        results = []
        for ind in hof:
            ind = np.clip(ind, 0, None)
            norm = ind / np.sum(ind) * 100
            if norm[pp_index] <= max([x for i, x in enumerate(norm) if i != pp_index]):
                continue
            pred_loi = model.predict(scaler.transform([norm]))[0]
            results.append(list(norm) + [pred_loi])

        if len(results) == 0:
            st.error("❌ 未能生成符合条件的配方，请尝试调整目标值或放宽条件。")
        else:
            df_result = pd.DataFrame(results[:10], columns=feature_names + ["预测 LOI"])

            if output_mode == "质量（g）":
                df_result.iloc[:, :-1] = df_result.iloc[:, :-1] * 1.0  # 总质量100g
                df_result.columns = [f"{col} (g)" if col != "预测 LOI" else col for col in df_result.columns]
            elif output_mode == "质量分数（wt%）":
                df_result.columns = [f"{col} (wt%)" if col != "预测 LOI" else col for col in df_result.columns]
            elif output_mode == "体积分数（vol%）":
                volume_fractions = df_result.iloc[:, :-1].div(df_result.iloc[:, :-1].sum(axis=1), axis=0) * 100
                df_result.iloc[:, :-1] = volume_fractions
                df_result.columns = [f"{col} (vol%)" if col != "预测 LOI" else col for col in df_result.columns]

            st.markdown("### 📋 推荐配方")
            st.dataframe(df_result.round(2))
