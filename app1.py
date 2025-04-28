import streamlit as st
import pandas as pd
import numpy as np
import joblib
import base64
import random
from deap import base, creator, tools, algorithms

# 辅助函数：图片转base64
def image_to_base64(image_path):
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode()

# 页面配置
image_path = "图片1.png"
icon_base64 = image_to_base64(image_path)
st.set_page_config(
    page_title="聚丙烯LOI和TS模型",
    layout="wide",
    page_icon=f"data:image/png;base64,{icon_base64}"
)

# 页面标题样式
width = 200
height = int(158 * (width / 507))
st.markdown(
    f"""
    <h1 style="display: flex; align-items: center;">
        <img src="data:image/png;base64,{icon_base64}" style="width: {width}px; height: {height}px; margin-right: 15px;" />
        阻燃聚合物复合材料智能设计平台
    </h1>
    """, 
    unsafe_allow_html=True
)

# 侧边栏导航
page = st.sidebar.selectbox("🔧 选择功能", ["性能预测", "配方建议"])

# 加载LOI模型和Scaler
loi_data = joblib.load("model_and_scaler_loi.pkl")
loi_model = loi_data["model"]
loi_scaler = loi_data["scaler"]

# 加载TS模型和Scaler
ts_data = joblib.load("model_and_scaler_ts1.pkl")
ts_model = ts_data["model"]
ts_scaler = ts_data["scaler"]

# 加载训练数据，获取特征名称
df_loi = pd.read_excel("trainrg3.xlsx")
df_ts = pd.read_excel("trainrg3TS.xlsx")

loi_feature_names = df_loi.columns.tolist()
ts_feature_names = df_ts.columns.tolist()

# 移除LOI和TS列，得到特征名称
if "LOI" in loi_feature_names:
    loi_feature_names.remove("LOI")

if "TS" in ts_feature_names:
    ts_feature_names.remove("TS")

# 性能预测页面
if page == "性能预测":
    st.subheader("🔮 性能预测：基于配方预测LOI和TS")

    # 合并LOI和TS输入区域
    st.write("请输入配方特征值：")

    # 输入所有特征值（统一输入）
    input_data = {}
    for feature in set(loi_feature_names + ts_feature_names):  # 合并LOI和TS的特征
        input_data[feature] = st.number_input(f"请输入 {feature} 的特征值", value=0.0, step=0.1)

    # 性能预测按钮
    if st.button("预测LOI和TS"):
        # 将输入数据转化为数组
        input_array = np.array([list(input_data.values())])

        # LOI预测：仅选择LOI相关特征
        loi_input_array = np.array([list(input_data[feature] for feature in loi_feature_names)])
        if len(loi_input_array[0]) == len(loi_feature_names):
            # 标准化并预测LOI
            loi_input_scaled = loi_scaler.transform(loi_input_array)
            predicted_loi = loi_model.predict(loi_input_scaled)[0]
        else:
            st.error(f"LOI输入特征数量不匹配：期望 {len(loi_feature_names)}，实际输入 {len(loi_input_array[0])}")

        # TS预测：仅选择TS相关特征
        ts_input_array = np.array([list(input_data[feature] for feature in ts_feature_names)])
        if len(ts_input_array[0]) == len(ts_feature_names):
            # 标准化并预测TS
            ts_input_scaled = ts_scaler.transform(ts_input_array)
            predicted_ts = ts_model.predict(ts_input_scaled)[0]
        else:
            st.error(f"TS输入特征数量不匹配：期望 {len(ts_feature_names)}，实际输入 {len(ts_input_array[0])}")

        # 显示预测结果
        if len(loi_input_array[0]) == len(loi_feature_names) and len(ts_input_array[0]) == len(ts_feature_names):
            st.success(f"预测的LOI值为：{predicted_loi:.2f} %")
            st.success(f"预测的TS值为：{predicted_ts:.2f} MPa")

# 配方建议页面
elif page == "配方建议":
    st.subheader("🧪 配方建议：根据性能反推配方")
    target_loi = st.number_input("目标LOI值", min_value=10.0, max_value=50.0, value=25.0, step=0.1)
    target_ts = st.number_input("目标TS值", min_value=0.0, max_value=100.0, value=50.0, step=0.1)
    
    # 遗传算法配置
    if not hasattr(creator, "FitnessMin"):
        creator.create("FitnessMin", base.Fitness, weights=(-1.0,))  # 目标是最小化
    if not hasattr(creator, "Individual"):
        creator.create("Individual", list, fitness=creator.FitnessMin)
    
    toolbox = base.Toolbox()
    toolbox.register("attr_float", random.uniform, 0.01, 50)
    toolbox.register("individual", tools.initRepeat, creator.Individual, toolbox.attr_float, n=len(loi_feature_names))  # 使用LOI特征数量
    toolbox.register("population", tools.initRepeat, list, toolbox.individual)
    
    def evaluate(individual):
        # 强制PP含量>=50且为最大值
        pp_index = loi_feature_names.index("PP")
        if individual[pp_index] < 50:
            return (1000,)
        if individual[pp_index] != max(individual):
            return (1000,)
            
        # 归一化处理
        total = sum(individual)
        normalized = [x/total*100 for x in individual]
        
        # 预测LOI
        input_array = np.array([normalized])
        input_scaled = loi_scaler.transform(input_array)
        predicted_loi = loi_model.predict(input_scaled)[0]
        
        # 预测TS
        ts_input_scaled = ts_scaler.transform(np.array([normalized]))
        predicted_ts = ts_model.predict(ts_input_scaled)[0]

        # 目标函数：最小化LOI和TS的差距
        return (abs(predicted_loi - target_loi) + abs(predicted_ts - target_ts),)
    
    toolbox.register("mate", tools.cxBlend, alpha=0.5)
    toolbox.register("mutate", tools.mutGaussian, mu=0, sigma=1, indpb=0.2)
    toolbox.register("select", tools.selTournament, tournsize=3)
    toolbox.register("evaluate", evaluate)

    # 执行遗传算法
    population = toolbox.population(n=10)
    result = algorithms.eaSimple(population, toolbox, cxpb=0.7, mutpb=0.2, ngen=10, verbose=False)
    
    # 输出最优配方
    best_individual = tools.selBest(population, 1)[0]
    st.write("最优配方：", best_individual)

