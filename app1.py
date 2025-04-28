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
fraction_type = st.sidebar.radio("📐 分数类型", ["质量", "质量分数", "体积分数"])

# 加载模型
@st.cache_resource
def load_models():
    loi_data = joblib.load("model_and_scaler_loi.pkl")
    ts_data = joblib.load("model_and_scaler_ts1.pkl")
    return {
        "loi_model": loi_data["model"],
        "loi_scaler": loi_data["scaler"],
        "ts_model": ts_data["model"],
        "ts_scaler": ts_data["scaler"],
        "loi_features": pd.read_excel("trainrg3.xlsx").drop(columns="LOI", errors='ignore').columns.tolist(),
        "ts_features": pd.read_excel("trainrg3TS.xlsx").drop(columns="TS", errors='ignore').columns.tolist(),
    }
models = load_models()

# 配方建议页面
if page == "配方建议":
    st.subheader("🧪 配方建议：根据性能反推配方")
    
    # 目标输入
    col1, col2 = st.columns(2)
    with col1:
        target_loi = st.number_input("目标LOI值（%）", min_value=10.0, max_value=50.0, value=25.0, step=0.1)
    with col2:
        target_ts = st.number_input("目标TS值（MPa）", min_value=0.0, max_value=100.0, value=50.0, step=0.1)
    
    # 遗传算法参数
    with st.expander("⚙️ 算法参数设置"):
        pop_size = st.number_input("种群数量", 50, 500, 200)
        n_gen = st.number_input("迭代代数", 10, 100, 50)
        cx_prob = st.slider("交叉概率", 0.1, 1.0, 0.7)
        mut_prob = st.slider("变异概率", 0.1, 1.0, 0.2)

    if st.button("🔍 开始优化", type="primary"):
        # 初始化遗传算法
        creator.create("FitnessMin", base.Fitness, weights=(-1.0,))  # 优化目标是最小化误差
        creator.create("Individual", list, fitness=creator.FitnessMin)
        
        toolbox = base.Toolbox()
        all_features = list(set(models["loi_features"] + models["ts_features"]))
        n_features = len(all_features)
        
        toolbox.register("attr_float", random.uniform, 0.1, 100)
        toolbox.register("individual", tools.initRepeat, creator.Individual, toolbox.attr_float, n=n_features)
        toolbox.register("population", tools.initRepeat, list, toolbox.individual)
        
        def evaluate(individual):
            # 单位转换处理
            if fraction_type == "体积分数":
                # 转换为质量分数
                vol_values = np.array(individual)
                mass_values = vol_values  # 直接使用体积分数比例表示质量分数
                total_mass = mass_values.sum()
                if total_mass == 0:
                    return (1e6,)
                mass_percent = (mass_values / total_mass) * 100
            else:
                total = sum(individual)
                if total == 0:
                    return (1e6,)
                mass_percent = np.array(individual) / total * 100
            
            # PP约束
            pp_index = all_features.index("PP")
            pp_content = mass_percent[pp_index]
            if pp_content < 50:  # PP含量过低惩罚
                return (1e6,)
            
            # LOI计算
            loi_input = mass_percent[:len(models["loi_features"])]
            loi_scaled = models["loi_scaler"].transform([loi_input])
            loi_pred = models["loi_model"].predict(loi_scaled)[0]
            loi_error = abs(target_loi - loi_pred)
            
            # TS计算
            ts_input = mass_percent[:len(models["ts_features"])]
            ts_scaled = models["ts_scaler"].transform([ts_input])
            ts_pred = models["ts_model"].predict(ts_scaled)[0]
            ts_error = abs(target_ts - ts_pred)
            
            return (loi_error + ts_error,)
        
        toolbox.register("mate", tools.cxBlend, alpha=0.5)
        toolbox.register("mutate", tools.mutGaussian, mu=0, sigma=1, indpb=0.2)
        toolbox.register("select", tools.selTournament, tournsize=3)
        toolbox.register("evaluate", evaluate)
        
        population = toolbox.population(n=pop_size)
        algorithms.eaSimple(population, toolbox, cxpb=cx_prob, mutpb=mut_prob, ngen=n_gen, verbose=False)
        
        best_individuals = tools.selBest(population, 10)  # 获取前10个最佳个体
        
        # 输出优化结果
        result_data = []
        for best_individual in best_individuals:
            best_values = [round(i, 2) for i in best_individual]
            # 确保每个配方的总和为100%
            total = sum(best_values)
            if abs(total - 100) > 1e-6:
                best_values = [val / total * 100 for val in best_values]  # 标准化为100%
            result_data.append(best_values)
        
        result_df = pd.DataFrame(result_data, columns=all_features)
        st.write(result_df)
