import streamlit as st
import random
import numpy as np
import pandas as pd
from deap import base, creator, tools, algorithms
import joblib
from sklearn.preprocessing import StandardScaler

# 假设您的模型文件存在以下路径
scaler_path = "models/scaler_fold_1.pkl"
svc_path = "models/svc_fold_1.pkl"
loi_model_path = "models/loi_model.pkl"
ts_model_path = "models/ts_model.pkl"
loi_scaler_path = "models/loi_scaler.pkl"
ts_scaler_path = "models/ts_scaler.pkl"

# 载入模型
class Predictor:
    def __init__(self, scaler_path, svc_path):
        self.scaler = joblib.load(scaler_path)
        self.svc = joblib.load(svc_path)

    def predict_one(self, features):
        features_scaled = self.scaler.transform([features])
        prediction = self.svc.predict(features_scaled)
        return prediction[0]

# 载入配方模型
models = {
    "loi_model": joblib.load(loi_model_path),
    "ts_model": joblib.load(ts_model_path),
    "loi_scaler": joblib.load(loi_scaler_path),
    "ts_scaler": joblib.load(ts_scaler_path),
    "loi_features": ["feature1", "feature2", "feature3"],  # 修改为实际的特征
    "ts_features": ["feature1", "feature2", "feature3"]  # 修改为实际的特征
}

def ensure_pp_first(features):
    if "PP" in features:
        features.remove("PP")
        features.insert(0, "PP")
    return features

def get_unit(fraction_type):
    return "体积分数" if fraction_type == "体积分数" else "质量百分比"

# 初始化遗传算法
creator.create("FitnessMin", base.Fitness, weights=(-1.0,))
creator.create("Individual", list, fitness=creator.FitnessMin)

toolbox = base.Toolbox()
all_features = ensure_pp_first(list(set(models["loi_features"] + models["ts_features"])))
n_features = len(all_features)

# 生成满足和为100的配方
def generate_individual():
    individual = [random.uniform(0, 100) for _ in range(n_features)]
    total = sum(individual)
    return [max(0, x / total * 100) for x in individual]

toolbox.register("individual", tools.initIterate, creator.Individual, generate_individual)
toolbox.register("population", tools.initRepeat, list, toolbox.individual)

def evaluate(individual):
    if fraction_type == "体积分数":
        vol_values = np.array(individual)
        mass_values = vol_values
        total_mass = mass_values.sum()
        mass_percent = (mass_values / total_mass) * 100
    else:
        total = sum(individual)
        mass_percent = np.array(individual) / total * 100

    pp_index = all_features.index("PP")
    pp_content = mass_percent[pp_index]
    if pp_content < 50:
        return (1e6,)

    loi_input = mass_percent[:len(models["loi_features"])]
    loi_scaled = models["loi_scaler"].transform([loi_input])
    loi_pred = models["loi_model"].predict(loi_scaled)[0]
    loi_error = abs(target_loi - loi_pred)

    ts_input = mass_percent[:len(models["ts_features"])]
    ts_scaled = models["ts_scaler"].transform([ts_input])
    ts_pred = models["ts_model"].predict(ts_scaled)[0]
    ts_error = abs(target_ts - ts_pred)

    return (loi_error + ts_error,)

toolbox.register("mate", tools.cxBlend, alpha=0.5)
toolbox.register("mutate", tools.mutGaussian, mu=0, sigma=1, indpb=0.2)
toolbox.register("select", tools.selTournament, tournsize=3)
toolbox.register("evaluate", evaluate)

# 页面结构
st.title("配方优化系统")

# 页面选择
sub_page = st.radio("请选择操作", ("配方优化", "添加剂推荐"))

if sub_page == "配方优化":
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
        population = toolbox.population(n=pop_size)
        algorithms.eaSimple(population, toolbox, cxpb=cx_prob, mutpb=mut_prob, ngen=n_gen, verbose=False)

        best_individuals = tools.selBest(population, 10)
        best_values = []
        for individual in best_individuals:
            total = sum(individual)
            best_values.append([round(max(0, i / total * 100), 2) for i in individual])

        result_df = pd.DataFrame(best_values, columns=all_features)
        units = [get_unit("体积分数") for _ in all_features]
        result_df.columns = [f"{col} ({unit})" for col, unit in zip(result_df.columns, units)]

        st.write(result_df)

elif sub_page == "添加剂推荐":
    st.subheader("根据输入性能推荐添加剂配方")
    predictor = Predictor(scaler_path, svc_path)
    
    # 获取输入特征
    features = st.text_input("输入特征值，格式：[feature1, feature2, feature3]")

    if st.button("🔍 开始预测"):
        if features:
            features = eval(features)  # 将输入的字符串转换为列表
            prediction = predictor.predict_one(features)
            st.write(f"推荐的添加剂配方为：{prediction}")
