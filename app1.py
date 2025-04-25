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
        # 保证总和为100
        if unit_type != "质量 (g)" and abs(total - 100) > 1e-3:
            st.warning("⚠️ 配方加和不为100，无法预测。请确保总和为100后再进行预测。")
        else:
            # 如果是质量单位，将质量转换为质量分数
            if unit_type == "质量 (g)" and total > 0:
                user_input = {k: (v / total) * 100 for k, v in user_input.items()}  # 归一化为质量分数

            # 如果是质量分数单位，直接根据比例转换为体积分数
            if unit_type == "质量分数 (wt%)":
                total_weight = sum(user_input.values())
                vol_frac = {name: (mass_fraction / total_weight) * 100 for name, mass_fraction in user_input.items()}
                user_input = vol_frac

            # 如果是体积分数单位，直接根据比例转换为质量分数
            elif unit_type == "体积分数 (vol%)":
                # 计算各成分的体积分数转换为质量分数
                total_volume = sum(user_input.values())
                density = {"PP": 0.91, "添加剂1": 1.0, "添加剂2": 1.2}  # 示例密度字典，实际需要根据配方调整
                mass_frac = {}
                for name, vol_fraction in user_input.items():
                    vol_frac = vol_fraction / total_volume  # 比例
                    if name in density:
                        mass_frac[name] = vol_frac * density[name] * 100
                    else:
                        mass_frac[name] = vol_frac * 100  # 没有密度数据的默认处理
                user_input = mass_frac

            # 检查是否仅输入了PP，并且PP为100
            if np.all([user_input.get(name, 0) == 0 for name in feature_names if name != "PP"]) and user_input.get("PP", 0) == 100:
                st.markdown("### 🎯 预测结果")
                st.metric(label="极限氧指数 (LOI)", value="17.5 %")
            else:
                input_array = np.array([list(user_input.values())])
                input_scaled = scaler.transform(input_array)
                prediction = model.predict(input_scaled)[0]

                st.markdown("### 🎯 预测结果")
                st.metric(label="极限氧指数 (LOI)", value=f"{prediction:.2f} %")

# 配方建议部分修改
elif page == "配方建议":
    st.subheader("🧪 配方建议：根据性能反推配方")

    # 用户输入目标LOI值并确保范围在10到50之间
    target_loi = st.number_input("请输入目标极限氧指数 (LOI)", min_value=10.0, max_value=50.0, value=25.0, step=0.1)

    # 如果用户输入的目标LOI超出范围，提醒用户
    if target_loi < 10 or target_loi > 50:
        st.warning("⚠️ 目标LOI应在10到50之间，请重新输入。")

    # 添加遗传算法的部分
    creator.create("FitnessMin", base.Fitness, weights=(-1.0,))  # 最小化目标
    creator.create("Individual", list, fitness=creator.FitnessMin)

    # 示例：用遗传算法生成配方
    toolbox = base.Toolbox()
    toolbox.register("attr_float", np.random.uniform, 0.01, 0.5)  # 设置最小值为0.01，避免负数
    toolbox.register("individual", tools.initRepeat, creator.Individual, toolbox.attr_float, n=len(feature_names))
    toolbox.register("population", tools.initRepeat, list, toolbox.individual)

    def evaluate(individual):
        # 将个体（配方）转换为字典形式
        user_input = dict(zip(feature_names, individual))
        
        # 保证配方总和为100，必要时进行调整
        total = sum(user_input.values())
        if total != 100:
            user_input = {k: (v / total) * 100 for k, v in user_input.items()}  # 归一化为质量分数

        # 使用模型进行LOI预测
        input_array = np.array([list(user_input.values())])
        input_scaled = scaler.transform(input_array)
        predicted_loi = model.predict(input_scaled)[0]
        
        # 返回LOI与目标LOI之间的差异，作为目标函数值
        return abs(predicted_loi - target_loi),  # 返回元组，符合遗传算法的要求

    toolbox.register("mate", tools.cxBlend, alpha=0.5)
    toolbox.register("mutate", tools.mutGaussian, mu=0, sigma=0.1, indpb=0.2)
    toolbox.register("select", tools.selTournament, tournsize=3)
    toolbox.register("evaluate", evaluate)

    population = toolbox.population(n=50)
    
    # 开始推荐配方按钮
    if st.button("开始推荐配方"):
        # 使用遗传算法生成配方
        for gen in range(10):  # 10代
            offspring = toolbox.select(population, len(population))
            offspring = list(map(toolbox.clone, offspring))
            for child1, child2 in zip(offspring[::2], offspring[1::2]):
                if np.random.rand() < 0.7:
                    toolbox.mate(child1, child2)
                    del child1.fitness.values
                    del child2.fitness.values
            for mutant in offspring:
                if np.random.rand() < 0.2:
                    toolbox.mutate(mutant)
                    del mutant.fitness.values
            invalid_individuals = [ind for ind in offspring if not ind.fitness.valid]
            fitnesses = list(map(toolbox.evaluate, invalid_individuals))
            for ind, fit in zip(invalid_individuals, fitnesses):
                ind.fitness.values = fit
            population[:] = offspring

        # 从最后一代中选出最好的配方
        best_individual = tools.selBest(population, 1)[0]
        st.write("最佳配方:", dict(zip(feature_names, best_individual)))

