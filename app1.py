import streamlit as st
import pandas as pd
import numpy as np
from deap import base, creator, tools
import joblib

# 加载模型与缩放器
data = joblib.load("model_and_scaler_loi.pkl")
model = data["model"]
scaler = data["scaler"]

df = pd.read_excel("trainrg3.xlsx")
feature_names = df.columns.tolist()
if "LOI" in feature_names:
    feature_names.remove("LOI")

unit_type = st.radio("📏 请选择配方输入单位", ["质量 (g)", "质量分数 (wt%)", "体积分数 (vol%)"], horizontal=True)

page = st.sidebar.selectbox("🔧 选择功能", ["性能预测", "配方建议"])

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

elif page == "配方建议":
    st.subheader("🧪 配方建议：根据性能反推配方")

    # 用户输入目标LOI值并确保范围在10到50之间
    target_loi = st.number_input("请输入目标极限氧指数 (LOI)", min_value=10.0, max_value=50.0, value=25.0)

    # 如果输入的目标值不在范围内，显示警告
    if target_loi < 10 or target_loi > 50:
        st.warning("⚠️ 请输入10到50之间的有效LOI目标值。")

    # 添加遗传算法的部分（例如）
    creator.create("FitnessMin", base.Fitness, weights=(-1.0,))  # 最小化目标
    creator.create("Individual", list, fitness=creator.FitnessMin)

    # 示例：用遗传算法生成配方
    toolbox = base.Toolbox()
    toolbox.register("attr_float", np.random.uniform, 0, 100)
    toolbox.register("individual", tools.initRepeat, creator.Individual, toolbox.attr_float, n=len(feature_names))
    toolbox.register("population", tools.initRepeat, list, toolbox.individual)

    def evaluate(individual):
        # 假设返回一个简单的LOI估算作为目标函数
        return (sum(individual),)

    toolbox.register("mate", tools.cxBlend, alpha=0.5)
    toolbox.register("mutate", tools.mutGaussian, mu=0, sigma=10, indpb=0.2)
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

        # 获取最佳配方
        best_individuals = tools.selBest(population, 10)  # 至少10个推荐配方
        
        # 修正配方中的负值，确保所有配方的成分都为正，且第一列不为0
        for ind in best_individuals:
            ind[:] = [max(0, value) for value in ind]  # 确保没有负值
            if ind[0] == 0:
                ind[0] = 0.1  # 确保第一列不为0

        st.write("### 推荐的配方:")

        # 将配方展示成表格
        formula_df = pd.DataFrame(best_individuals, columns=feature_names)
        st.dataframe(formula_df)
