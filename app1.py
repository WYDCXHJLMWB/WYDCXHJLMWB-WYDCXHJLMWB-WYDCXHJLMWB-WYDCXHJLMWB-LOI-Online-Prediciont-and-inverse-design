import numpy as np
import pandas as pd
import streamlit as st
from deap import base, creator, tools
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression

# 加载模型和Scaler（假设您已经有这些训练好的模型）
model = LinearRegression()  # 假设已经训练好
scaler = StandardScaler()  # 假设已经训练好

# 假设有配方成分特征
feature_names = ['成分1', '成分2', '成分3', '成分4', '成分5']

# 假设目标LOI
target_loi = 50  # 目标LOI值

# 创建适应度和个体类
creator.create("FitnessMin", base.Fitness, weights=(-1.0,))
creator.create("Individual", list, fitness=creator.FitnessMin)

# 设置遗传算法工具箱
toolbox = base.Toolbox()
toolbox.register("attr_float", np.random.uniform, 0.01, 0.5)  # 设置最小值为0.01，避免负数
toolbox.register("individual", tools.initRepeat, creator.Individual, toolbox.attr_float, n=len(feature_names))
toolbox.register("population", tools.initRepeat, list, toolbox.individual)

# 修改适应度评估函数
def evaluate(individual):
    user_input = dict(zip(feature_names, individual))
    
    # 保证配方总和为100，必要时进行调整
    total = sum(user_input.values())
    if total != 100:
        user_input = {k: (v / total) * 100 for k, v in user_input.items()}  # 归一化为质量分数
    
    # 使用模型进行LOI预测
    input_array = np.array([list(user_input.values())])
    input_scaled = scaler.transform(input_array)
    predicted_loi = model.predict(input_scaled)[0]
    
    return abs(predicted_loi - target_loi),

toolbox.register("mate", tools.cxBlend, alpha=0.5)
toolbox.register("mutate", tools.mutGaussian, mu=0, sigma=0.1, indpb=0.2)
toolbox.register("select", tools.selTournament, tournsize=3)
toolbox.register("evaluate", evaluate)

# 修改个体生成方式，确保生成的个体总和为100，且第一列含量最多
def create_individual():
    individual = np.random.uniform(0.01, 0.5, len(feature_names))  # 生成0.01到0.5之间的值
    individual[0] = max(individual[0], 50.0)  # 确保第一列的值大于等于50
    total = sum(individual)
    individual = (individual / total) * 100  # 确保总和为100
    return individual

population = [create_individual() for _ in range(100)]

# 运行遗传算法
for gen in range(100):
    offspring = toolbox.select(population, len(population))
    offspring = list(map(toolbox.clone, offspring))

    for child1, child2 in zip(offspring[::2], offspring[1::2]):
        if np.random.random() < 0.7:  # 70%的概率交叉
            toolbox.mate(child1, child2)
            del child1.fitness.values  # 删除适应度，准备重新评估
            del child2.fitness.values  # 删除适应度，准备重新评估

    for mutant in offspring:
        if np.random.random() < 0.2:  # 20%的概率变异
            toolbox.mutate(mutant)
            del mutant.fitness.values  # 删除适应度，准备重新评估

    # 重新评估个体的适应度
    invalid_individuals = [ind for ind in offspring if not ind.fitness.valid]
    fitnesses = map(toolbox.evaluate, invalid_individuals)
    for ind, fit in zip(invalid_individuals, fitnesses):
        ind.fitness.values = fit

    population[:] = offspring

# 获取最优解并输出为数据框格式
best_individual = tools.selBest(population, 1)[0]
best_result = dict(zip(feature_names, best_individual))

# 将结果转换为数据框
result_df = pd.DataFrame(list(best_result.items()), columns=["成分", "质量分数 (wt%)"])

# 显示配方建议
st.markdown("### 🎯 建议配方")
st.dataframe(result_df)
