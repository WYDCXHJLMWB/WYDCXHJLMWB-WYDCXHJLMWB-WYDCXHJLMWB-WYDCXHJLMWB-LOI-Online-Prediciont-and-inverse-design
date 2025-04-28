import streamlit as st
import pandas as pd
import numpy as np
import joblib
from sklearn.preprocessing import StandardScaler
from deap import base, creator, tools, algorithms
import random
import base64

# 辅助函数：图片转base64
def image_to_base64(image_path):
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode()

# 页面配置
image_path = "图片1.png"
icon_base64 = image_to_base64(image_path)
st.set_page_config(
    page_title="聚丙烯LOI模型",
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

# 加载模型和数据
data = joblib.load("model_and_scaler_loi.pkl")
model = data["model"]
scaler = data["scaler"]
df = pd.read_excel("trainrg3.xlsx")
feature_names = df.columns.tolist()
if "LOI" in feature_names:
    feature_names.remove("LOI")

# 单位类型处理
unit_type = st.radio("📏 请选择配方输入单位", ["质量 (g)", "质量分数 (wt%)", "体积分数 (vol%)"], horizontal=True)

# 性能预测页面
if page == "性能预测":
    st.subheader("🔮 性能预测：基于配方预测LOI")
    user_input = {}

    for feature in feature_names:
        user_input[feature] = st.number_input(f"请输入 {feature} 的值", value=0.0, step=0.1)

    # 性能预测按钮
    if st.button("预测LOI"):
        input_data = np.array([list(user_input.values())])
        input_scaled = scaler.transform(input_data)
        predicted_loi = model.predict(input_scaled)[0]
        st.success(f"预测的LOI值为：{predicted_loi:.2f}")
    
# 配方建议页面
elif page == "配方建议":
    st.subheader("🧪 配方建议：根据性能反推配方")
    target_loi = st.number_input("目标LOI值", min_value=10.0, max_value=50.0, value=25.0, step=0.1)
    
    # 遗传算法配置
    creator.create("FitnessMin", base.Fitness, weights=(-1.0,))  
    creator.create("Individual", list, fitness=creator.FitnessMin)
    
    toolbox = base.Toolbox()
    toolbox.register("attr_float", random.uniform, 0.01, 50)
    toolbox.register("individual", tools.initRepeat, creator.Individual, toolbox.attr_float, n=len(feature_names))
    toolbox.register("population", tools.initRepeat, list, toolbox.individual)
    
    def evaluate(individual):
        # 强制PP含量>=50且为最大值
        pp_index = feature_names.index("PP")
        if individual[pp_index] < 50:
            return (1000,)
        if individual[pp_index] != max(individual):
            return (1000,)
            
        # 归一化处理
        total = sum(individual)
        normalized = [x/total*100 for x in individual]
        
        # 预测LOI
        input_array = np.array([normalized])
        input_scaled = scaler.transform(input_array)
        predicted = model.predict(input_scaled)[0]
        
        return (abs(predicted - target_loi),)
    
    # 遗传算法操作配置
    toolbox.register("mate", tools.cxBlend, alpha=0.5)
    toolbox.register("mutate", tools.mutGaussian, mu=0, sigma=5, indpb=0.2)
    toolbox.register("select", tools.selTournament, tournsize=3)
    toolbox.register("evaluate", evaluate)
    
    if st.button("生成推荐配方"):
        with st.spinner("🔍 正在优化配方..."):
            # 算法参数
            POP_SIZE = 200  # 增大种群规模
            GEN_NUM = 100   # 增加进化代数
            CXPB = 0.7
            MUTPB = 0.3
            
            pop = toolbox.population(n=POP_SIZE)
            hof = tools.HallOfFame(10)  # 保存前10个最佳个体
            stats = tools.Statistics(lambda ind: ind.fitness.values)
            stats.register("avg", np.mean)
            stats.register("min", np.min)
            
            # 进化循环
            algorithms.eaSimple(pop, toolbox, cxpb=CXPB, mutpb=MUTPB, ngen=GEN_NUM, 
                              stats=stats, halloffame=hof, verbose=False)
            
            # 收集有效配方，确保多样性
            valid_recipes = []
            unique_recipes = set()  # 用于确保配方不同
            
            for ind in hof:
                if ind.fitness.values[0] < 1000:  # 过滤有效解
                    total = sum(ind)
                    recipe = {name: (val/total)*100 for name, val in zip(feature_names, ind)}
                    
                    # 生成配方唯一标识
                    recipe_tuple = tuple(recipe.items())
                    if recipe_tuple not in unique_recipes:
                        unique_recipes.add(recipe_tuple)
                        valid_recipes.append(recipe)
                if len(valid_recipes) >= 10:
                    break
            
            if not valid_recipes:
                st.error("无法找到有效配方，请调整目标值或参数")
            else:
                st.success(f"✅ 找到 {len(valid_recipes)} 个有效配方！")
                
                # 生成结果表格
                recipe_df = pd.DataFrame(valid_recipes)
                recipe_df.index = [f"配方 {i+1}" for i in range(len(recipe_df))]
                
                # 根据单位类型调整显示
                unit_label = {
                    "质量 (g)": "g",
                    "质量分数 (wt%)": "wt%",
                    "体积分数 (vol%)": "vol%"
                }[unit_type]
                
                # 单位转换处理：直接使用质量分数作为体积分数
                if unit_type == "体积分数 (vol%)":
                    # 体积分数即为质量分数的比例
                    for name in feature_names:
                        recipe_df[name] = recipe_df[name]  # 体积分数等于质量分数
                
                recipe_df.columns = [f"{name} ({unit_label})" for name in feature_names]
                
                st.dataframe(recipe_df)
