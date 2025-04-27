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
            val = cols[i%3].number_input(
                f"{name} ({unit_label})", 
                value=0.0, 
                step=0.1 if "质量" in unit_type else 0.01
            )
            user_input[name] = val
            total += val

        submitted = st.form_submit_button("📊 开始预测")

    if submitted:
        if unit_type != "质量 (g)" and abs(total - 100) > 1e-3:
            st.warning("⚠️ 配方加和不为100，无法预测。请确保总和为100后再进行预测。")
        else:
            # 单位转换逻辑
            if unit_type == "质量 (g)" and total > 0:
                user_input = {k: (v/total)*100 for k,v in user_input.items()}
            # 体积分数计算逻辑（基于质量分数比例）
            elif unit_type == "质量分数 (wt%)":
                total_weight = sum(user_input.values())
                user_input = {k: (v/total_weight)*100 for k,v in user_input.items()}
            elif unit_type == "体积分数 (vol%)":
                total_weight = sum(user_input.values())
                user_input = {k: (v/total_weight)*100 for k,v in user_input.items()}


            # 预测逻辑
            if all(v==0 for k,v in user_input.items() if k!="PP") and user_input.get("PP",0)==100:
                st.metric("极限氧指数 (LOI)", "17.5%")
            else:
                input_array = np.array([list(user_input.values())])
                input_scaled = scaler.transform(input_array)
                prediction = model.predict(input_scaled)[0]
                st.metric("极限氧指数 (LOI)", f"{prediction:.2f}%")

elif page == "配方建议":
    st.subheader("🧪 配方建议：根据性能反推配方")
    target_loi = st.number_input("目标LOI值", min_value=10.0, max_value=50.0, value=25.0, step=0.1)
    
    # 修复1：确保DEAP creator只创建一次
    if 'FitnessMin' not in creator.__dict__:
        creator.create("FitnessMin", base.Fitness, weights=(-1.0,))
    if 'Individual' not in creator.__dict__:
        creator.create("Individual", list, fitness=creator.FitnessMin)
    
    # 遗传算法配置
    toolbox = base.Toolbox()
    toolbox.register("attr_float", random.uniform, 0.01, 50)
    toolbox.register("individual", tools.initRepeat, creator.Individual, toolbox.attr_float, n=len(feature_names))
    toolbox.register("population", tools.initRepeat, list, toolbox.individual)
    toolbox.register("select", tools.selTournament, tournsize=3)
    # 注册交叉操作 (使用两点交叉)
    toolbox.register("mate", tools.cxTwoPoint)
    # 注册变异操作 (使用高斯变异)
    toolbox.register("mutate", tools.mutGaussian, mu=0, sigma=1.0, indpb=0.2)

    def evaluate(individual):
        # 强制PP含量>=50且为最大值
        pp_index = feature_names.index("PP")  # 获取PP的索引
        if individual[pp_index] < 50:
            return (1000,)  # 如果PP含量小于50，则直接返回较大的惩罚值
        if individual[pp_index] != max(individual):
            return (1000,)  # 如果PP含量不是最大值，返回惩罚值
    
        # 归一化处理，确保总和为100
        total = sum(individual)
        normalized = [x/total*100 for x in individual]
    
        # 预测LOI
        input_array = np.array([normalized])
        input_scaled = scaler.transform(input_array)  # 对输入进行缩放
        predicted = model.predict(input_scaled)[0]  # 获取预测结果
    
        # 如果预测结果无效，则返回较大的惩罚值
        if predicted is None or isinstance(predicted, str):
            return (1000,)
        
        return (abs(predicted - target_loi),)  # 返回预测值与目标LOI值的差
    toolbox.register("evaluate", evaluate)  # 这里注册evaluate函数
    # 配方生成部分（确保PP含量强制≥50）
    if st.button("生成推荐配方"):
        with st.spinner("🔍 正在优化配方..."):
            # 初始化最优个体
            hof = tools.HallOfFame(1)
    
            # 遗传算法参数
            POP_SIZE = 100
            GEN_NUM = 50
            CXPB = 0.7
            MUTPB = 0.3
    
            pop = toolbox.population(n=POP_SIZE)
            stats = tools.Statistics(lambda ind: ind.fitness.values)
            stats.register("avg", np.mean)
            stats.register("min", np.min)
    
            # 使用遗传算法简化流程
            algorithms.eaSimple(pop, toolbox, cxpb=CXPB, mutpb=MUTPB, ngen=GEN_NUM, 
                               stats=stats, halloffame=hof, verbose=False)
    
            # 获取最佳个体并转换为配方
            best = hof[0]
            total = sum(best)
            recipe_wt = {name: (val/total)*100 for name, val in zip(feature_names, best)}
    
            # 根据单位类型处理配方
            if unit_type == "质量 (g)":
                recipe = recipe_wt  # 数值直接显示为克数（假设总质量100g）
                unit_label = "g"
            elif unit_type == "质量分数 (wt%)":
                recipe = recipe_wt
                unit_label = "wt%"
            elif unit_type == "体积分数 (vol%)":
                recipe = recipe_wt  # 假设体积分数与质量分数数值相同
                unit_label = "vol%"
    
            # 创建配方DataFrame
            columns_with_units = [f"{name} ({unit_label})" for name in feature_names]
            recipe_df = pd.DataFrame([recipe]*10, columns=columns_with_units)
    
            # 展示结果
            st.write("优化后的配方：")
            st.write(recipe_df)
