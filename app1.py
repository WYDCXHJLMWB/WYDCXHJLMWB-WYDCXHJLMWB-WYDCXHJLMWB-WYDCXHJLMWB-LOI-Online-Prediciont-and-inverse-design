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

# 配方建议页面
elif page == "配方建议":
    st.subheader("🧪 配方建议：根据性能反推配方")
    target_loi = st.number_input("目标LOI值", min_value=10.0, max_value=50.0, value=25.0, step=0.1)
    unit_type = st.radio("📏 请选择配方显示单位", ["质量 (g)", "质量分数 (wt%)", "体积分数 (vol%)"], horizontal=True)

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

    if st.button("生成推荐配方"):
        with st.spinner("🔍 正在优化配方..."):
            # 进化算法参数优化
            POP_SIZE = 200
            GEN_NUM = 100
            CXPB = 0.8
            MUTPB = 0.4

            pop = toolbox.population(n=POP_SIZE)
            hof = tools.HallOfFame(10)
            stats = tools.Statistics(lambda ind: ind.fitness.values)
            stats.register("avg", np.mean)
            stats.register("min", np.min)

            # 使用eaSimple算法并增加多样性
            algorithms.eaSimple(pop, toolbox, cxpb=CXPB, mutpb=MUTPB, ngen=GEN_NUM, 
                               stats=stats, halloffame=hof, verbose=False)

            # 收集有效配方
            valid_recipes = []
            for ind in hof + pop:  # 合并精英和种群
                pp_index = feature_names.index("PP")
                if ind[pp_index] >= 50 and ind[pp_index] == max(ind):
                    total = sum(ind)
                    normalized = [val/total*100 for val in ind]
                    valid_recipes.append(normalized)
                
                if len(valid_recipes) >= 30:  # 收集足够样本用于去重
                    break

            # 去重处理
            unique_recipes = []
            seen = set()
            for recipe in valid_recipes:
                key = tuple(round(x, 2) for x in recipe)
                if key not in seen:
                    seen.add(key)
                    unique_recipes.append(recipe)
                if len(unique_recipes) >= 10:
                    break

            # 单位转换处理
            converted_data = []
            for recipe in unique_recipes[:10]:  # 取前10个
                total_mass = 100  # 假设总质量为100g
                mass_percent = {name: val for name, val in zip(feature_names, recipe)}
                
                # 体积分数计算（根据质量分数比例）
                volume_percent = {}
                if "vol" in unit_type:
                    total_mass = sum(mass_percent.values())
                    for name, percent in mass_percent.items():
                        volume_percent[name] = percent * (percent / total_mass)
                    total_volume = sum(volume_percent.values())
                    for name in volume_percent:
                        volume_percent[name] = volume_percent[name] / total_volume * 100
                
                # 根据单位类型转换
                converted = {}
                for name in feature_names:
                    if unit_type == "质量 (g)":
                        converted[name] = mass_percent[name] * total_mass / 100
                    elif unit_type == "质量分数 (wt%)":
                        converted[name] = mass_percent[name]
                    else:  # 体积分数
                        converted[name] = volume_percent.get(name, mass_percent[name])
                
                converted_data.append(converted)

            # 创建DataFrame
            unit_label = {
                "质量 (g)": "g",
                "质量分数 (wt%)": "wt%",
                "体积分数 (vol%)": "vol%"
            }[unit_type]
            
            columns = [f"{name} ({unit_label})" for name in feature_names]
            recipe_df = pd.DataFrame(converted_data, columns=columns)
            recipe_df.index = [f"配方 {i+1}" for i in range(recipe_df.shape[0])]

            # 展示数据表
            st.subheader("推荐配方")
            st.dataframe(recipe_df)
