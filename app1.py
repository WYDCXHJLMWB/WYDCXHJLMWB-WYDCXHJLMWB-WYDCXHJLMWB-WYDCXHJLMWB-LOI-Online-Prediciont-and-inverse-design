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
        error_flag = False
        # 输入验证和单位转换
        try:
            if unit_type == "质量 (g)":
                total_mass = sum(user_input.values())
                if total_mass <= 0:
                    st.error("总质量必须大于0")
                    error_flag = True
                else:
                    # 转换为质量分数
                    user_input = {k: (v/total_mass)*100 for k,v in user_input.items()}
            elif unit_type == "体积分数 (vol%)":
                if abs(total - 100) > 1e-3:
                    st.warning("体积分数总和必须为100%")
                    error_flag = True
                else:
                    # 转换为质量分数
                    masses = [user_input[name] for name in feature_names]  # 直接使用体积分数作为质量分数
                    total_mass = sum(masses)
                    if total_mass <= 0:
                        st.error("总质量计算错误")
                        error_flag = True
                    else:
                        user_input = {name: (masses[i]/total_mass)*100 for i, name in enumerate(feature_names)}
            else:  # 质量分数 (wt%)
                if abs(total - 100) > 1e-3:
                    st.warning("质量分数总和必须为100%")
                    error_flag = True

            if not error_flag:
                # 预测逻辑
                if all(v==0 for k,v in user_input.items() if k!="PP") and user_input.get("PP",0)==100:
                    st.metric("极限氧指数 (LOI)", "17.5%")
                else:
                    input_array = np.array([list(user_input.values())])
                    input_scaled = scaler.transform(input_array)
                    prediction = model.predict(input_scaled)[0]
                    st.metric("极限氧指数 (LOI)", f"{prediction:.2f}%")
        except Exception as e:
            st.error(f"输入处理错误: {str(e)}")

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
            
            # 收集有效配方
            valid_recipes = []
            for ind in hof:
                if ind.fitness.values[0] < 1000:  # 过滤有效解
                    total = sum(ind)
                    recipe = {name: (val/total)*100 for name, val in zip(feature_names, ind)}
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
