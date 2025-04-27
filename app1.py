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
# 确保特征顺序与训练时一致，这里假设data中保存了特征顺序
feature_names = data["feature_names"]

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
            # 预测逻辑
            input_array = np.array([list(user_input.values())])
            input_scaled = scaler.transform(input_array)
            prediction = model.predict(input_scaled)[0]
            st.metric("极限氧指数 (LOI)", f"{prediction:.2f}%")

elif page == "配方建议":
    st.subheader("🧪 配方建议：根据性能反推配方")
    target_loi = st.number_input("目标LOI值", min_value=10.0, max_value=50.0, value=25.0, step=0.1)
    
    # 确保DEAP creator只创建一次
    if 'FitnessMin' not in creator.__dict__:
        creator.create("FitnessMin", base.Fitness, weights=(-1.0,))
    if 'Individual' not in creator.__dict__:
        creator.create("Individual", list, fitness=creator.FitnessMin)
    
    # 遗传算法配置
    toolbox = base.Toolbox()
    # 调整取值范围为0.01-30，更接近实际配方范围
    toolbox.register("attr_float", random.uniform, 0.01, 30)
    toolbox.register("individual", tools.initRepeat, creator.Individual, toolbox.attr_float, n=len(feature_names))
    toolbox.register("population", tools.initRepeat, list, toolbox.individual)
    
    def evaluate(individual):
        total = sum(individual)
        if total == 0:
            return (1000,)
        normalized = [x/total*100 for x in individual]
        # 检查归一化后的值是否合理
        if any(val < 0 or val > 100 for val in normalized):
            return (1000,)
        # 预测LOI
        input_array = np.array([normalized])
        input_scaled = scaler.transform(input_array)
        predicted = model.predict(input_scaled)[0]
        error = abs(predicted - target_loi)
        return (error,)

    # 遗传算法操作配置
    toolbox.register("mate", tools.cxBlend, alpha=0.5)
    toolbox.register("mutate", tools.mutGaussian, mu=0, sigma=5, indpb=0.1)
    toolbox.register("select", tools.selTournament, tournsize=3)
    toolbox.register("evaluate", evaluate)
    
    if st.button("生成推荐配方"):
        with st.spinner("🔍 正在优化配方..."):
            hof = tools.HallOfFame(1)
            # 调整算法参数
            POP_SIZE = 200
            GEN_NUM = 100
            CXPB = 0.5
            MUTPB = 0.2
            
            pop = toolbox.population(n=POP_SIZE)
            stats = tools.Statistics(lambda ind: ind.fitness.values)
            stats.register("avg", np.mean)
            stats.register("min", np.min)
            
            algorithms.eaSimple(pop, toolbox, cxpb=CXPB, mutpb=MUTPB, ngen=GEN_NUM, 
                               stats=stats, halloffame=hof, verbose=False)
            
            if not hof:
                st.error("未能找到有效配方。")
            else:
                best = hof[0]
                total = sum(best)
                if total == 0:
                    st.error("无效配方，所有成分为零。")
                else:
                    recipe_wt = {name: (val/total)*100 for name, val in zip(feature_names, best)}
                    # 根据单位类型转换
                    if unit_type == "质量 (g)":
                        recipe = {name: val for name, val in recipe_wt.items()}  # 假设总质量100g
                        unit_label = "g"
                    else:
                        recipe = recipe_wt
                        unit_label = "wt%" if unit_type == "质量分数 (wt%)" else "vol%"
                    
                    # 创建DataFrame
                    columns_with_units = [f"{name} ({unit_label})" for name in feature_names]
                    recipe_df = pd.DataFrame([recipe], columns=columns_with_units)
                    recipe_df.index = ["推荐配方"]
                    
                    st.success("✅ 配方优化完成！")
                    st.subheader("推荐配方")
                    st.dataframe(recipe_df.style.format("{:.2f}"))
                    
                    # 显示预测LOI
                    input_array = np.array([[recipe_wt[name] for name in feature_names]])
                    input_scaled = scaler.transform(input_array)
                    predicted_loi = model.predict(input_scaled)[0]
                    st.metric("预测LOI", f"{predicted_loi:.2f}%")
