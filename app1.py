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
# 性能预测页面
if page == "性能预测":
    st.subheader("🔬 正向预测：配方 → LOI")
    
    # 使用 with st.form() 来创建表单
    with st.form("input_form"):
        user_input = {}
        total = 0
        cols = st.columns(3)
        
        # 选择基体材料
        base_material = st.selectbox("请选择基体材料", ["PP", "PA", "PC/ABS", "POM", "PBT", "PVC", "其他"])
        
        # 用户输入的配方
        for i, name in enumerate(feature_names):
            if name == "PP":
                continue  # PP单独处理，不放在这里
            unit_label = {
                "质量 (g)": "g",
                "质量分数 (wt%)": "wt%",
                "体积分数 (vol%)": "vol%"
            }[unit_type]
            val = cols[i % 3].number_input(
                f"{name} ({unit_label})", 
                value=0.0, 
                step=0.1 if "质量" in unit_type else 0.01,
                key=f"{name}_input_{i}"  # 为每个number_input添加唯一的key
            )
            user_input[name] = val
            total += val

        # 添加PP输入选项，用户选择后输入其量
        pp_value = st.number_input(
            f"PP ({unit_label})", 
            value=0.0, 
            step=0.1 if "质量" in unit_type else 0.01,
            key="PP_input"  # 给PP单独添加key
        )
        user_input["PP"] = pp_value
        total += pp_value
        
        # 提交按钮放在表单里
        submit_button = st.form_submit_button(label="提交")
        
        # 判断配方加和是否为100
        if unit_type != "质量 (g)" and abs(total - 100) > 1e-3:
            st.warning("⚠️ 配方加和不为100，无法预测。请确保总和为100后再进行预测。")

    # 决定是否提交表单
    submitted = st.form_submit_button("📊 开始预测")

    if submitted:
        # 单位转换逻辑：如果单位不是质量，则将配方的比例转换为100为单位
        if unit_type == "质量 (g)" and total > 0:
            user_input = {k: (v/total)*100 for k,v in user_input.items()}
        elif unit_type == "质量分数 (wt%)" and total != 100:
            st.warning("⚠️ 质量分数总和不为100，请检查输入值。")
        elif unit_type == "体积分数 (vol%)" and total != 100:
            st.warning("⚠️ 体积分数总和不为100，请检查输入值。")
        
        # 预测逻辑
        if all(v==0 for k,v in user_input.items() if k!="PP") and user_input.get("PP", 0) == 100:
            st.metric("极限氧指数 (LOI)", "17.5%")  # 如果PP占100%，则返回一个固定值
        else:
            try:
                # 对输入数据进行预处理
                input_array = np.array([list(user_input.values())])
                input_scaled = scaler.transform(input_array)  # 使用标准化器进行标准化
                prediction = model.predict(input_scaled)[0]  # 预测LOI值
                st.metric("极限氧指数 (LOI)", f"{prediction:.2f}%")
            except Exception as e:
                st.error(f"预测过程中出现错误: {e}")

# 配方建议页面
elif page == "配方建议":
    st.subheader("🧪 配方建议：根据性能反推配方")
    target_loi = st.number_input("目标LOI值", min_value=10.0, max_value=50.0, value=25.0, step=0.1)
    
    # 遗传算法配置
    creator.create("FitnessMin", base.Fitness, weights=(-1.0,))
    creator.create("Individual", list, fitness=creator.FitnessMin)
    
    toolbox = base.Toolbox()
    toolbox.register("attr_float", random.uniform, 0.01, 50)  # 初始范围调整
    toolbox.register("individual", tools.initRepeat, creator.Individual, toolbox.attr_float, n=len(feature_names))
    toolbox.register("population", tools.initRepeat, list, toolbox.individual)
    
    def evaluate(individual):
        # 强制PP含量>=50且为最大值
        if individual[0] < 50:
            return (1000,)
        if individual[0] != max(individual):
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
            POP_SIZE = 100
            GEN_NUM = 50
            CXPB = 0.7
            MUTPB = 0.3
            
            pop = toolbox.population(n=POP_SIZE)
            hof = tools.HallOfFame(10)  # 获取10个最佳配方
            stats = tools.Statistics(lambda ind: ind.fitness.values)
            stats.register("avg", np.mean)
            stats.register("min", np.min)
            
            # 进化循环
            for gen in range(GEN_NUM):
                offspring = toolbox.select(pop, len(pop))
                offspring = list(map(toolbox.clone, offspring))
                
                # 交叉
                for child1, child2 in zip(offspring[::2], offspring[1::2]):
                    if random.random() < CXPB:
                        toolbox.mate(child1, child2)
                        # 删除不合规子代
                        del child1.fitness.values
                        del child2.fitness.values
                
                # 变异
                for mutant in offspring:
                    if random.random() < MUTPB:
                        toolbox.mutate(mutant)
                        # 删除不合规子代
                        del mutant.fitness.values
                
                # 评估所有个体
                invalid_individuals = [ind for ind in offspring if not ind.fitness.valid]
                fitnesses = map(toolbox.evaluate, invalid_individuals)
                for ind, fit in zip(invalid_individuals, fitnesses):
                    ind.fitness.values = fit
                
                # 更新种群
                pop[:] = offspring
                hof.update(pop)
            
            # 显示结果
            best_individual = hof[0]
            best_values = [round(val, 2) for val in best_individual]
            st.write(f"最佳配方 (质量分数)：{dict(zip(feature_names, best_values))}")
            st.write(f"目标LOI: {target_loi}%")
