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

# 基体材料选项
base_materials = ["PP", "PA", "PC/ABS", "POM", "PBT", "PVC", "其他"]
# 阻燃剂选项
flame_retardant_options = [
    "PAPP", "DOPO", "APP", "MPP", "XS-HFFR-8332", 
    "ZS", "ZHS", "Al(OH)3", "ZBS-PV-OA", 
    "ammonium octamolybdate", "Mg(OH)2", "antimony oxides", 
    "Pentaerythritol", "XS-FR-8310", "Xiucheng", "其他"
]
# 助剂选项
additive_options = [
    "silane coupling agent", "antioxidant", "EBS", "Anti-drip-agent",
    "ZnB", "CFA", "wollastonite", "TCA", "M-2200B", "其他"
]

# 单位类型处理
unit_type = st.radio("📏 请选择配方输入单位", ["质量 (g)", "质量分数 (wt%)", "体积分数 (vol%)"], horizontal=True, key="unit_type")

# 性能预测页面
if page == "性能预测":
    st.subheader("🔬 正向预测：配方 → LOI")
    
    # 阻燃剂和助剂选择（在表单外）
    flame_retardant_selection = st.multiselect(
        "选择阻燃剂",
        flame_retardant_options,
        key="flame_retardant_selection"
    )
    
    additive_selection = st.multiselect(
        "选择助剂",
        additive_options,
        key="additive_selection"
    )

    # 使用唯一键的表单
    with st.form(key='input_form'):
        user_input = {name: 0.0 for name in feature_names}  # 初始化所有特征为0
        total = 0.0

        # 基体材料选择（只能选一个）
        selected_base = st.multiselect(
            "选择基体材料（只能选一个）",
            base_materials,
            max_selections=1,
            key='base_material_multiselect'
        )

        # 基体材料输入
        if selected_base:
            base_name = selected_base[0]
            unit_label = {
                "质量 (g)": "g",
                "质量分数 (wt%)": "wt%",
                "体积分数 (vol%)": "vol%"
            }[unit_type]
            
            base_value = st.number_input(
                f"{base_name} ({unit_label})",
                value=0.0,
                step=0.1 if "质量" in unit_type else 0.01,
                key=f'base_{base_name}'
            )
            user_input[base_name] = base_value
            total += base_value

        # 阻燃剂输入
        for flame in flame_retardant_selection:
            qty = st.number_input(
                f"{flame} ({unit_label})",
                min_value=0.0,
                value=0.0,
                step=0.1,
                key=f'flame_{flame}'
            )
            user_input[flame] = qty  # 假设特征名称与选项一致
            total += qty

        # 助剂输入
        for additive in additive_selection:
            qty = st.number_input(
                f"{additive} ({unit_label})",
                min_value=0.0,
                value=0.0,
                step=0.1,
                key=f'additive_{additive}'
            )
            user_input[additive] = qty  # 假设特征名称与选项一致
            total += qty

        # 其他成分输入（非基体材料、非阻燃剂、非助剂）
        other_features = [name for name in feature_names 
                        if name not in base_materials 
                        and name not in flame_retardant_options
                        and name not in additive_options]
        
        for name in other_features:
            unit_label = {
                "质量 (g)": "g",
                "质量分数 (wt%)": "wt%",
                "体积分数 (vol%)": "vol%"
            }[unit_type]
            val = st.number_input(
                f"{name} ({unit_label})", 
                value=0.0, 
                step=0.1 if "质量" in unit_type else 0.01,
                key=f'input_{name}'
            )
            user_input[name] = val
            total += val

        # 提交按钮
        submitted = st.form_submit_button("📊 开始预测")

    # 提交后的处理逻辑
    if submitted:
        # 验证单位类型
        if unit_type != "质量 (g)" and abs(total - 100) > 1e-3:
            st.warning("⚠️ 配方加和不为100，无法预测。请确保总和为100后再进行预测。")
        else:
            # 单位转换逻辑
            if unit_type == "质量 (g)" and total > 0:
                user_input = {k: (v/total)*100 for k,v in user_input.items()}
            elif unit_type == "质量分数 (wt%)":
                total_weight = sum(user_input.values())
                user_input = {k: (v/total_weight)*100 for k,v in user_input.items()}
            elif unit_type == "体积分数 (vol%)":
                total_weight = sum(user_input.values())
                user_input = {k: (v/total_weight)*100 for k,v in user_input.items()}

            # 预测逻辑
            input_array = np.array([list(user_input.values())])
            input_scaled = scaler.transform(input_array)
            prediction = model.predict(input_scaled)[0]
            st.metric("极限氧指数 (LOI)", f"{prediction:.2f}%")

# 配方建议页面（保持不变）
elif page == "配方建议":
    st.subheader("🧪 配方建议：根据性能反推配方")
    target_loi = st.number_input("目标LOI值", min_value=10.0, max_value=50.0, value=25.0, step=0.1)
    
    # 遗传算法配置（保持不变）
    creator.create("FitnessMin", base.Fitness, weights=(-1.0,))
    creator.create("Individual", list, fitness=creator.FitnessMin)
    
    toolbox = base.Toolbox()
    toolbox.register("attr_float", random.uniform, 0.01, 50)
    toolbox.register("individual", tools.initRepeat, creator.Individual, toolbox.attr_float, n=len(feature_names))
    toolbox.register("population", tools.initRepeat, list, toolbox.individual)
    
    def evaluate(individual):
        if individual[0] < 50:
            return (1000,)
        if individual[0] != max(individual):
            return (1000,)
            
        total = sum(individual)
        normalized = [x/total*100 for x in individual]
        
        input_array = np.array([normalized])
        input_scaled = scaler.transform(input_array)
        predicted = model.predict(input_scaled)[0]
        
        return (abs(predicted - target_loi),)
    
    toolbox.register("mate", tools.cxBlend, alpha=0.5)
    toolbox.register("mutate", tools.mutGaussian, mu=0, sigma=5, indpb=0.2)
    toolbox.register("select", tools.selTournament, tournsize=3)
    toolbox.register("evaluate", evaluate)
    
    if st.button("生成推荐配方"):
        with st.spinner("🔍 正在优化配方..."):
            POP_SIZE = 100
            GEN_NUM = 50
            CXPB = 0.7
            MUTPB = 0.3
            
            pop = toolbox.population(n=POP_SIZE)
            hof = tools.HallOfFame(10)
            stats = tools.Statistics(lambda ind: ind.fitness.values)
            stats.register("avg", np.mean)
            stats.register("min", np.min)
            
            for gen in range(GEN_NUM):
                offspring = toolbox.select(pop, len(pop))
                offspring = list(map(toolbox.clone, offspring))
                
                for child1, child2 in zip(offspring[::2], offspring[1::2]):
                    if random.random() < CXPB:
                        toolbox.mate(child1, child2)
                        for i in range(len(child1)):
                            child1[i] = max(child1[i], 0.01)
                            child2[i] = max(child2[i], 0.01)
                        del child1.fitness.values
                        del child2.fitness.values
                
                for mutant in offspring:
                    if random.random() < MUTPB:
                        toolbox.mutate(mutant)
                        for i in range(len(mutant)):
                            mutant[i] = max(mutant[i], 0.01)
                        del mutant.fitness.values
                
                invalid_ind = [ind for ind in offspring if not ind.fitness.valid]
                fitnesses = toolbox.map(toolbox.evaluate, invalid_ind)
                for ind, fit in zip(invalid_ind, fitnesses):
                    ind.fitness.values = fit
                
                pop[:] = offspring
                hof.update(pop)
            
            best_individuals = hof[:10]
            
            recipe_list = []
            for best in best_individuals:
                total = sum(best)
                recipe = {name: (val/total)*100 for name, val in zip(feature_names, best)}
                recipe_list.append(recipe)
            
            st.success("✅ 配方优化完成！")
            
            recipe_df = pd.DataFrame(recipe_list)
            recipe_df.index = [f"配方 {i+1}" for i in range(10)]
            
            unit_label = {
                "质量 (g)": "g",
                "质量分数 (wt%)": "wt%",
                "体积分数 (vol%)": "vol%"
            }[unit_type]
            recipe_df.columns = [f"{col} ({unit_label})" for col in recipe_df.columns]
            st.dataframe(recipe_df)
