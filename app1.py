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
    "Pentaerythritol", "XS-FR-8310", "XiuCheng", "其他"
]
# 助剂选项
additive_options = [
    "silane coupling agent", "antioxidant", "EBS", "Anti-drip-agent",
    "ZnB", "CFA", "wollastonite", "TCA", "M-2200B", "其他"
]

# 性能预测页面
if page == "性能预测":
    # 单位类型处理（仅本页面）
    unit_type = st.radio("📏 请选择配方输入单位", 
                       ["质量 (g)", "质量分数 (wt%)", "体积分数 (vol%)"], 
                       horizontal=True, 
                       key="unit_type")
    
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

        # 根据单位类型确定标签（在所有输入字段之前定义）
        unit_label = {
            "质量 (g)": "g",
            "质量分数 (wt%)": "wt%",
            "体积分数 (vol%)": "vol%"
        }[unit_type]

        # 基体材料选择（只能选一个）
        selected_base = st.selectbox(
            "选择基体材料（必选）",
            base_materials,
            index=None,  # 默认不选择
            placeholder="请选择基体材料...",
            key='base_material_select'
        )

        # 输出当前选择的基体材料，调试用
        st.write("选中的基体材料：", selected_base)

        # 基体材料输入处理
        if selected_base:
            base_value = st.number_input(
                f"{selected_base} ({unit_label})",
                value=0.0,
                min_value=0.0,
                step=0.1 if "质量" in unit_type else 0.01,
                key=f'base_{selected_base}'
            )
            user_input[selected_base] = base_value
            total += base_value
        else:
            st.warning("⚠️ 请选择基体材料")

        # 阻燃剂输入
        for flame in flame_retardant_selection:
            qty = st.number_input(
                f"{flame} ({unit_label})",
                min_value=0.0,  # 防止负值
                value=0.0,
                step=0.1,
                key=f'flame_{flame}'
            )
            user_input[flame] = qty
            total += qty

        # 助剂输入
        for additive in additive_selection:
            qty = st.number_input(
                f"{additive} ({unit_label})",
                min_value=0.0,  # 防止负值
                value=0.0,
                step=0.1,
                key=f'additive_{additive}'
            )
            user_input[additive] = qty
            total += qty

        # 其他成分输入（非基体材料、非阻燃剂、非助剂）
        other_features = [name for name in feature_names 
                        if name not in base_materials 
                        and name not in flame_retardant_options
                        and name not in additive_options]
        
        for name in other_features:
            val = st.number_input(
                f"{name} ({unit_label})", 
                value=0.0, 
                min_value=0.0,  # 防止负值
                step=0.1 if "质量" in unit_type else 0.01,
                key=f'input_{name}'
            )
            user_input[name] = val
            total += val

        # 提交按钮
        submitted = st.form_submit_button("📊 开始预测")

# 在性能预测部分的预测逻辑中做如下修改（约第200行附近）
        if submitted:
            # 基体材料必选验证
            if not selected_base:
                st.error("❌ 必须选择基体材料")
            elif unit_type != "质量 (g)" and abs(total - 100) > 1e-3:
                st.warning("⚠️ 配方加和不为100，无法预测。请确保总和为100后再进行预测。")
            else:
                # 转换为百分比
                if unit_type == "质量 (g)" and total > 0:
                    user_input = {k: (v/total)*100 for k,v in user_input.items()}
                
                # 创建输入数组
                input_array = np.array([list(user_input.values())])
                
                # 获取训练数据统计信息
                train_loi_min = df["LOI"].min()  # 获取训练数据最小LOI值
                train_loi_max = df["LOI"].max()  # 获取训练数据最大LOI值
                
                try:
                    # 直接使用模型预测（假设模型未使用特征缩放）
                    prediction = model.predict(input_array)[0]
                    
                    # 应用物理约束
                    prediction = max(prediction, train_loi_min)  # 确保不低于训练数据最小值
                    prediction = min(prediction, train_loi_max)  # 确保不超过训练数据最大值
                    
                    st.metric("极限氧指数 (LOI)", f"{prediction:.2f}%")
                    
                except Exception as e:
                    st.error(f"预测失败: {str(e)}")
                    st.info("可能原因：输入特征超出模型训练范围")


# 配方建议页面
elif page == "配方建议":
    st.subheader("🧪 配方建议：根据性能反推配方")
    
    # 添加独立的单位选择（仅本页面使用）
    inverse_unit_type = st.radio("📏 请选择配方显示单位", 
                               ["质量 (g)", "质量分数 (wt%)", "体积分数 (vol%)"], 
                               horizontal=True, 
                               key="inverse_unit")
    
    target_loi = st.number_input("目标LOI值", min_value=10.0, max_value=50.0, value=25.0, step=0.1)
    
    # 遗传算法配置
    creator.create("FitnessMin", base.Fitness, weights=(-1.0,)) 
    creator.create("Individual", list, fitness=creator.FitnessMin)
    
    toolbox = base.Toolbox()
    toolbox.register("attr_float", random.uniform, 0.01, 50)
    toolbox.register("individual", tools.initRepeat, creator.Individual, toolbox.attr_float, n=len(feature_names))
    toolbox.register("population", tools.initRepeat, list, toolbox.individual)
    
    def evalFormula(individual):
        return sum(individual), 
    
    toolbox.register("mate", tools.cxBlend, alpha=0.5)
    toolbox.register("mutate", tools.mutGaussian, mu=0.0, sigma=1.0, indpb=0.2)
    toolbox.register("select", tools.selTournament, tournsize=3)
    toolbox.register("evaluate", evalFormula)
    
    population = toolbox.population(n=100)
    generations = 100
    for gen in range(generations):
        offspring = list(map(toolbox.clone, population))
        for child1, child2 in zip(offspring[::2], offspring[1::2]):
            if random.random() < 0.7:
                toolbox.mate(child1, child2)
                del child1.fitness.values
                del child2.fitness.values
        
        for mutant in offspring:
            if random.random() < 0.2:
                toolbox.mutate(mutant)
                del mutant.fitness.values
        
        for individual in offspring:
            if not individual.fitness.valid:
                individual.fitness.values = toolbox.evaluate(individual)
        
        population[:] = offspring
        
    best_individual = tools.selBest(population, 1)[0]
    st.write("建议配方：", best_individual)
