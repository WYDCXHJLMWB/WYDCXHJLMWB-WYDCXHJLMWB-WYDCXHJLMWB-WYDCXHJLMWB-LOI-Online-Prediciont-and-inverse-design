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
    page_title="聚丙烯性能预测平台",
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
        聚合物复合材料智能设计平台
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
    unit_type = st.radio("📏 请选择配方输入单位", 
                       ["质量 (g)", "质量分数 (wt%)", "体积分数 (vol%)"], 
                       horizontal=True, 
                       key="unit_type")
    
    st.subheader("🔬 正向预测：配方 → 性能")
    
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

    with st.form(key='input_form'):
        user_input = {name: 0.0 for name in feature_names}
        total = 0.0

        unit_label = {
            "质量 (g)": "g",
            "质量分数 (wt%)": "wt%",
            "体积分数 (vol%)": "vol%"
        }[unit_type]

        selected_base = st.selectbox(
            "选择基体材料（必选）",
            base_materials,
            index=None,
            placeholder="请选择基体材料...",
            key='base_material_select'
        )

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

        for flame in flame_retardant_selection:
            qty = st.number_input(
                f"{flame} ({unit_label})",
                min_value=0.0,
                value=0.0,
                step=0.1,
                key=f'flame_{flame}'
            )
            user_input[flame] = qty
            total += qty

        for additive in additive_selection:
            qty = st.number_input(
                f"{additive} ({unit_label})",
                min_value=0.0,
                value=0.0,
                step=0.1,
                key=f'additive_{additive}'
            )
            user_input[additive] = qty
            total += qty

        other_features = [name for name in feature_names 
                        if name not in base_materials 
                        and name not in flame_retardant_options
                        and name not in additive_options]
        
        for name in other_features:
            val = st.number_input(
                f"{name} ({unit_label})", 
                value=0.0, 
                min_value=0.0,
                step=0.1 if "质量" in unit_type else 0.01,
                key=f'input_{name}'
            )
            user_input[name] = val
            total += val

        submitted = st.form_submit_button("📊 开始预测")

        if submitted:
            if not selected_base:
                st.error("❌ 必须选择基体材料")
            elif unit_type != "质量 (g)" and abs(total - 100) > 1e-3:
                st.warning("⚠️ 配方加和不为100，请确保总和为100后再进行预测。")
            else:
                if unit_type == "质量 (g)" and total > 0:
                    user_input = {k: (v/total)*100 for k,v in user_input.items()}
                
                input_array = np.array([list(user_input.values())])  # 修复了缺少的圆括号
                
                try:
                    prediction = model.predict(input_array)[0]
                    # 确保预测值非负
                    prediction = max(prediction, 0.0)
                    st.metric("预测性能指标", f"{prediction:.2f}")
                    
                except Exception as e:
                    st.error(f"预测失败: {str(e)}")

# 配方建议页面
elif page == "配方建议":
    st.subheader("🧪 配方建议：根据性能反推配方")
    
    # 新增单位选择
    unit_type_for_recipe = st.radio("📏 请选择目标值单位", 
                                   ["质量 (g)", "质量分数 (wt%)", "体积分数 (vol%)"], 
                                   horizontal=True, 
                                   key="unit_type_for_recipe")

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
        with st.spinner("🔍 正在计算配方建议..."):
            population = toolbox.population(n=100)
            algorithms.eaSimple(population, toolbox, cxpb=0.7, mutpb=0.2, ngen=10, verbose=True)
            best = tools.selBest(population, 1)[0]
            
            st.subheader("最佳配方建议")
            st.write("配方中各材料百分比（总和为100%）：")
            for name, value in zip(feature_names, best):
                st.write(f"{name}: {value:.2f}%")
