import streamlit as st
import pandas as pd
import numpy as np
import joblib
import base64
import random
from deap import base, creator, tools, algorithms

# 辅助函数：图片转base64
def image_to_base64(image_path):
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode()

# 页面配置
image_path = "图片1.png"
icon_base64 = image_to_base64(image_path)
st.set_page_config(
    page_title="聚丙烯LOI和TS模型",
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
fraction_type = st.sidebar.radio("📐 分数类型", ["质量分数", "体积分数"])

# 加载模型
@st.cache_resource
def load_models():
    loi_data = joblib.load("model_and_scaler_loi.pkl")
    ts_data = joblib.load("model_and_scaler_ts1.pkl")
    return {
        "loi_model": loi_data["model"],
        "loi_scaler": loi_data["scaler"],
        "ts_model": ts_data["model"],
        "ts_scaler": ts_data["scaler"],
        "loi_features": pd.read_excel("trainrg3.xlsx").drop(columns="LOI").columns.tolist(),
        "ts_features": pd.read_excel("trainrg3TS.xlsx").drop(columns="TS").columns.tolist(),
        "density": {  # 材料密度字典 (g/cm³)
            'PP': 0.9,
            'APP': 1.2,
            'FR': 1.8,
            'PER': 1.3,
            'MC': 1.4
        }
    }
models = load_models()

# 性能预测页面
if page == "性能预测":
    st.subheader("🔮 性能预测：基于配方预测LOI和TS")
    
    # 动态生成输入框
    input_values = {}
    features = sorted(set(models["loi_features"] + models["ts_features"]))
    cols = st.columns(2)
    
    for i, feature in enumerate(features):
        with cols[i % 2]:
            unit = "wt%" if fraction_type == "质量分数" else "vol%"
            input_values[feature] = st.number_input(
                f"{feature} ({unit})",
                min_value=0.0,
                max_value=100.0,
                value=50.0 if feature == "PP" else 0.0,
                step=0.1
            )

    # 输入验证
    total = sum(input_values.values())
    is_only_pp = all(v == 0 for k, v in input_values.items() if k != "PP")
    
    with st.expander("✅ 输入验证"):
        if abs(total - 100.0) > 1e-6:
            st.error(f"❗ 成分总和必须为100%（当前：{total:.2f}%）")
        else:
            st.success("成分总和验证通过")
            if is_only_pp:
                st.info("检测到纯PP配方")

    if st.button("🚀 开始预测", type="primary"):
        if abs(total - 100.0) > 1e-6:
            st.error("预测中止：成分总和必须为100%")
            st.stop()
            
        # 单位转换处理
        if fraction_type == "体积分数":
            # 转换为质量分数
            vol_values = np.array([input_values[f] for f in features])
            densities = np.array([models["density"].get(f, 1.0) for f in features])
            mass_values = vol_values * densities
            total_mass = mass_values.sum()
            input_values = {f: (mass_values[i]/total_mass)*100 for i, f in enumerate(features)}
            
        if is_only_pp:
            st.success(f"预测LOI值：17.5%")
            st.success(f"预测TS值：35.0 MPa")
        else:
            # LOI预测
            loi_input = np.array([[input_values[f] for f in models["loi_features"]]])
            loi_scaled = models["loi_scaler"].transform(loi_input)
            loi_pred = models["loi_model"].predict(loi_scaled)[0]
            
            # TS预测
            ts_input = np.array([[input_values[f] for f in models["ts_features"]]])
            ts_scaled = models["ts_scaler"].transform(ts_input)
            ts_pred = models["ts_model"].predict(ts_scaled)[0]
            
            # 显示结果
            col1, col2 = st.columns(2)
            with col1:
                st.metric(label="LOI预测值", value=f"{loi_pred:.2f}%")
            with col2:
                st.metric(label="TS预测值", value=f"{ts_pred:.2f} MPa")

# 配方建议页面
elif page == "配方建议":
    st.subheader("🧪 配方建议：根据性能反推配方")
    
    # 目标输入
    col1, col2 = st.columns(2)
    with col1:
        target_loi = st.number_input("目标LOI值（%）", min_value=10.0, max_value=50.0, value=25.0, step=0.1)
    with col2:
        target_ts = st.number_input("目标TS值（MPa）", min_value=0.0, max_value=100.0, value=50.0, step=0.1)
    
    # 遗传算法参数
    with st.expander("⚙️ 算法参数设置"):
        pop_size = st.number_input("种群数量", 50, 500, 200)
        n_gen = st.number_input("迭代代数", 10, 100, 50)
        cx_prob = st.slider("交叉概率", 0.1, 1.0, 0.7)
        mut_prob = st.slider("变异概率", 0.1, 1.0, 0.2)

    if st.button("🔍 开始优化", type="primary"):
        # 初始化遗传算法
        creator.create("FitnessMin", base.Fitness, weights=(-1.0,))
        creator.create("Individual", list, fitness=creator.FitnessMin)
        
        toolbox = base.Toolbox()
        all_features = list(set(models["loi_features"] + models["ts_features"]))
        n_features = len(all_features)
        
        toolbox.register("attr_float", random.uniform, 0.1, 100)
        toolbox.register("individual", tools.initRepeat, creator.Individual, toolbox.attr_float, n=n_features)
        toolbox.register("population", tools.initRepeat, list, toolbox.individual)
        
        def evaluate(individual):
            # 单位转换处理
            if fraction_type == "体积分数":
                # 转换为质量分数
                vol_values = np.array(individual)
                densities = np.array([models["density"].get(f, 1.0) for f in all_features])
                mass_values = vol_values * densities
                total_mass = mass_values.sum()
                if total_mass == 0:
                    return (1e6,)
                mass_percent = (mass_values / total_mass) * 100
            else:
                total = sum(individual)
                if total == 0:
                    return (1e6,)
                mass_percent = np.array(individual) / total * 100
            
            # PP约束
            pp_index = all_features.index("PP")
            pp_content = mass_percent[pp_index]
            if pp_content < 50 or pp_content != max(mass_percent):
                return (1e6,)
            
            # LOI预测
            loi_input = np.array([[mass_percent[all_features.index(f)] for f in models["loi_features"]]])
            loi_scaled = models["loi_scaler"].transform(loi_input)
            loi_pred = models["loi_model"].predict(loi_scaled)[0]
            
            # TS预测
            ts_input = np.array([[mass_percent[all_features.index(f)] for f in models["ts_features"]]])
            ts_scaled = models["ts_scaler"].transform(ts_input)
            ts_pred = models["ts_model"].predict(ts_scaled)[0]
            
            fitness = abs(loi_pred - target_loi) + abs(ts_pred - target_ts)
            return (fitness,)
        
        # 注册遗传算子
        toolbox.register("evaluate", evaluate)
        toolbox.register("mate", tools.cxBlend, alpha=0.5)
        toolbox.register("mutate", tools.mutGaussian, mu=0, sigma=5, indpb=0.2)
        toolbox.register("select", tools.selTournament, tournsize=3)
        
        # 运行算法
        pop = toolbox.population(n=pop_size)
        hof = tools.HallOfFame(10)
        stats = tools.Statistics(lambda ind: ind.fitness.values)
        stats.register("avg", np.mean)
        stats.register("min", np.min)
        
        with st.spinner("🚀 正在优化配方..."):
            algorithms.eaSimple(pop, toolbox, cxpb=cx_prob, mutpb=mut_prob, 
                               ngen=n_gen, stats=stats, halloffame=hof, verbose=False)
        
        # 处理结果
        solutions = []
        for ind in hof:
            # 单位处理
            if fraction_type == "体积分数":
                total_vol = sum(ind)
                if total_vol == 0:
                    continue
                formula = np.array(ind) / total_vol * 100
                unit = "vol%"
            else:
                total_mass = sum(ind)
                if total_mass == 0:
                    continue
                formula = np.array(ind) / total_mass * 100
                unit = "wt%"
            
            # 转换为字典
            solution = {f: f"{formula[i]:.2f}{unit}" for i, f in enumerate(all_features)}
            
            # 预测性能
            loi_input = np.array([[formula[all_features.index(f)] for f in models["loi_features"]]).reshape(1, -1)
            ts_input = np.array([[formula[all_features.index(f)] for f in models["ts_features"]]).reshape(1, -1)
            
            loi_pred = models["loi_model"].predict(models["loi_scaler"].transform(loi_input))[0]
            ts_pred = models["ts_model"].predict(models["ts_scaler"].transform(ts_input))[0]
            
            solution["LOI"] = f"{loi_pred:.2f}%"
            solution["TS"] = f"{ts_pred:.2f} MPa"
            solutions.append(solution)
        
        if solutions:
            df = pd.DataFrame(solutions)
            ordered_columns = ["PP"] + [f for f in all_features if f != "PP"] + ["LOI", "TS"]
            st.subheader("🏆 推荐配方列表")
            st.dataframe(df[ordered_columns].style.format({
                **{f: "{:.2f}" + ("vol%" if fraction_type == "体积分数" else "wt%") for f in all_features},
                "LOI": "{:.2f}%",
                "TS": "{:.2f} MPa"
            }), height=600)
            
            # 下载按钮
            csv = df.to_csv(index=False).encode('utf-8')
            st.download_button(
                label="📥 下载配方数据",
                data=csv,
                file_name="recommended_formulas.csv",
                mime="text/csv"
            )
        else:
            st.warning("⚠️ 未找到满足条件的配方，请尝试：\n"
                      "1. 调整目标值范围\n"
                      "2. 增加迭代代数\n"
                      "3. 扩大种群数量")

# 添加页脚
st.markdown("---")
st.markdown("© 2025 上海大学功能高分子课题组 | 版本 1.1 | [联系我们](#)")
