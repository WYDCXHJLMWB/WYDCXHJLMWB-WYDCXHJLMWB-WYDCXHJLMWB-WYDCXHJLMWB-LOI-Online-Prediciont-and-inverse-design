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
        "loi_features": [f for f in pd.read_excel("trainrg3.xlsx").columns if f != "LOI"],
        "ts_features": [f for f in pd.read_excel("trainrg3TS.xlsx").columns if f != "TS"]
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
            input_values[feature] = st.number_input(
                f"{feature} ({fraction_type})",
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
            
        if is_only_pp:
            st.success(f"预测LOI值：17.5%")
            st.success(f"预测TS值：35.0 MPa")
        else:
            # 按特征名称来选取LOI相关特征
            loi_input = np.array([[input_values[f] for f in models["loi_features"]]])
            loi_scaled = models["loi_scaler"].transform(loi_input)
            loi_pred = models["loi_model"].predict(loi_scaled)[0]
            
            # 按特征名称来选取TS相关特征
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
        n_features = len(models["loi_features"])
        toolbox.register("attr_float", random.uniform, 0.1, 100)
        toolbox.register("individual", tools.initRepeat, creator.Individual, toolbox.attr_float, n=n_features)
        toolbox.register("population", tools.initRepeat, list, toolbox.individual)
        
        def evaluate(individual):
            # 归一化处理
            total = sum(individual)
            if total == 0:
                return (1e6,)
            normalized = [x/total*100 for x in individual]
            
            # PP约束
            pp_index = models["loi_features"].index("PP")
            pp_content = normalized[pp_index]
            if pp_content < 50 or pp_content != max(normalized):
                return (1e6,)
            
            # LOI预测
            loi_input = np.array([normalized])
            loi_scaled = models["loi_scaler"].transform(loi_input)
            loi_pred = models["loi_model"].predict(loi_scaled)[0]
            
            # TS预测
            ts_scaled = models["ts_scaler"].transform(loi_input)
            ts_pred = models["ts_model"].predict(ts_scaled)[0]
            
            # 适应度计算
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
            total = sum(ind)
            if total == 0:
                continue
            normalized = [x/total*100 for x in ind]
            if abs(sum(normalized) - 100) > 1e-6:
                continue
            
            # 转换为字典
            solution = {name: f"{val:.2f}" for name, val in zip(models["loi_features"], normalized)}
            solution["LOI"] = f"{evaluate(ind)[0]/2 + target_loi:.2f}"
            solution["TS"] = f"{target_ts - evaluate(ind)[0]/2:.2f}"
            solutions.append(solution)
        
        if solutions:
            df = pd.DataFrame(solutions)
            df = df[["PP"] + [c for c in df.columns if c not in ["PP", "LOI", "TS"]] + ["LOI", "TS"]]
            
            st.subheader("🏆 推荐配方列表")
            st.dataframe(df.style.format({
                **{col: "{:.2f}%" for col in models["loi_features"]},
                "LOI": "{:.2f}%",
                "TS": "{:.2f} MPa"
            }), height=600)
            
            # 添加下载按钮
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
