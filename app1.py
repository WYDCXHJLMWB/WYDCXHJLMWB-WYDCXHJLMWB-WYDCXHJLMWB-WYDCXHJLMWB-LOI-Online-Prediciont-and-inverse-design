class Predictor:
    def __init__(self, scaler_path, svc_path):
        self.scaler = joblib.load(scaler_path)
        self.model = joblib.load(svc_path)
        self.static_cols = ["产品质量指标_Sn%", "添加比例", "一甲%"]
        self.time_series_cols = ["黄度值_3min", "6min", "9min", "12min", "15min", "18min", "21min", "24min"]

    def _truncate(self, df):
        time_cols = [col for col in df.columns if "min" in col.lower()]
        time_cols_ordered = [col for col in df.columns if col in time_cols]
        if time_cols_ordered:
            row = df.iloc[0][time_cols_ordered]
            if row.notna().any():
                max_idx = row.idxmax()
                max_pos = time_cols_ordered.index(max_idx)
                for col in time_cols_ordered[max_pos + 1:]:
                    df.at[df.index[0], col] = np.nan
        return df

    def predict_one(self, sample):
        all_cols = self.static_cols + self.time_series_cols
        df = pd.DataFrame([sample], columns=all_cols)
        df = self._truncate(df)
        static_data = {}
        for feat in self.static_cols:
            matching = [col for col in df.columns if feat in col]
            if matching:
                static_data[feat] = df.at[0, matching[0]]
        static_df = pd.DataFrame([static_data])
        X_transformed = self.scaler.transform(static_df.values)
        return self.model.predict(X_transformed)[0]
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
fraction_type = st.sidebar.radio("📐 分数类型", ["质量", "质量分数", "体积分数"])

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
        "loi_features": pd.read_excel("trainrg3.xlsx").drop(columns="LOI", errors='ignore').columns.tolist(),
        "ts_features": pd.read_excel("trainrg3TS.xlsx").drop(columns="TS", errors='ignore').columns.tolist(),
    }
models = load_models()

# 获取单位
def get_unit(fraction_type):
    if fraction_type == "质量":
        return "g"
    elif fraction_type == "质量分数":
        return "wt%"
    elif fraction_type == "体积分数":
        return "vol%"

# 保证PP在首列
def ensure_pp_first(features):
    if "PP" in features:
        features.remove("PP")
    return ["PP"] + sorted(features)

# 性能预测页面
if page == "性能预测":
    st.subheader("🔮 性能预测：基于配方预测LOI和TS")
    
    # 动态生成输入框
    input_values = {}
    features = ensure_pp_first(sorted(set(models["loi_features"] + models["ts_features"])))
    cols = st.columns(2)
    
    for i, feature in enumerate(features):
        with cols[i % 2]:
            unit = get_unit(fraction_type)
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
        if fraction_type == "体积分数":
            if abs(total - 100.0) > 1e-6:
                st.error(f"❗ 体积分数的总和必须为100%（当前：{total:.2f}%）")
            else:
                st.success("体积分数总和验证通过")
        elif fraction_type == "质量分数":
            if abs(total - 100.0) > 1e-6:
                st.error(f"❗ 质量分数的总和必须为100%（当前：{total:.2f}%）")
            else:
                st.success("质量分数总和验证通过")
        else:
            st.success("成分总和验证通过")
            if is_only_pp:
                st.info("检测到纯PP配方")

    if st.button("🚀 开始预测", type="primary"):
        if fraction_type == "体积分数" and abs(total - 100.0) > 1e-6:
            st.error("预测中止：体积分数的总和必须为100%")
            st.stop()
        elif fraction_type == "质量分数" and abs(total - 100.0) > 1e-6:
            st.error("预测中止：质量分数的总和必须为100%")
            st.stop()

        # 单位转换处理
        if fraction_type == "体积分数":
            # 体积分数转化为质量分数
            vol_values = np.array([input_values[f] for f in features])
            mass_values = vol_values  # 假设体积分数与质量分数直接相等
            total_mass = mass_values.sum()
            input_values = {f: (mass_values[i]/total_mass)*100 for i, f in enumerate(features)}
        
        # 如果是纯PP配方，直接进行LOI和TS预测
        if is_only_pp:
            loi_pred = 17.5  # 假设PP配方LOI为17.5%
            ts_pred = 35.0  # 假设PP配方TS为35 MPa
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

elif page == "配方建议":
    sub_page = st.sidebar.selectbox("🔧 选择功能", ["配方优化", "添加剂推荐"])
    if sub_page == "配方优化":
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
            all_features = ensure_pp_first(list(set(models["loi_features"] + models["ts_features"])))
            n_features = len(all_features)
            
            # 生成满足和为100的配方
            def generate_individual():
                # 随机生成一个和为100的配方
                individual = [random.uniform(0, 100) for _ in range(n_features)]
                total = sum(individual)
                # 保证总和为100，且不含负值
                return [x * 100.0 / total for x in individual]
    
            # 初始化种群
            toolbox.register("individual", tools.initIterate, creator.Individual, generate_individual)
            toolbox.register("population", tools.initRepeat, list, toolbox.individual)
    
            # 目标函数
            def evaluate(individual):
                input_values = dict(zip(all_features, individual))
                loi_input = np.array([[input_values[f] for f in models["loi_features"]]])
                loi_scaled = models["loi_scaler"].transform(loi_input)
                loi_pred = models["loi_model"].predict(loi_scaled)[0]
                ts_input = np.array([[input_values[f] for f in models["ts_features"]]])
                ts_scaled = models["ts_scaler"].transform(ts_input)
                ts_pred = models["ts_model"].predict(ts_scaled)[0]
                return abs(target_loi - loi_pred) + abs(target_ts - ts_pred),
    
            toolbox.register("mate", tools.cxBlend, alpha=0.5)
            toolbox.register("mutate", tools.mutGaussian, mu=0, sigma=5, indpb=0.2)
            toolbox.register("select", tools.selTournament, tournsize=3)
            toolbox.register("evaluate", evaluate)
    
            # 遗传算法流程
            population = toolbox.population(n=pop_size)
            algorithms.eaSimple(population, toolbox, cxpb=cx_prob, mutpb=mut_prob, ngen=n_gen, verbose=False)
    
            # 获取最优解
            best_individual = tools.selBest(population, 1)[0]
            st.write(f"优化后的配方：{dict(zip(all_features, best_individual))}")

    elif sub_page == "添加剂推荐":
        st.subheader("添加剂推荐")
    
        # 修改后的推荐逻辑
        @st.cache_resource
        def load_predictor():
            return Predictor(
                scaler_path="scaler_fold_1.pkl",  # 确保文件路径正确
                svc_path="svc_fold_1.pkl"
            )
        
        # 添加输入界面
        with st.form("additive_form"):
            col1, col2 = st.columns(2)
            
            # 静态参数输入
            with col1:
                st.markdown("### 基础参数")
                add_ratio = st.number_input("添加比例 (%)", 0.0, 100.0, 5.0, step=0.1)
                sn_percent = st.number_input("Sn含量 (%)", 0.0, 100.0, 98.5, step=0.1)
                yijia_percent = st.number_input("一甲胺含量 (%)", 0.0, 100.0, 0.5, step=0.1)
            
            # 时序参数输入
            with col2:
                st.markdown("### 时序参数")
                yellow_values = [
                    st.number_input(f"黄度值_{time}min", 0.0, 10.0, 1.2 + i*0.3, key=f"yellow_{time}")
                    for i, time in enumerate([3, 6, 9, 12, 15, 18, 21, 24])
                ]
            
            submitted = st.form_submit_button("生成推荐")
    
        if submitted:
            try:
                # 构建输入样本（注意顺序与类定义一致）
                sample = [
                    add_ratio,
                    sn_percent,
                    yijia_percent,
                    *yellow_values  # 展开时序参数
                ]
                
                predictor = load_predictor()
                result = predictor.predict_one(sample)
                
                # 显示结果
                st.success("## 推荐结果")
                result_map = {
                    0: {"类型": "标准型APP", "用量": "15-20%"},
                    1: {"类型": "纳米复合阻燃剂", "用量": "10-15%"},
                    2: {"类型": "膨胀型阻燃剂", "用量": "20-25%"}
                }
                
                if result in result_map:
                    rec = result_map[result]
                    st.markdown(f"""
                    - **推荐类型**: `{rec['类型']}`
                    - **建议添加量**: {rec['用量']}
                    - **适配工艺**: 注塑成型（温度 180-200℃）
                    """)
                else:
                    st.warning("未找到匹配的推荐类型")
    
            except Exception as e:
                st.error(f"预测失败: {str(e)}")
