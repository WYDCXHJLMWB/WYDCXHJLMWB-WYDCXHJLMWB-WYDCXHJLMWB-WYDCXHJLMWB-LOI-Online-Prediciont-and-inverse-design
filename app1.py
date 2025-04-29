import pandas as pd
import numpy as np
from sklearn.impute import SimpleImputer
import joblib

class Predictor:
    def __init__(self, scaler_path, svc_path):
        self.scaler = joblib.load(scaler_path)
        self.model = joblib.load(svc_path)
        self.static_cols = ["产品质量指标_Sn%", "添加比例", "一甲%"]
        self.time_series_cols = ["黄度值_3min", "6min", "9min", "12min", "15min", "18min", "21min", "24min"]
        self.imputer = SimpleImputer(strategy="mean")

    def _truncate(self, df):
        ts_data = df[self.time_series_cols].iloc[0]
        if ts_data.notna().any():
            last_valid_idx = ts_data.last_valid_index()
            if last_valid_idx:
                truncate_pos = self.time_series_cols.index(last_valid_idx) + 1
                for col in self.time_series_cols[truncate_pos:]:
                    df[col] = np.nan
        return df

    def _extract_time_series_features(self, df):
        ts_series = df[self.time_series_cols].iloc[0].dropna()
        if ts_series.empty:
            raise ValueError("时间序列数据全为空，无法提取特征")
        
        ts_filled = self.imputer.fit_transform(ts_series.values.reshape(-1, 1)).flatten()
        ts_series = pd.Series(ts_filled, index=ts_series.index)
        
        features = {
            'seq_length': len(ts_series),
            'max_value': ts_series.max(),
            'mean_value': ts_series.mean(),
            'min_value': ts_series.min(),
            'std_value': ts_series.std(),
            'trend': (ts_series.iloc[-1] - ts_series.iloc[0]) / len(ts_series),
            'range_value': ts_series.max() - ts_series.min(),
            'autocorr': ts_series.autocorr() if len(ts_series) > 1 else 0
        }
        return pd.DataFrame([features]), None

    def predict_one(self, sample):
        try:
            # 构造输入数据
            df = pd.DataFrame([sample], columns=self.static_cols + self.time_series_cols)
            
            # 预处理
            df = self._truncate(df)
            
            # 提取特征
            static_data = {col: df[col].iloc[0] for col in self.static_cols}
            eng_df, _ = self._extract_time_series_features(df)
            combined = pd.concat([pd.DataFrame([static_data]), eng_df], axis=1)
            
            # 维度验证
            if combined.shape[1] != self.scaler.n_features_in_:
                raise ValueError(
                    f"特征维度不匹配！当前：{combined.shape[1]}，预期：{self.scaler.n_features_in_}\n"
                    f"当前特征：{combined.columns.tolist()}\n"
                    f"预期特征：{self.scaler.feature_names_in_.tolist()}"
                )
            
            # 预测
            X_scaled = self.scaler.transform(combined)
            return self.model.predict(X_scaled)[0]
        except Exception as e:
            print(f"预测失败，错误堆栈：{str(e)}")
            return -1  # 明确返回错误代码
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


# 侧边栏主导航
page = st.sidebar.selectbox(
    "🔧 主功能选择",
    ["性能预测", "配方建议"],
    key="main_nav"
)

# 子功能选择（仅在配方建议时显示）
sub_page = None
if page == "配方建议":
    sub_page = st.sidebar.selectbox(
        "🔧 子功能选择",
        ["配方优化", "添加剂推荐"],
        key="sub_nav"
    )

# 单位类型选择（动态显示）
if page == "性能预测" or (page == "配方建议" and sub_page == "配方优化"):
    fraction_type = st.sidebar.radio(
        "📐 单位类型",
        ["质量", "质量分数", "体积分数"],
        key="unit_type"
    )
else:
    fraction_type = None

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
    sub_page = st.sidebar.selectbox("🔧 选择功能", ["","配方优化", "添加剂推荐"])
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
                return [max(0, x / total * 100) for x in individual]
            
            toolbox.register("individual", tools.initIterate, creator.Individual, generate_individual)
            toolbox.register("population", tools.initRepeat, list, toolbox.individual)
            
            def evaluate(individual):
                # 单位转换处理
                if fraction_type == "体积分数":
                    # 转换为质量分数
                    vol_values = np.array(individual)
                    mass_values = vol_values  # 直接使用体积分数比例表示质量分数
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
                if pp_content < 50:  # PP含量过低惩罚
                    return (1e6,)
                
                # LOI计算
                loi_input = mass_percent[:len(models["loi_features"])]
                loi_scaled = models["loi_scaler"].transform([loi_input])
                loi_pred = models["loi_model"].predict(loi_scaled)[0]
                loi_error = abs(target_loi - loi_pred)
                
                # TS计算
                ts_input = mass_percent[:len(models["ts_features"])]
                ts_scaled = models["ts_scaler"].transform([ts_input])
                ts_pred = models["ts_model"].predict(ts_scaled)[0]
                ts_error = abs(target_ts - ts_pred)
                total = sum(mass_percent)
                if abs(total - 100) > 1e-6:
                    return (1e6,)
                return (loi_error + ts_error,)
            
            toolbox.register("mate", tools.cxBlend, alpha=0.5)
            toolbox.register("mutate", tools.mutGaussian, mu=0, sigma=1, indpb=0.2)
            toolbox.register("select", tools.selTournament, tournsize=3)
            toolbox.register("evaluate", evaluate)
            
            population = toolbox.population(n=pop_size)
            algorithms.eaSimple(population, toolbox, cxpb=cx_prob, mutpb=mut_prob, ngen=n_gen, verbose=False)
            
            # 选择10个配方并确保每个配方的总和为100
            best_individuals = tools.selBest(population, 10)
            best_values = []
            for individual in best_individuals:
                # 确保每个配方的总和为100，并修正负值
                total = sum(individual)
                best_values.append([round(max(0, i / total * 100), 2) for i in individual])
    
            # 输出优化结果
            result_df = pd.DataFrame(best_values, columns=all_features)
            
            # 添加单位列
            units = [get_unit(fraction_type) for _ in all_features]
            result_df.columns = [f"{col} ({unit})" for col, unit in zip(result_df.columns, units)]
            
            st.write(result_df)
    elif sub_page == "添加剂推荐":
        st.subheader("🧪 添加剂智能推荐")
    
        @st.cache_resource
        def load_predictor():
            return Predictor(
                scaler_path="scaler_fold_1.pkl",
                svc_path="svc_fold_1.pkl"
            )
    
        predictor = load_predictor()  # 注意这里修正了拼写错误
    
        # 创建输入表单
        with st.form("additive_form"):
            st.markdown("### 基础参数")
            col_static = st.columns(3)
            with col_static[0]:
                add_ratio = st.number_input("添加比例 (%)", 0.0, 100.0, 5.0, step=0.1)
            with col_static[1]:
                sn_percent = st.number_input("Sn含量 (%)", 0.0, 100.0, 98.5, step=0.1)
            with col_static[2]:
                yijia_percent = st.number_input("一甲胺含量 (%)", 0.0, 100.0, 0.5, step=0.1)
    
            st.markdown("### 时序参数（黄度值随时间变化）")
    
            time_points = [
                ("3min", 1.2), ("6min", 1.5), ("9min", 1.8),
                ("12min", 2.0), ("15min", 2.2), ("18min", 2.5),
                ("21min", 2.8), ("24min", 3.0)
            ]
    
            yellow_values = {}
            cols = st.columns(4)
            for idx, (time, default) in enumerate(time_points):
                with cols[idx % 4]:
                    yellow_values[time] = st.number_input(
                        f"{time} 黄度值",
                        min_value=0.0,
                        value=default,
                        step=0.1,
                        key=f"yellow_{time}"
                    )
    
            submitted = st.form_submit_button("生成推荐方案")
    
            if submitted:
                try:
                    # 构建输入样本（顺序与类定义一致）
                    sample = [
                        sn_percent,    # 对应 static_cols[0]
                        add_ratio,     # 对应 static_cols[1]
                        yijia_percent, # 对应 static_cols[2]
                        yellow_values["3min"],
                        yellow_values["6min"],
                        yellow_values["9min"],
                        yellow_values["12min"],
                        yellow_values["15min"],
                        yellow_values["18min"],
                        yellow_values["21min"],
                        yellow_values["24min"]
                    ]
    
                    prediction = predictor.predict_one(sample)
    
                    # 结果映射表
                    result_map = {
                        1: {"name": "无"},
                        2: {"name": "氯化石蜡"},
                        3: {"name": "EA12（脂肪酸复合醇酯）"},
                        4: {"name": "EA15（市售液体钙锌稳定剂）"},
                        5: {"name": "EA16（环氧大豆油）"},
                        6: {"name": "G70L（多官能团的脂肪酸复合酯混合物）"},
                        7: {"name": "EA6（亚磷酸酯）"}
                    }
    
                    st.success("### 推荐结果")
                    col1, col2 = st.columns([1, 2])
                    with col1:
                        st.metric("推荐添加剂类型", result_map[prediction]["name"])
                    with col2:
                        st.markdown(f"""
                        **推荐添加剂**: {result_map[prediction]["name"]}
                        """)
                    
                except Exception as e:
                    st.error(f"预测时发生错误：{str(e)}")

