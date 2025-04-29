import pandas as pd
import numpy as np
import warnings
from scipy import stats
from scipy.stats import kurtosis, skew
from sklearn.impute import SimpleImputer
import joblib

class Predictor:
    def __init__(self, scaler_path, svc_path):
        self.scaler = joblib.load(scaler_path)
        self.model = joblib.load(svc_path)
        
        # 特征列配置
        self.static_cols = ["产品质量指标_Sn%", "添加比例", "一甲%"]
        self.time_series_cols = [
            "黄度值_3min", "6min", "9min", "12min",
            "15min", "18min", "21min", "24min"
        ]
        self.eng_features = [
            'seq_length', 'max_value', 'mean_value', 'min_value',
            'std_value', 'trend', 'range_value', 'autocorr'
        ]
        self.imputer = SimpleImputer(strategy="mean")

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
    
    def _get_slope(self, row, col=None):
        # col 是可选的，将被忽略
        x = np.arange(len(row))
        y = row.values
        mask = ~np.isnan(y)
        if sum(mask) >= 2:
            return stats.linregress(x[mask], y[mask])[0]
        return np.nan
    def _calc_autocorr(self, row):
        """计算一阶自相关系数"""
        # 去除NaN值
        values = row.dropna().values
        if len(values) > 1:
            # 计算一阶自相关系数
            n = len(values)
            mean = np.mean(values)
            # 计算协方差和方差
            numerator = sum((values[:-1] - mean) * (values[1:] - mean))
            denominator = sum((values - mean) ** 2)
            if denominator != 0:
                return numerator / denominator
        return np.nan

    def _extract_time_series_features(self, df):
        """修复后的时序特征提取"""
        time_data = df[self.time_series_cols]
        time_data_filled = time_data.ffill(axis=1)  # ✅ 沿时间轴填充
        
        features = pd.DataFrame()
        features['seq_length'] = time_data_filled.notna().sum(axis=1)
        features['max_value'] = time_data_filled.max(axis=1)
        features['mean_value'] = time_data_filled.mean(axis=1)
        features['min_value'] = time_data_filled.min(axis=1)
        features['std_value'] = time_data_filled.std(axis=1)
        features['range_value'] = features['max_value'] - features['min_value']
        features['trend'] = time_data_filled.apply(self._get_slope, axis=1)
        features['autocorr'] = time_data_filled.apply(self._calc_autocorr, axis=1)
        return features

    def predict_one(self, sample):
        full_cols = self.static_cols + self.time_series_cols
        df = pd.DataFrame([sample], columns=full_cols)
        df = self._truncate(df)  # ✅ 调用已定义的方法
        
        # 特征合并
        static_features = df[self.static_cols]
        time_features = self._extract_time_series_features(df)
        feature_df = pd.concat([static_features, time_features], axis=1)
        feature_df = feature_df[self.static_cols + self.eng_features]  # ✅ 确保列顺序
        
        # 验证维度
        if feature_df.shape[1] != self.scaler.n_features_in_:
            raise ValueError(f"特征维度不匹配！当前：{feature_df.shape[1]}，需要：{self.scaler.n_features_in_}")
        
        X_scaled = self.scaler.transform(feature_df)
        return self.model.predict(X_scaled)[0]

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
    
    # 定义分类的材料
    matrix_materials = [
        "PP",  "PA","PC/ABS","POM","PBT","PVC","其他"
    ]
    flame_retardants = [
       "AHP"，"ammonium octamolybdate", "Al(OH)3", "CFA", "APP", "Pentaerythritol","DOPO", "EPFR-1100NT", "XS-FR-8310", "ZS", "XiuCheng", "ZHS", "ZnB", "antimony oxides"，"Mg(OH)2", "TCA", "MPP", "PAPP",
    ,"其他"]
    additives = [
        "wollastonite", "M-2200B", "ZBS-PV-OA", "FP-250S",  "silane coupling agent",  "antioxidant"， "SiO2","其他"
    ]
    
    # 用户选择的单位类型
    fraction_type = st.selectbox("选择输入的单位", ["质量", "质量分数", "体积分数"])
    
    # 显示分类选择：基体、阻燃剂和助剂的下拉菜单
    st.subheader("请选择配方中的基体、阻燃剂和助剂")
    
    # 基体、阻燃剂和助剂的下拉菜单
    selected_matrix = st.selectbox("选择基体", matrix_materials)
    selected_flame_retardant = st.selectbox("选择阻燃剂", flame_retardants)
    selected_additive = st.selectbox("选择助剂", additives)
    
    # 输入其他材料的数量（假设按质量分数）
    input_values = {}
    input_values["matrix"] = st.number_input(f"选择 {selected_matrix} 的质量分数 (%)", min_value=0.0, max_value=100.0, value=50.0, step=0.1)
    input_values["flame_retardant"] = st.number_input(f"选择 {selected_flame_retardant} 的质量分数 (%)", min_value=0.0, max_value=100.0, value=20.0, step=0.1)
    input_values["additive"] = st.number_input(f"选择 {selected_additive} 的质量分数 (%)", min_value=0.0, max_value=100.0, value=30.0, step=0.1)
    
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

        # 如果是纯PP配方，直接进行LOI和TS预测
        if is_only_pp:
            loi_pred = 17.5  # 假设PP配方LOI为17.5%
            ts_pred = 35.0  # 假设PP配方TS为35 MPa
        else:
            # 单位转换处理
            if fraction_type == "体积分数":
                # 体积分数转化为质量分数
                vol_values = np.array([input_values[f] for f in ["matrix", "flame_retardant", "additive"]])
                mass_values = vol_values  # 假设体积分数与质量分数直接相等
                total_mass = mass_values.sum()
                input_values = {f: (mass_values[i]/total_mass)*100 for i, f in enumerate(["matrix", "flame_retardant", "additive"])}
            
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

        predictor = load_predictor()

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

                    # 显示推荐结果
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
