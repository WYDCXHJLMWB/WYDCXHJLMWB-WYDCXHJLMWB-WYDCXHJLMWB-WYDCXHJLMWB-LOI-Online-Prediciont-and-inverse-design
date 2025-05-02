import pandas as pd
import numpy as np
import warnings
from scipy import stats
from scipy.stats import kurtosis, skew
from sklearn.impute import SimpleImputer
import joblib


import pandas as pd
import numpy as np
import joblib
from scipy import stats
from sklearn.impute import SimpleImputer
import warnings
warnings.filterwarnings("ignore")

class Predictor:
    def __init__(self, scaler_path, svc_path):
        # 加载预处理对象
        self.scaler = joblib.load(scaler_path)
        self.model = joblib.load(svc_path)
        
        # 定义当前代码期望的特征结构
        self.static_cols = ["产品质量指标_Sn%", "添加比例", "一甲%"]
        self.time_series_cols = [
            "3min", "6min", "9min", "12min",
            "15min", "18min", "21min", "24min"
        ]
        self.eng_features = [
            'seq_length', 'max_value', 'mean_value', 'min_value',
            'std_value', 'trend', 'range_value', 'autocorr'
        ]
        self.expected_columns = self.static_cols + self.eng_features
        self.full_cols = self.static_cols + self.time_series_cols
        
        # 初始化时立即执行检查
        self._validate_components()
        
    def _validate_components(self):
        """核心验证方法"""
        # ================= 特征维度验证 =================
        # 获取标准化器和模型的特征维度
        scaler_features = getattr(self.scaler, "n_features_in_", None)
        model_features = getattr(self.model, "n_features_in_", None)
        
        # 获取当前代码的特征维度
        code_features = len(self.expected_columns)
        
        error_msgs = []
        if scaler_features != code_features:
            error_msgs.append(
                f"标准化器特征维度不匹配！代码: {code_features}，标准化器: {scaler_features}"
            )
        if model_features and model_features != code_features:
            error_msgs.append(
                f"模型特征维度不匹配！代码: {code_features}，模型: {model_features}"
            )
        if error_msgs:
            raise ValueError("\n".join(error_msgs))

        # ================= 特征名称验证 =================
        # 检查模型是否保存了原始特征名
        model_feature_names = getattr(self.model, "feature_names_in_", None)
        if model_feature_names is not None:
            # 对比特征名称和顺序
            if list(model_feature_names) != self.expected_columns:
                msg = [
                    "特征名称或顺序不匹配！",
                    f"模型特征名: {list(model_feature_names)}",
                    f"代码预期: {self.expected_columns}"
                ]
                raise ValueError("\n".join(msg))

        # ================= 虚拟数据测试 =================
        # 生成符合当前代码维度的随机数据
        np.random.seed(42)
        dummy_data = np.random.rand(1, code_features)
        
        try:
            dummy_scaled = self.scaler.transform(dummy_data)
            _ = self.model.predict(dummy_scaled)
        except Exception as e:
            raise RuntimeError(
                f"虚拟数据测试失败！请检查预处理流程：{str(e)}"
            ) from e

    def _truncate(self, df):
        """时间序列截断逻辑"""
        time_cols = self.time_series_cols
        row = df[time_cols].iloc[0]
        
        # 寻找最后一个有效点
        last_valid_idx = next(
            (idx for idx in reversed(range(len(time_cols))) if not pd.isna(row.iloc[idx])),
            None
        )
        
        # 执行截断
        if last_valid_idx is not None and last_valid_idx < len(time_cols)-1:
            invalid_cols = time_cols[last_valid_idx+1:]
            df[invalid_cols] = np.nan
            
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
        values = row.dropna().values
        if len(values) > 1:
            n = len(values)
            mean = np.mean(values)
            numerator = sum((values[:-1] - mean) * (values[1:] - mean))
            denominator = sum((values - mean) ** 2)
            if denominator != 0:
                return numerator / denominator
        return np.nan
    def _extract_time_series_features(self, df):
        """时序特征提取（带列名保护）"""
        time_data = df[self.time_series_cols].ffill(axis=1)
        
        # 显式指定列名
        features = pd.DataFrame({
            'seq_length': time_data.notna().sum(axis=1),
            'max_value': time_data.max(axis=1),
            'mean_value': time_data.mean(axis=1),
            'min_value': time_data.min(axis=1),
            'std_value': time_data.std(axis=1),
            'trend': time_data.apply(self._get_slope, axis=1),
            'range_value': time_data.max(axis=1) - time_data.min(axis=1),
            'autocorr': time_data.apply(self._calc_autocorr, axis=1)
        }, columns=self.eng_features)  # 强制列名
        
        return features

    def predict_one(self, sample):
        # 输入数据验证
        if len(sample) != len(self.full_cols):
            raise ValueError(
                f"输入数据长度错误！需要 {len(self.full_cols)} 个特征，实际 {len(sample)} 个"
            )
        
        # 创建DataFrame（强制列名）
        df = pd.DataFrame([sample], columns=self.full_cols)
        df = self._truncate(df)
        
        # 特征工程
        static_features = df[self.static_cols].copy()
        time_features = self._extract_time_series_features(df)
        
        # 合并特征（带列名保护）
        feature_df = pd.concat([static_features, time_features], axis=1)
        feature_df = feature_df[self.expected_columns]  # 强制列顺序
        
        # 最终维度验证
        if feature_df.shape[1] != len(self.expected_columns):
            raise RuntimeError(
                f"最终特征维度异常！预期 {len(self.expected_columns)}，实际 {feature_df.shape[1]}"
            )
            
        # 执行预测
        X_scaled = self.scaler.transform(feature_df)
        return self.model.predict(X_scaled)[0]

    # 保持其他辅助方法不变 (_get_slope, _calc_autocorr)
    # ...
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
    ["首页","性能预测", "配方建议"],
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

# 首页
if page == "首页":
    st.markdown("""
    本平台基于先进的人工智能和材料科学技术，致力于提供聚丙烯（PP）等聚合物复合材料的性能预测与配方优化建议。
    通过本平台，用户可以进行材料性能预测（如LOI和TS预测），并根据性能目标优化配方，推荐适合的助剂。
    """)
    st.markdown("<hr>", unsafe_allow_html=True)  # 添加水平分隔线
    # 功能概览
    st.markdown("""
    ## 功能概览
    1. **性能预测**：通过输入材料配方，预测聚合物复合材料的LOI和TS性能。
    2. **配方建议**：根据目标性能，优化材料配方。
    3. **添加剂推荐**：根据黄度值等时序数据，智能推荐最佳添加剂。
    """)
    st.markdown("<hr>", unsafe_allow_html=True)  # 添加水平分隔线
    # 引用部分
    st.markdown("""
    ## **引用**
    Weibin, Ma; Ling, Li; Yu, Zhang et al. Active learning-based generative design of halogen-free flame-retardant polymeric composites. Journal of Materials Informatics
    """)

    # 致谢部分优化，添加换行符
    st.markdown("""
    ## **致谢**<br>
    *贡献者*：<br>
    *团队*：<br>
    上海大学功能高分子组<br>
    *开发者*：<br>
    马维宾博士生<br>
    *审查*：<br>
    丁鹏教授<br>
    *基金支持*：<br>
    云南省科技重点计划项目 （202302AB080022）、苏州市重点技术研究项目 （SYG2024017）
    """, unsafe_allow_html=True)

    # 添加分隔线和背景色
    st.markdown("<hr>", unsafe_allow_html=True)  # 添加水平分隔线


# 性能预测页面
elif page == "性能预测":
    st.subheader("🔮 性能预测：基于配方预测LOI和TS")
    
    matrix_materials = ["PP", "PA", "PC/ABS", "POM", "PBT", "PVC", "其他"]
    flame_retardants = [
        "AHP", "ammonium octamolybdate", "Al(OH)3", "CFA", "APP", "Pentaerythritol", "DOPO", 
        "EPFR-1100NT", "XS-FR-8310", "ZS", "XiuCheng", "ZHS", "ZnB", "antimony oxides", 
        "Mg(OH)2", "TCA", "MPP", "PAPP", "其他"
    ]
    additives = [
        "wollastonite", "M-2200B", "ZBS-PV-OA", "FP-250S", "silane coupling agent", "antioxidant", 
        "SiO2", "其他"
    ]
    
    fraction_type = st.sidebar.selectbox("选择输入的单位", ["质量", "质量分数", "体积分数"])

    st.subheader("请选择配方中的基体、阻燃剂和助剂")
    selected_matrix = st.selectbox("选择基体", matrix_materials, index=0)
    selected_flame_retardants = st.multiselect("选择阻燃剂", flame_retardants, default=["ZS"])
    selected_additives = st.multiselect("选择助剂", additives, default=["wollastonite"])
    
    input_values = {}
    unit_matrix = get_unit(fraction_type)
    unit_flame_retardant = get_unit(fraction_type)
    unit_additive = get_unit(fraction_type)
    
    input_values[selected_matrix] = st.number_input(f"选择 {selected_matrix} ({unit_matrix})", min_value=0.0, max_value=100.0, value=50.0, step=0.1)
    
    for fr in selected_flame_retardants:
        input_values[fr] = st.number_input(f"选择 {fr}({unit_flame_retardant})", min_value=0.0, max_value=100.0, value=10.0, step=0.1)
    
    for ad in selected_additives:
        input_values[ad] = st.number_input(f"选择 {ad}  ({unit_additive})", min_value=0.0, max_value=100.0, value=10.0, step=0.1)
    
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
        if fraction_type in ["体积分数", "质量分数"] and abs(total - 100.0) > 1e-6:
            st.error(f"预测中止：{fraction_type}的总和必须为100%")
            st.stop()

        if is_only_pp:
            loi_pred = 17.5
            ts_pred = 35.0
        else:
            if fraction_type == "体积分数":
                vol_values = np.array(list(input_values.values()))
                mass_values = vol_values
                total_mass = mass_values.sum()
                input_values = {k: (v / total_mass * 100) for k, v in zip(input_values.keys(), mass_values)}
            
            for feature in models["loi_features"]:
                if feature not in input_values:
                    input_values[feature] = 0.0

            loi_input = np.array([[input_values[f] for f in models["loi_features"]]])
            loi_scaled = models["loi_scaler"].transform(loi_input)
            loi_pred = models["loi_model"].predict(loi_scaled)[0]
        
            for feature in models["ts_features"]:
                if feature not in input_values:
                    input_values[feature] = 0.0

            ts_input = np.array([[input_values[f] for f in models["ts_features"]]])
            ts_scaled = models["ts_scaler"].transform(ts_input)
            ts_pred = models["ts_model"].predict(ts_scaled)[0]
        
        col1, col2 = st.columns(2)
        with col1:
            st.metric(label="LOI预测值", value=f"{loi_pred:.2f}%")
        with col2:
            st.metric(label="TS预测值", value=f"{ts_pred:.2f} MPa")

# 配方建议页面
elif page == "配方建议":
    if sub_page == "配方优化":
        fraction_type = st.sidebar.radio(
            "📐 单位类型",
            ["质量", "质量分数", "体积分数"],
            key="unit_type"
        )
        st.subheader("🧪 配方建议：根据性能反推配方")
    
        col1, col2 = st.columns(2)
        with col1:
            target_loi = st.number_input("目标LOI值（%）", min_value=10.0, max_value=50.0, value=25.0, step=0.1)
        with col2:
            target_ts = st.number_input("目标TS值（MPa）", min_value=10.0, max_value=100.0, value=50.0, step=0.1)
        
        with st.expander("⚙️ 算法参数设置"):
            pop_size = st.number_input("种群数量", 50, 500, 200)
            n_gen = st.number_input("迭代代数", 10, 100, 50)
            cx_prob = st.slider("交叉概率", 0.1, 1.0, 0.7)
            mut_prob = st.slider("变异概率", 0.1, 1.0, 0.2)
    
        if st.button("🔍 开始优化", type="primary"):
            creator.create("FitnessMin", base.Fitness, weights=(-1.0,))
            creator.create("Individual", list, fitness=creator.FitnessMin)
            
            toolbox = base.Toolbox()
            all_features = ensure_pp_first(list(set(models["loi_features"] + models["ts_features"])))
            n_features = len(all_features)
            
            def generate_individual():
                individual = [random.uniform(0, 100) for _ in range(n_features)]
                total = sum(individual)
                return [max(0, x / total * 100) for x in individual]
            
            toolbox.register("individual", tools.initIterate, creator.Individual, generate_individual)
            toolbox.register("population", tools.initRepeat, list, toolbox.individual)
            
            def evaluate(individual):
                if fraction_type == "体积分数":
                    vol_values = np.array(individual)
                    mass_values = vol_values
                    total_mass = mass_values.sum()
                    if total_mass == 0:
                        return (1e6,)
                    mass_percent = (mass_values / total_mass) * 100
                else:
                    total = sum(individual)
                    if total == 0:
                        return (1e6,)
                    mass_percent = np.array(individual) / total * 100
                
                pp_index = all_features.index("PP")
                pp_content = mass_percent[pp_index]
                if pp_content < 50:
                    return (1e6,)
                
                loi_input = mass_percent[:len(models["loi_features"])]
                loi_scaled = models["loi_scaler"].transform([loi_input])
                loi_pred = models["loi_model"].predict(loi_scaled)[0]
                loi_error = abs(target_loi - loi_pred)
                
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
            
            best_individuals = tools.selBest(population, 10)
            best_values = []
            for individual in best_individuals:
                total = sum(individual)
                best_values.append([round(max(0, i / total * 100), 2) for i in individual])  # 修正括号闭合
            
            result_df = pd.DataFrame(best_values, columns=all_features)
            units = [get_unit(fraction_type) for _ in all_features]
            result_df.columns = [f"{col} ({unit})" for col, unit in zip(result_df.columns, units)]
            st.write(result_df)
    
    elif sub_page == "添加剂推荐":
        st.subheader("🧪 PVC添加剂智能推荐")
        predictor = Predictor("scaler_fold_1.pkl", "svc_fold_1.pkl")
        
        with st.form("additive_form"):
            st.markdown("### 基础参数")
            col_static = st.columns(3)
            with col_static[0]:
                add_ratio = st.number_input(
                    "产品质量指标_Sn%", 
                    min_value=0.0,
                    max_value=100.0,
                    value=5.0,
                    step=0.1
                )
            with col_static[1]:
                sn_percent = st.number_input(
                    "添加比例", 
                    min_value=0.0, 
                    max_value=100.0,
                    value=14.0,
                    step=0.1,
                    #help="锡含量范围0%~19%"
                )
            with col_static[2]:
                yijia_percent = st.number_input(
                    "一甲%",
                    min_value=0.0,
                    max_value=100.0,
                    value=23.55,
                    step=0.1,
                    #help="一甲含量范围15.1%~32%"
                )
            
            st.markdown("### 时序参数（黄度值随时间变化）")
            time_points = [
                ("3min", 15.0), ("6min", 16.0), ("9min", 17.0),
                ("12min", 18.0), ("15min", 19.0), ("18min", 20.0),
                ("21min", 21.0), ("24min", 22.0)
            ]
            yellow_values = {}
            prev_value = 5.0  # 初始最小值
            cols = st.columns(4)
            
            for idx, (time, default) in enumerate(time_points):
                with cols[idx % 4]:
                    if time == "3min":
                        current = st.number_input(
                            f"{time} 黄度值", 
                            min_value=5.0,
                            max_value=25.0,
                            value=default,
                            step=0.1,
                            key=f"yellow_{time}"
                        )
                    else:
                        current = st.number_input(
                            f"{time} 黄度值",
                            min_value=prev_value,
                            value=default,
                            step=0.1,
                            key=f"yellow_{time}"
                        )
                    yellow_values[time] = current
                    prev_value = current
    
            submit_btn = st.form_submit_button("生成推荐方案")
    
        if submit_btn:
            # 时序数据验证
            time_sequence = [yellow_values[t] for t, _ in time_points]
            if any(time_sequence[i] > time_sequence[i+1] for i in range(len(time_sequence)-1)):
                st.error("错误：黄度值必须随时间递增！请检查输入数据")
                st.stop()
                
            try:
                sample = [
                    sn_percent, add_ratio, yijia_percent,
                    yellow_values["3min"], yellow_values["6min"],
                    yellow_values["9min"], yellow_values["12min"],
                    yellow_values["15min"], yellow_values["18min"],
                    yellow_values["21min"], yellow_values["24min"]
                ]
                prediction = predictor.predict_one(sample)
                result_map = {
                    1: "无推荐添加剂", 
                    2: "氯化石蜡", 
                    3: "EA12（脂肪酸复合醇酯）",
                    4: "EA15（市售液体钙锌稳定剂）", 
                    5: "EA16（环氧大豆油）",
                    6: "G70L（多官能团的脂肪酸复合酯混合物）", 
                    7: "EA6（亚磷酸酯）"
                }
                
                # ============== 修改后的稳定剂显示部分 ==============
                additive_amount = 0.0 if prediction == 1 else add_ratio
                additive_name = result_map[prediction]
    
                # 构建完整配方表
                formula_data = [
                    ["PVC份数", 100.00],
                    ["加工助剂ACR份数", 1.00],
                    ["外滑剂70S份数", 0.35],
                    ["MBS份数", 5.00],
                    ["316A份数", 0.20],
                    ["稳定剂组成", ""],  # 主标题
                    ["  份数", 1.00],
                    ["  一甲%", yijia_percent],
                    ["  Sn%", sn_percent],
                ]
    
                # 动态添加添加剂信息
                if prediction != 1:
                    formula_data.extend([
                        ["  添加剂类型", additive_name],
                        ["  含量（wt%）", additive_amount]
                    ])
                else:
                    formula_data.append(["  推荐添加剂", "无"])
    
                # 创建格式化表格
                df = pd.DataFrame(formula_data, columns=["材料名称", "含量"])
                styled_df = df.style.format({"含量": "{:.2f}"})\
                                  .hide(axis="index")\
                                  .set_properties(**{
                                      'text-align': 'left',
                                      'font-size': '14px'
                                  })\
                                  .set_properties(
                                      subset=df.index[df['材料名称'].str.startswith('  ')],
                                      **{'padding-left': '30px', 'color': '#2c3e50'}
                                  )
    
                # 双列布局展示
                col1, col2 = st.columns([1, 2])
                with col1:
                    st.success(f"**推荐结果**\n\n{additive_name}")
                    st.metric(
                        "建议添加量", 
                        f"{additive_amount:.2f}%",
                        delta="无添加" if prediction == 1 else None,
                        delta_color="off"
                    )
                    
                with col2:
                    st.markdown("**完整配方表（基于PVC 100份）**")
                    st.dataframe(
                        styled_df,
                        use_container_width=True,
                        height=320,
                        column_config={
                            "材料名称": st.column_config.Column(
                                width="medium",
                                help="配方组成明细"
                            ),
                            "含量": st.column_config.NumberColumn(
                                format="%.2f",
                                width="small"
                            )
                        }
                    )
                # ============== 修改结束 ==============
                
            except Exception as e:
                st.error(f"预测过程中发生错误：{str(e)}")
                st.stop()
# 添加页脚
def add_footer():
    st.markdown("""
    <hr>
    <footer style="text-align: center;">
        <p>© 2025 阻燃聚合物复合材料智能设计平台</p>
        <p>开发者: 马维宾</p>
        <p>平台性质声明：本平台为科研协作网络服务平台，所有内容仅供学术研究、技术验证等非营利性科研活动使用，严禁用于任何商业用途。</p>
    </footer>
    """, unsafe_allow_html=True)

add_footer() 
