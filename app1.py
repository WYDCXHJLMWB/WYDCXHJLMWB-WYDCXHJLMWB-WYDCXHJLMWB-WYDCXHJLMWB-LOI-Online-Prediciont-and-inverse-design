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
        self.scaler = joblib.load(scaler_path)
        self.model = joblib.load(svc_path)
        
        # 特征结构定义
        self.static_cols = ["产品质量指标_Sn%", "添加比例", "一甲%"]
        self.time_series_cols = ["3min", "6min", "9min", "12min", "15min", "18min", "21min", "24min"]
        self.eng_features = ['seq_length', 'max_value', 'mean_value', 'min_value', 'std_value', 'trend', 'range_value', 'autocorr']
        self.expected_columns = self.static_cols + self.eng_features
        self.full_cols = self.static_cols + self.time_series_cols
        
        # 初始化验证
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
        # 移除前向填充，直接使用原始数据（包含NaN）
        time_data = df[self.time_series_cols].copy()
        
        return pd.DataFrame({
            'seq_length': time_data.count(axis=1),  # 计算非NaN值的数量
            'max_value': time_data.max(axis=1),
            'mean_value': time_data.mean(axis=1),
            'min_value': time_data.min(axis=1),
            'std_value': time_data.std(axis=1),
            'trend': time_data.apply(self._get_slope, axis=1),
            'range_value': time_data.max(axis=1) - time_data.min(axis=1),
            'autocorr': time_data.apply(self._calc_autocorr, axis=1)
        }, columns=self.eng_features)
    def predict_one(self, sample):
        if len(sample) != len(self.full_cols):
            raise ValueError(f"需要{len(self.full_cols)}个特征，实际{len(sample)}个。完整顺序：{self.full_cols}")
        
        df = pd.DataFrame([sample], columns=self.full_cols)
        df = self._truncate(df)
        
        # 单次合并并强制列名
        static_features = df[self.static_cols]
        time_features = self._extract_time_series_features(df)
        feature_df = pd.concat([static_features.reset_index(drop=True), 
                             time_features.reset_index(drop=True)], axis=1)
        feature_df.columns = self.expected_columns
        
        # 最终验证
        if list(feature_df.columns) != self.expected_columns:
            raise ValueError(f"列名不匹配！\n预期：{self.expected_columns}\n实际：{feature_df.columns.tolist()}")
        
        return self.model.predict(self.scaler.transform(feature_df))[0]


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
    page_title="阻燃聚合物复合材料智能设计平台",
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
    <div style="text-align: center;">

        <h1 style="color: #4A90E2;">聚合物复合材料智能平台</h1>
    </div>
    """, unsafe_allow_html=True)
    
    # 水平分隔线
    st.markdown("<hr>", unsafe_allow_html=True)
    
    # 功能概览 - 卡片式布局
    st.markdown("""
    ## 功能概览
    <div style="display: flex; justify-content: space-between; gap: 20px;">
        <div style="background: #f0f2f6; padding: 20px; border-radius: 10px; flex: 1;">
            <h3>📊 性能预测</h3>
            <p>通过输入材料配方，预测聚合物复合材料的 LOI 和TS 性能。</p>
        </div>
        <div style="background: #f0f2f6; padding: 20px; border-radius: 10px; flex: 1;">
            <h3>🔧 配方建议</h3>
            <p>根据目标性能，优化材料配方。</p>
        </div>
        <div style="background: #f0f2f6; padding: 20px; border-radius: 10px; flex: 1;">
            <h3>🧪 添加剂推荐</h3>
            <p>根据黄度值等时序数据，智能推荐最佳添加剂。</p>
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    # 水平分隔线
    st.markdown("<hr>", unsafe_allow_html=True)
    
    # 引用部分
    st.markdown("""
    ## 引用
    <div style="background: #f0f2f6; padding: 20px; border-radius: 10px;">
        <p>Ma W, Li L, Zhang Y, Li M, Song N, Ding P. Active learning-based generative design of halogen-free flame-retardant polymeric composites. J Mater Inf 2025;5:[Accept]. <a href="http://dx.doi.org/10.20517/jmi.2025.09 " target="_blank">DOI</a></p>
    </div>
    """, unsafe_allow_html=True)
    
    # 水平分隔线
    st.markdown("<hr>", unsafe_allow_html=True)
    
    # 致谢部分
    st.markdown("""
    ## 致谢
    <div style="background: #f0f2f6; padding: 20px; border-radius: 10px;">
        <p>云南省科技重点计划项目（202302AB080022）</p>
        <p><strong>开发者：</strong>上海大学功能高分子团队：马维宾、李凌、张瑜、宋娜、丁鹏</p>
        <p><strong>审查：</strong>丁鹏</p>
    </div>
    """, unsafe_allow_html=True)
    
    # 底部渐变背景
    st.markdown("""
    <style>
    .footer {
        position: fixed;
        left: 0;
        bottom: 0;
        width: 100%;
        background: linear-gradient(90deg, #4A90E2, #6A82FB);
        color: white;
        text-align: center;
        padding: 10px;
    }


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
        "Anti-drip-agent", "wollastonite", "M-2200B", "ZBS-PV-OA", "FP-250S", "silane coupling agent", "antioxidant",
        "SiO2", "其他"
    ]

    fraction_type = st.sidebar.selectbox("选择输入的单位", ["质量", "质量分数", "体积分数"])

    st.subheader("请选择配方中的基体、阻燃剂和助剂")
    selected_matrix = st.selectbox("选择基体", matrix_materials, index=0)
    selected_flame_retardants = st.multiselect("选择阻燃剂", flame_retardants, default=["ZS"])
    selected_additives = st.multiselect("选择助剂", additives, default=["wollastonite"])

    input_values = {}
    unit = get_unit(fraction_type)

    input_values[selected_matrix] = st.number_input(f"选择 {selected_matrix} ({unit})", min_value=0.0, max_value=100.0, value=50.0, step=0.1)

    for fr in selected_flame_retardants:
        input_values[fr] = st.number_input(f"选择 {fr} ({unit})", min_value=0.0, max_value=100.0, value=10.0, step=0.1)

    for ad in selected_additives:
        input_values[ad] = st.number_input(f"选择 {ad} ({unit})", min_value=0.0, max_value=100.0, value=10.0, step=0.1)

    total = sum(input_values.values())
    is_only_pp = all(v == 0 for k, v in input_values.items() if k != "PP")

    with st.expander("✅ 输入验证"):
        if fraction_type in ["体积分数", "质量分数"]:
            if abs(total - 100.0) > 1e-6:
                st.error(f"❗ {fraction_type}的总和必须为100%（当前：{total:.2f}%）")
            else:
                st.success(f"{fraction_type}总和验证通过")
        else:
            st.success("成分总和验证通过")
        if is_only_pp:
            st.info("检测到纯PP配方")

    with st.expander("🔍 模型验证（质量分数参考样本）", expanded=True):
        st.markdown("### 标准参考样本验证（质量分数基准）")

        reference_samples = {
            "阻燃PP-1": {
                "composition": {
                    "PP": 61.7, "PAPP": 23.0, "MPP": 9.0, "wollastonite": 5.0, "ZS": 1.0, "Anti-drip-agent": 0.3,
                },
                "actual": {"LOI": 43, "TS": 15.832}
            },
            "阻燃PP-2": {
                "composition": {
                    "PP": 65.2, "PAPP": 23.0, "MPP": 7.0, "wollastonite": 3.0, "ZS": 1.5, "Anti-drip-agent": 0.3,
                },
                "actual": {"LOI": 43, "TS": 16.94}
            },
            "阻燃PP-3": {
                "composition": {
                    "PP": 59.7, "PAPP": 23.0, "MPP": 13.0, "wollastonite": 3.0, "ZS": 1.0, "Anti-drip-agent": 0.3,
                },
                "actual": {"LOI": 43, "TS": 15.289}
            }
        }

        cols = st.columns(3)
        for idx, (sample_name, sample_data) in enumerate(reference_samples.items()):
            with cols[idx]:
                st.markdown(f"##### {sample_name}")
                comp_df = pd.DataFrame(
                    [(k, f"{v}%") for k, v in sample_data["composition"].items()],
                    columns=["材料", "质量分数"]
                )
                st.dataframe(comp_df, hide_index=True, use_container_width=True, height=200)

                if st.button(f"验证 {sample_name}", key=f"verify_{sample_name}", help="点击自动填充并验证该样本"):
                    input_values.clear()
                    for material, percent in sample_data["composition"].items():
                        input_values[material] = percent

                    actual_loi = sample_data["actual"]["LOI"]
                    actual_ts = sample_data["actual"]["TS"]

                    try:
                        loi_input = np.array([[input_values.get(f, 0.0) for f in models["loi_features"]]])
                        loi_scaled = models["loi_scaler"].transform(loi_input)
                        pred_loi = models["loi_model"].predict(loi_scaled)[0]

                        ts_input = np.array([[input_values.get(f, 0.0) for f in models["ts_features"]]])
                        ts_scaled = models["ts_scaler"].transform(ts_input)
                        pred_ts = models["ts_model"].predict(ts_scaled)[0]

                        col1, col2 = st.columns(2)
                        with col1:
                            delta_loi = abs(pred_loi - actual_loi)
                            st.metric("LOI预测值", f"{pred_loi:.1f}%", delta=f"Δ{delta_loi:.1f}%", help=f"实际值: {actual_loi}%")
                        with col2:
                            delta_ts = abs(pred_ts - actual_ts)
                            st.metric("TS预测值", f"{pred_ts:.1f}MPa", delta=f"Δ{delta_ts:.1f}MPa", help=f"实际值: {actual_ts}MPa")

                        st.markdown(f"""
                            ###### 误差分析
                            - LOI绝对误差: `{delta_loi:.2f}%`  
                            - TS绝对误差: `{delta_ts:.2f}MPa`  
                            - LOI相对误差: `{(delta_loi/actual_loi)*100:.1f}%`  
                            - TS相对误差: `{(delta_ts/actual_ts)*100:.1f}%`
                        """)
                        loi_accuracy = 100 - (delta_loi/actual_loi)*100
                        ts_accuracy = 100 - (delta_ts/actual_ts)*100
                        
                        if loi_accuracy >= 85 and ts_accuracy >= 85:
                            st.success(f"✅ 模型精度超过85%（LOI：{loi_accuracy:.1f}%，TS：{ts_accuracy:.1f}%）")
                        else:
                            st.error(f"⚠️ 模型精度未达标（LOI：{loi_accuracy:.1f}%，TS：{ts_accuracy:.1f}%）")
                    except Exception as e:
                        st.error(f"验证失败: {str(e)}")
                        st.stop()


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
                mass_values = vol_values  # 若有密度数据可替换此行
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
        try:
            predictor = Predictor("scaler_fold_1.pkl", "svc_fold_1.pkl")
        except Exception as e:
            st.error(f"初始化失败: {str(e)}")
            st.stop()
        with st.expander("📚 参考样本数据（点击展开）"):
            st.markdown("""
            ### 典型样本数据参考
            以下为验证模型效果的标准样本数据：
            """)
            
            # 样本数据定义
            sample_data = {
                "样本名称": ["无添加剂样本", "氯化石蜡样本", "EA15样本"],
                "样本说明": [
                    "预期结果：无推荐添加剂",
                    "预期结果：氯化石蜡",
                    "预期结果：EA15（市售液体钙锌稳定剂）"
                ],
                "数据明细": [
                    [19.2, 0, 32, 5.36, 6.29, 7.57, 8.57, 10.26, 13.21, 16.54, 27.47],
                    [18.5, 3.64, 31.05, 5.29, 6.83, 8.00, 9.32, 11.40, 14.12, 18.37, 30.29],
                    [19, 1.04, 31.88, 5.24, 6.17, 7.11, 8.95, 10.33, 13.21, 17.48, 28.08]
                ]
            }
            
        # 创建展示表格
        for i in range(3):
            cols = st.columns([0.2, 1, 3])
            with cols[0]:
                st.metric(label="样本编号", value=f"#{i+1}")
            with cols[1]:
                st.markdown(f"""
                **{sample_data['样本名称'][i]}**  
                {sample_data['样本说明'][i]}
                """)
            with cols[2]:
                df = pd.DataFrame(
                    [sample_data['数据明细'][i]],
                    columns=["Sn%", "添加比例", "一甲%", 
                            "3min", "6min", "9min", "12min",
                            "15min", "18min", "21min", "24min"]
                ).T.reset_index()
                df.columns = ["参数", "数值"]
                st.dataframe(df.style.format({"数值": "{:.2f}"}), 
                           height=300,
                           use_container_width=True)
            st.markdown("---")
        with st.form("additive_form"):
            example_options = {
            "无添加剂样本": [19.2, 0, 32, 5.36, 6.29, 7.57, 8.57, 10.26, 13.21, 16.54, 27.47],
            "氯化石蜡样本": [18.5, 3.64, 31.05, 5.29, 6.83, 8.00, 9.32, 11.40, 14.12, 18.37, 30.29],
            "EA15样本": [19, 1.041666667, 31.88, 5.24, 6.17, 7.11, 8.95, 10.33, 13.21, 17.48, 28.08]
        }
            
            # 基础参数
            col1, col2, col3 = st.columns(3)
            with col1: sn = st.number_input("Sn%", 0.0, 100.0, 5.0)
            with col2: ratio = st.number_input("添加比例", 0.0, 100.0, 14.0)
            with col3: yijia = st.number_input("一甲%", 0.0, 100.0, 23.55)
            
            # 时序参数（修改部分）
            st.markdown("### 黄度值时序参数（0-50）")
            time_points = ["3min", "6min", "9min", "12min", "15min", "18min", "21min", "24min"]
            yellow = {}
            cols = st.columns(4)
            
            for idx, t in enumerate(time_points):
                with cols[idx%4]:
                    yellow[t] = st.number_input(
                        f"{t} 黄度值",
                        min_value=0.0,    # 修改最小值固定为0
                        max_value=50.0,   # 修改最大值调整为50
                        value=15.0+idx,   # 默认值保持不变
                        key=f"yellow_{t}"
                    )
            
            # 提交按钮
            if st.form_submit_button("生成推荐"):
                try:
                    # 构建输入样本（保持原有逻辑）
                    sample = [sn, ratio, yijia] + [yellow[t] for t in time_points]
                    
                    # 执行预测（保持原有逻辑）
                    pred = predictor.predict_one(sample)
                    
                    # 显示结果（保持原有逻辑）
                    result_map = {
                        1: "无推荐添加剂", 2: "氯化石蜡", 3: "EA12", 
                        4: "EA15", 5: "EA16", 6: "G70L", 7: "EA6"
                    }
                    additive = result_map.get(pred, "未知")
                    
                    # 构建展示数据（保持原有逻辑）
                    formula = [
                        ["PVC份数", 100.0], ["ACR份数", 1.0], ["70S份数", 0.35],
                        ["MBS份数", 5.0], ["316A份数", 0.2], ["稳定剂份数", 1.0],
                        ["一甲%", yijia], ["Sn%", sn]
                    ]
                    if pred != 1:
                        formula.extend([[f"{additive}含量", f"{ratio if pred!=1 else 0}%"]])
    
                    # 显示结果（保持原有逻辑）
                    col1, col2 = st.columns(2)
                    with col1:
                        st.metric("推荐结果", additive)
                    with col2:
                        st.dataframe(pd.DataFrame(formula, columns=["材料", "含量"]), 
                                   hide_index=True)
                            
                except Exception as e:
                    st.error(f"预测错误: {str(e)}")
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
