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
        """修复后的时序特征提取"""
        time_data = df[self.time_series_cols]
        time_data_filled = time_data.ffill(axis=1)
        
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
        df = self._truncate(df)
        
        # 特征合并
        static_features = df[self.static_cols]
        time_features = self._extract_time_series_features(df)
        feature_df = pd.concat([static_features, time_features], axis=1)
        feature_df = feature_df[self.static_cols + self.eng_features]
        
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

# 页面配置
# 页面配置
import base64
from PIL import Image
import io

def image_to_base64(image_path, quality=95):
    """高质量图片转base64"""
    img = Image.open(image_path)
    
    # 保持原始分辨率进行缩放
    if img.width != 1000:
        img = img.resize((1000, int(img.height * (1000 / img.width))), 
                        resample=Image.LANCZOS)
    
    buffered = io.BytesIO()
    img.save(buffered, format="PNG", quality=quality, optimize=True)
    return base64.b64encode(buffered.getvalue()).decode()

# 页面配置
image_path = "图片1.jpg"
icon_base64 = image_to_base64(image_path)  # 质量参数设为95

st.set_page_config(
    page_title="阻燃聚合物复合材料智能设计平台",
    layout="wide",
    page_icon=f"data:image/png;base64,{icon_base64}"
)

# 获取精确尺寸
img = Image.open(image_path)
target_width = 800
target_height = int(img.height * (target_width / img.width))

# 图片显示样式
st.markdown(f"""
<style>
    .fixed-width-img {{
        width: {target_width}px !important;
        height: {target_height}px !important;
        object-fit: contain;
        margin-left: 0;
        padding: 0;
        image-rendering: -webkit-optimize-contrast; /* Safari */
        image-rendering: crisp-edges; /* Standard */
    }}
    
    @media (max-width: 1050px) {{
        .fixed-width-img {{
            width: 95% !important;
            height: auto !important;
            max-width: 1000px;
        }}
    }}
</style>
""", unsafe_allow_html=True)

# 全局页眉样式
st.markdown("""
<style>
    .global-header {
        display: flex;
        align-items: center;
        gap: 25px;
        margin: 0 0 2rem 0;
        padding: 1rem 0;
        border-bottom: 3px solid #1e3d59;
        position: sticky;
        top: 0;
        background: white;
        z-index: 1000;
    }
    
    .header-logo {
        width: 80px;
        height: auto;
        flex-shrink: 0;
        border-radius: 8px;
        box-shadow: 0 2px 6px rgba(0,0,0,0.1);
    }
    
    .header-title {
        font-size: 4.4rem !important;
        color: #1e3d59;
        margin: 0;
        line-height: 1.2;
        font-family: 'Microsoft YaHei', sans-serif;
    }
    
    .header-subtitle {
        font-size: 1.1rem;
        color: #3f87a6;
        margin: 0.3rem 0 0 0;
    }

    @media (max-width: 768px) {
        .global-header {
            gap: 15px;
            padding: 0.5rem 0;
        }
        
        .header-logo {
            width: 60px;
        }
        
        .header-title {
            font-size: 1.8rem !important;
        }
        
        .header-subtitle {
            font-size: 0.9rem;
        }
    }
</style>
""", unsafe_allow_html=True)

# 全局页眉HTML
st.markdown(f"""
<div class="global-header">
    <img src="data:image/png;base64,{icon_base64}" 
         class="header-logo"
         alt="Platform Logo">
    <div>
        <h1 class="header-title">阻燃聚合物复合材料智能设计平台</h1>
        <p class="header-subtitle">Flame Retardant Polymer Composite Intelligent Platform</p>
    </div>
</div>
""", unsafe_allow_html=True)
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
    <style>
        :root {
            /* 字号系统 */
            --text-base: 1.15rem;
            --text-lg: 1.3rem;
            --text-xl: 1.5rem;
            --title-sm: 1.75rem;
            --title-md: 2rem;
            --title-lg: 2.25rem;
            
            /* 颜色系统 */
            --primary: #1e3d59;
            --secondary: #3f87a6;
            --accent: #2c2c2c;
        }

        body {
            /* 中文字体优先使用微软雅黑，英文使用Times New Roman */
            font-family: "Times New Roman", "微软雅黑", SimSun, serif;
            font-size: var(--text-base);
            line-height: 1.7;
            color: var(--accent);
        }

        /* 标题系统 */
        .platform-title {
            font-family: "Times New Roman", "微软雅黑", SimSun, serif;
            font-size: var(--title-lg);
            font-weight: 600;
            color: var(--primary);
            margin: 0 0 1.2rem 1.5rem;
            line-height: 1.3;
        }

        .section-title {
            font-family: "Times New Roman", "微软雅黑", SimSun, serif;
            font-size: var(--title-md);
            font-weight: 600;
            color: var(--primary);
            margin: 2rem 0 1.5rem;
            padding-bottom: 0.5rem;
            border-bottom: 2px solid var(--secondary);
        }

        /* 内容区块 */
        .feature-section {
            background: white;
            border-radius: 12px;
            padding: 1.5rem;
            margin: 1.5rem 0;
            box-shadow: 0 2px 12px rgba(0, 0, 0, 0.08);
        }

        .feature-section p {
            font-size: var(--text-lg);
            line-height: 1.8;
            margin: 0.8rem 0;
        }

        /* 功能列表 */
        .feature-list li {
            font-size: var(--text-lg);
            padding-left: 2rem;
            margin: 1rem 0;
            position: relative;
        }

        .feature-list li:before {
            content: "•";
            color: var(--secondary);
            font-size: 1.5em;
            position: absolute;
            left: 0;
            top: -0.1em;
        }

        /* 引用区块 */
        .quote-section {
            font-size: var(--text-lg);
            background: #f8f9fa;
            border-left: 3px solid var(--secondary);
            padding: 1.2rem;
            margin: 1.5rem 0;
            border-radius: 0 8px 8px 0;
        }

        /* 响应式调整 */
        @media (min-width: 768px) {
            :root {
                --text-base: 1.2rem;
                --text-lg: 1.35rem;
                --text-xl: 1.6rem;
                --title-sm: 1.9rem;
                --title-md: 2.2rem;
                --title-lg: 2.5rem;
            }
            
            .section-title {
                margin: 2.5rem 0 2rem;
            }
        }

        @media (max-width: 480px) {
            :root {
                --text-base: 1.1rem;
                --title-lg: 2rem;
            }
        }
    </style>
    """, unsafe_allow_html=True)

    # 平台简介
    st.markdown("""
    <div class="feature-section">
        <p>
            本平台融合AI与材料科学技术，用于可持续高分子复合材料智能设计，重点关注材料阻燃、力学和耐热等性能的优化与调控。
        </p>
    </div>
    """, unsafe_allow_html=True)

    # 核心功能
    st.markdown('<div class="section-title">核心功能</div>', unsafe_allow_html=True)
    st.markdown("""
    <div class="feature-section">
        <ul class="feature-list">
            <li><strong>性能预测</strong></li>
            <li><strong>配方建议</strong></li>
            <li><strong>添加剂推荐</strong></li>
        </ul>
    </div>
    """, unsafe_allow_html=True)

    # 研究成果
    st.markdown('<div class="section-title">研究成果</div>', unsafe_allow_html=True)
    st.markdown("""
    <div class="quote-section">
        Ma Weibin, Li Ling, Zhang Yu, Li Minjie, Song Na, Ding Peng. <br>
        <em>Active learning-based generative design of halogen-free flame-retardant polymeric composites.</em> <br>
        <strong>J Mater Inf</strong> 2025;5:09. DOI: <a href="http://dx.doi.org/10.20517/jmi.2025.09" target="_blank">10.20517/jmi.2025.09</a>
    </div>
    """, unsafe_allow_html=True)

    # 致谢部分
    st.markdown('<div class="section-title">致谢</div>', unsafe_allow_html=True)
    st.markdown("""
    <div class="feature-section">
        <p style="font-size: var(--text-lg);">
            本研究获得云南省科技重点计划项目(202302AB080022)支持
        </p>
    </div>
    """, unsafe_allow_html=True)

    # 开发者信息
    st.markdown('<div class="section-title">开发者</div>', unsafe_allow_html=True)
    st.markdown("""
    <div class="feature-section">
        <p style="font-size: var(--text-lg);">
            上海大学功能高分子团队-PolyDesign：马维宾，李凌，张瑜，宋娜，丁鹏
        </p>
    </div>
    """, unsafe_allow_html=True)
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
    unit_matrix = get_unit(fraction_type)
    unit_flame_retardant = get_unit(fraction_type)
    unit_additive = get_unit(fraction_type)

    input_values[selected_matrix] = st.number_input(f"选择 {selected_matrix} ({unit_matrix})", min_value=0.0, max_value=100.0, value=50.0, step=0.1)

    for fr in selected_flame_retardants:
        input_values[fr] = st.number_input(f"选择 {fr}({unit_flame_retardant})", min_value=0.0, max_value=100.0, value=10.0, step=0.1)

    for ad in selected_additives:
        input_values[ad] = st.number_input(f"选择 {ad} ({unit_additive})", min_value=0.0, max_value=100.0, value=10.0, step=0.1)

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

    # 模型验证样本
    with st.expander("📊 模型精度验证样本（预测误差<15%）"):
        samples = [
            {
                "name": "配方1",
                "配方": {"PP": 63.2, "PAPP": 23.0, "ZS": 1.5, "Anti-drip-agent": 0.3, "MPP": 9.0, "wollastonite": 3.0},
                "LOI_真实值": 43.5,
                "TS_真实值": 15.845
            },
            {
                "name": "配方2",
                "配方": {"PP": 65.2, "PAPP": 23.0, "ZS": 1.5, "Anti-drip-agent": 0.3, "MPP": 7.0, "wollastonite": 3.0},
                "LOI_真实值": 43.0,
                "TS_真实值": 16.94
            },
            {
                "name": "配方3",
                "配方": {"PP": 58.2, "PAPP": 23.0, "ZS": 0.5, "Anti-drip-agent": 0.3, "MPP": 13.0, "wollastonite": 5.0},
                "LOI_真实值": 43.5,
                "TS_真实值": 15.303
            }
        ]

        st.markdown("""
        <style>
            .sample-box {
                border: 1px solid #e6e6e6;
                border-radius: 8px;
                padding: 1.2rem;
                margin: 1rem 0;
                background: #f9fafb;
            }
            .sample-title {
                color: #2c3e50;
                font-weight: 600;
                margin-bottom: 0.8rem;
            }
            .metric-badge {
                background: #f0f2f6;
                padding: 0.3rem 0.8rem;
                border-radius: 20px;
                display: inline-block;
                margin: 0.2rem;
            }
        </style>
        """, unsafe_allow_html=True)

        for sample in samples:
            all_features = set(models["loi_features"]) | set(models["ts_features"])
            input_vector = {k: 0.0 for k in all_features}
            for k, v in sample["配方"].items():
                input_vector[k] = v

            loi_input = np.array([[input_vector[f] for f in models["loi_features"]]])
            loi_scaled = models["loi_scaler"].transform(loi_input)
            loi_pred = models["loi_model"].predict(loi_scaled)[0]

            ts_input = np.array([[input_vector[f] for f in models["ts_features"]]])
            ts_scaled = models["ts_scaler"].transform(ts_input)
            ts_pred = models["ts_model"].predict(ts_scaled)[0]

            loi_error = abs(sample["LOI_真实值"] - loi_pred) / sample["LOI_真实值"] * 100
            ts_error = abs(sample["TS_真实值"] - ts_pred) / sample["TS_真实值"] * 100

            loi_color = "#2ecc71" if loi_error < 15 else "#e74c3c"
            ts_color = "#2ecc71" if ts_error < 15 else "#e74c3c"

            st.markdown(f"""
            <div class="sample-box">
                <div class="sample-title">📌 {sample["name"]}</div>
                <div class="metric-badge" style="color: {loi_color}">LOI误差: {loi_error:.1f}%</div>
                <div class="metric-badge" style="color: {ts_color}">TS误差: {ts_error:.1f}%</div>
                <div style="margin-top: 0.8rem;">
                    🔥 真实LOI: {sample["LOI_真实值"]}% → 预测LOI: {loi_pred:.2f}%
                </div>
                <div>💪 真实TS: {sample["TS_真实值"]} MPa → 预测TS: {ts_pred:.2f} MPa</div>
            </div>
            """, unsafe_allow_html=True)

            if loi_error < 15 and ts_error < 15:
                st.success(f"✅ {sample['name']}：模型精度超过85%")
            else:
                st.warning(f"⚠️ {sample['name']}：模型预测误差较大")

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
                add_ratio = st.number_input("添加比例 (%)", 
                                          min_value=0.0,
                                          max_value=100.0,
                                          value=5.0,
                                          step=0.1)
            with col_static[1]:
                sn_percent = st.number_input("Sn含量 (%)", 
                                           min_value=0.0, 
                                           max_value=19.0,
                                           value=14.0,
                                           step=0.1,
                                           help="锡含量范围0%~19%")
            with col_static[2]:
                yijia_percent = st.number_input("一甲含量 (%)",
                                               min_value=15.1,
                                               max_value=32.0,
                                               value=23.55,
                                               step=0.1,
                                               help="一甲胺含量范围15.1%~32%")
            
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
                
                # ============== 修改开始 ==============
                # 动态确定添加量和显示名称
                additive_amount = 0.0 if prediction == 1 else add_ratio
                additive_name = result_map[prediction]
    
                # 构建完整配方表
                formula_data = [
                    ["PVC份数", 100.00],
                    ["加工助剂ACR份数", 1.00],
                    ["外滑剂70S份数", 0.35],
                    ["MBS份数", 5.00],
                    ["316A份数", 0.20],
                    ["稳定剂份数", 1.00]
                ]
                
                # 根据预测结果动态添加条目
                if prediction != 1:
                    formula_data.append([f"{additive_name}含量（wt%）", additive_amount])
                else:
                    formula_data.append([additive_name, additive_amount])
                # ============== 修改结束 ==============
    
                # 创建格式化表格
                df = pd.DataFrame(formula_data, columns=["材料名称", "含量"])
                styled_df = df.style.format({"含量": "{:.2f}"})\
                                  .hide(axis="index")\
                                  .set_properties(**{'text-align': 'left'})
                
                # 双列布局展示
                col1, col2 = st.columns([1, 2])
                with col1:
                    # ============== 修改开始 ==============
                    st.success(f"**推荐添加剂类型**  \n{additive_name}")
                    st.metric("建议添加量", 
                             f"{additive_amount:.2f}%",
                             delta="无添加" if prediction == 1 else None)
                    # ============== 修改结束 ==============
                    
                with col2:
                    st.markdown("**完整配方表（基于PVC 100份）**")
                    st.dataframe(styled_df,
                                use_container_width=True,
                                height=280,
                                column_config={
                                    "材料名称": "材料名称",
                                    "含量": st.column_config.NumberColumn(
                                        "含量",
                                        format="%.2f"
                                    )
                                })
                
    
                
            except Exception as e:
                st.error(f"预测过程中发生错误：{str(e)}")
                st.stop()
# 添加页脚
def add_footer():
    st.markdown("""
    <hr>
    <footer style="text-align: center;">
        <p>© 2025 阻燃聚合物复合材料智能设计平台</p>
        <p>声明：本平台仅供学术研究、技术验证等非营利性科研活动使用，严禁用于任何商业用途。</p>
    </footer>
    """, unsafe_allow_html=True)

add_footer()
