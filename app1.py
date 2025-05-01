import pandas as pd
import numpy as np
import warnings
from scipy import stats
from scipy.stats import kurtosis, skew
from sklearn.impute import SimpleImputer
import joblib
import streamlit as st
import base64
import random
from deap import base, creator, tools, algorithms

# ==================== Predictor类 ====================
class Predictor:
    def __init__(self, scaler_path, svc_path):
        self.scaler = joblib.load(scaler_path)
        self.model = joblib.load(svc_path)
        
        # 明确特征顺序（必须与训练时完全一致）
        self.static_cols = ["产品质量指标_Sn%", "添加比例", "一甲%"]
        self.time_series_cols = [
            "黄度值_3min", "6min", "9min", "12min",
            "15min", "18min", "21min", "24min"
        ]
        self.eng_features = [
            'seq_length', 'max_value', 'mean_value', 'min_value',
            'std_value', 'trend', 'range_value', 'autocorr'
        ]
        # 定义完整特征顺序
        self.expected_features = self.static_cols + self.eng_features
        
        # 验证scaler维度
        if self.scaler.n_features_in_ != len(self.expected_features):
            raise ValueError(f"Scaler特征数不匹配！当前：{self.scaler.n_features_in_}，需要：{len(self.expected_features)}")

    def _truncate(self, df):
        """改进后的截断逻辑：基于变化率阈值"""
        time_cols = sorted(  # 修复括号闭合问题
            [col for col in df.columns if "min" in col],
            key=lambda x: int(x.split('_')[-1].replace('min',''))
        )  # 补全这个括号
        
        values = df[time_cols].iloc[0].values
        threshold = 0.3
        
        truncate_pos = len(values)
        for i in range(1, len(values)):
            if pd.isna(values[i]) or pd.isna(values[i-1]):
                continue
            rate = abs(values[i] - values[i-1]) / (values[i-1] + 1e-6)
            if rate < threshold:
                truncate_pos = i
                break
        
        for col in time_cols[truncate_pos:]:
            df[col] = np.nan
        return df

    def _get_slope(self, row):
        x = np.arange(len(row))
        y = row.values
        mask = ~np.isnan(y)
        if sum(mask) >= 2:
            return stats.linregress(x[mask], y[mask])[0]
        return 0.0

    def _calc_autocorr(self, row):
        values = row.dropna().values
        if len(values) > 1:
            return np.corrcoef(values[:-1], values[1:])[0, 1]
        return 0.0

    def _extract_time_series_features(self, df):
        time_data = df[self.time_series_cols]
        time_data_filled = time_data.ffill(axis=1).bfill(axis=1)
        
        features = pd.DataFrame()
        features['seq_length'] = time_data_filled.notna().sum(axis=1)
        features['max_value'] = time_data_filled.max(axis=1)
        features['mean_value'] = time_data_filled.mean(axis=1)
        features['min_value'] = time_data_filled.min(axis=1)
        features['std_value'] = time_data_filled.std(axis=1)
        features['range_value'] = features['max_value'] - features['min_value']
        features['trend'] = time_data_filled.apply(self._get_slope, axis=1)
        features['autocorr'] = time_data_filled.apply(self._calc_autocorr, axis=1)
        return features.fillna(0)

    def predict_one(self, sample):
        # 构建输入数据框架
        full_cols = self.static_cols + self.time_series_cols
        df = pd.DataFrame([sample], columns=full_cols)
        
        # 数据预处理
        df = self._truncate(df)
        
        # 特征提取
        static_features = df[self.static_cols]
        time_features = self._extract_time_series_features(df)
        
        # 特征合并与对齐
        feature_df = pd.concat([static_features, time_features], axis=1)
        feature_df = feature_df.reindex(columns=self.expected_features, fill_value=0)
        
        # 维度验证
        if feature_df.shape[1] != len(self.expected_features):
            raise ValueError(f"特征维度错误！当前：{feature_df.shape[1]}，需要：{len(self.expected_features)}")
        
        # 数据标准化
        X_scaled = self.scaler.transform(feature_df)
        
        # 预测与结果处理
        prediction = self.model.predict(X_scaled)[0]
        proba = self.model.predict_proba(X_scaled)[0]
        return prediction, proba

# ==================== Streamlit界面 ====================
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

# 子功能选择
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

# 首页
if page == "首页":
    st.markdown("""
    本平台基于先进的人工智能和材料科学技术，致力于提供聚丙烯（PP）等聚合物复合材料的性能预测与配方优化建议。
    通过本平台，用户可以进行材料性能预测（如LOI和TS预测），并根据性能目标优化配方，推荐适合的助剂。
    """)
    st.markdown("<hr>", unsafe_allow_html=True)
    st.markdown("""
    ## 功能概览
    1. **性能预测**：通过输入材料配方，预测聚合物复合材料的LOI和TS性能。
    2. **配方建议**：根据目标性能，优化材料配方。
    3. **添加剂推荐**：根据黄度值等时序数据，智能推荐最佳添加剂。
    """)
    st.markdown("<hr>", unsafe_allow_html=True)
    st.markdown("""
    ## **引用**
    Weibin, Ma; Ling, Li; Yu, Zhang et al. Active learning-based generative design of halogen-free flame-retardant polymeric composites. Journal of Materials Informatics
    """)
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
    st.markdown("<hr>", unsafe_allow_html=True)

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

    if st.button("🚀 开始预测", type="primary"):
        if fraction_type in ["体积分数", "质量分数"] and abs(total - 100.0) > 1e-6:
            st.error(f"预测中止：{fraction_type}的总和必须为100%")
            st.stop()

        if is_only_pp:
            loi_pred, ts_pred = 17.5, 35.0
        else:
            # ...（保持原有数据处理逻辑）...
        
        col1, col2 = st.columns(2)
        with col1:
            st.metric(label="LOI预测值", value=f"{loi_pred:.2f}%")
        with col2:
            st.metric(label="TS预测值", value=f"{ts_pred:.2f} MPa")

# 配方建议页面
elif page == "配方建议":
    if sub_page == "配方优化":
        # ...（保持原有配方优化代码）...
    
    elif sub_page == "添加剂推荐":
        st.subheader("🧪 PVC添加剂智能推荐")
        predictor = Predictor("scaler_fold_1.pkl", "svc_fold_1.pkl")
        
        with st.form("additive_form"):
            # ...（保持原有表单代码）...
        
        if submit_btn:
            try:
                # ...（保持原有预测处理代码）...
            except Exception as e:
                st.error(f"预测错误：{str(e)}")
                st.stop()

# 页脚
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
