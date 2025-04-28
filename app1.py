import pandas as pd
import numpy as np
import joblib
import streamlit as st

class Predictor:
    def __init__(self, scaler_path, svc_path):
        # 加载训练好的 scaler 和 svc 模型
        self.scaler = joblib.load(scaler_path)
        self.model = joblib.load(svc_path)
        # 定义静态特征和时序特征的列名（顺序与训练时一致）
        self.static_cols = ["产品质量指标_Sn%", "添加比例", "一甲%"]
        self.time_series_cols = ["黄度值_3min", "6min", "9min", "12min", "15min", "18min", "21min", "24min"]

    def _truncate(self, df):
        """处理时序数据截断（保持原有逻辑）"""
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

    def _extract_features(self, df):
        """提取完整的特征工程（必须与训练时一致）"""
        # 静态特征
        static_data = {
            col: df[col].values[0] 
            for col in self.static_cols
        }
        
        # 时序特征工程
        ts_cols = [col for col in df.columns if "min" in col.lower()]
        ts_series = df[ts_cols].iloc[0].dropna()
        
        eng_features = {
            'seq_length': len(ts_series),
            'max_value': ts_series.max(),
            'mean_value': ts_series.mean(),
            'min_value': ts_series.min(),
            'std_value': ts_series.std(),
            'trend': (ts_series[-1] - ts_series[0])/len(ts_series),
            'range_value': ts_series.max() - ts_series.min(),
            'autocorr': ts_series.autocorr()
        }
        
        # 合并所有特征
        return {**static_data, **eng_features}

    def predict_one(self, sample):
        """完整的预测流程"""
        # 构造完整输入DataFrame（包含所有原始列） 
        full_cols = self.static_cols + self.time_series_cols
        df = pd.DataFrame([sample], columns=full_cols)
        
        # 预处理流程
        df = self._truncate(df)
        
        # 特征工程
        features = self._extract_features(df)
        feature_df = pd.DataFrame([features])[self.static_cols + self.eng_features]
        
        # 验证特征维度
        if feature_df.shape[1] != self.scaler.n_features_in_:
            raise ValueError(
                f"特征维度不匹配！当前：{feature_df.shape[1]}，"
                f"需要：{self.scaler.n_features_in_}"
            )
        
        # 标准化 & 预测
        X_scaled = self.scaler.transform(feature_df)
        return self.model.predict(X_scaled)[0]
    def extract_time_series_features(df, feature_types=None, static_features=None):
        """
        提取时序特征并与静态特征合并
        
        参数:
        df: DataFrame, 原始数据
        feature_types: list, 要提取的时序特征类型列表，默认为['seq_length', 'max_value', 'mean_value']
        static_features: list, 静态特征的列名列表
        
        可用的特征类型:
        - seq_length: 序列长度（非NaN值的数量）
        - max_value: 最大值
        - mean_value: 均值
        - min_value: 最小值
        - std_value: 标准差
        - trend: 趋势（线性回归斜率）
        - range_value: 数值范围（最大值-最小值）
        - kurtosis: 峰度
        - skewness: 偏度
        - autocorr: 一阶自相关系数
        
        返回:
        DataFrame: 合并后的特征数据框
        time_data: 原始时序数据
        """
        if feature_types is None:
            feature_types = ['seq_length', 'max_value', 'mean_value']
        
        # 识别时序特征列
        time_cols = [col for col in df.columns if 'min' in str(col).lower()]
        time_data = df[time_cols]
    
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

# 添加剂推荐页面
if page == "配方建议":
    sub_page = st.sidebar.selectbox("🔧 选择功能", ["添加剂推荐"])

    if sub_page == "添加剂推荐":
        st.subheader("添加剂推荐")
        
        @st.cache_resource
        def load_predictor():
            return Predictor(
                scaler_path="scaler_fold_1.pkl",
                svc_path="svc_fold_1.pkl"
            )
        
        # 动态生成输入表单
        with st.form("additive_form"):
            col1, col2 = st.columns(2)
            
            # 静态参数
            with col1:
                st.markdown("### 基础参数")
                sn_percent = st.number_input("Sn含量 (%)", 0.0, 100.0, 98.5)
                add_ratio = st.number_input("添加比例 (%)", 0.0, 100.0, 5.0)  # 恢复 ratio 参数
                yijia_percent = st.number_input("一甲胺含量 (%)", 0.0, 100.0, 0.5)
            
            # 时序参数
            with col2:
                st.markdown("### 黄度值时序参数")
                time_points = [3, 6, 9, 12, 15, 18, 21, 24]
                yellow_values = [
                    st.number_input(
                        f"{time}min 黄度值", 
                        min_value=0.0, 
                        max_value=10.0, 
                        value=1.2 + 0.3*i,
                        key=f"yellow_{time}"
                    )
                    for i, time in enumerate(time_points)
                ]
            
            submitted = st.form_submit_button("生成推荐")
        
        if submitted:
            try:
                # 构建完整输入样本（顺序必须与类定义一致！）
                sample = [
                    sn_percent,    # 对应 static_cols[0]
                    add_ratio,     # 对应 static_cols[1] 恢复 ratio 参数
                    yijia_percent, # 对应 static_cols[2]
                    *yellow_values # 展开时序参数
                ]
                
                predictor = load_predictor()
                result = predictor.predict_one(sample)
                
                # 推荐添加剂种类
                result_map = {
                    1: {"name": "无"},
                    2: {"name": "氯化石蜡"},
                    3: {"name": "EA12（脂肪酸复合醇酯）"},
                    4: {"name": "EA15（市售液体钙锌稳定剂）"},
                    5: {"name": "EA16（环氧大豆油）"},
                    6: {"name": "G70L（多官能团的脂肪酸复合酯混合物）"},
                    7: {"name": "EA6（亚磷酸酯）"}   
                }
                
                if result not in result_map:
                    raise ValueError("未知预测结果")
                
                st.success("### 推荐方案")
                st.markdown(f"""
                **推荐类型**: {result_map[result]['name']}
                - 适配工艺参数:
                  - 加工温度: 180-200℃
                  - 混料时间: 15-20分钟
                """)
                
            except Exception as e:
                st.error(f"""
                ## 预测失败
                错误: {str(e)}
                """)
