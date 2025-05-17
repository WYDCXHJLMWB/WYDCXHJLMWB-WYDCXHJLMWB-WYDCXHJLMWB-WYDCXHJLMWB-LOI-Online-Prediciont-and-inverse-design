import pandas as pd
import numpy as np
import warnings
from scipy import stats
from scipy.stats import kurtosis, skew
from sklearn.impute import SimpleImputer
import joblib
from deap import base, creator, tools, algorithms
import streamlit as st
import base64
import random
from PIL import Image
import io
import json
import hashlib
import os
# 在文件顶部添加版本兼容性处理
def st_rerun():
    """兼容所有 Streamlit 版本的刷新函数"""
    if hasattr(st, "rerun"):
        st.rerun()
    elif hasattr(st, "experimental_rerun"):
        st.rerun()
    else:
        raise st.stop()  #
USER_DATA_FILE = "users.json"

# 读取用户数据（用户名: 密码哈希）
def load_users():
    if not os.path.exists(USER_DATA_FILE):
        return {}
    with open(USER_DATA_FILE, "r") as f:
        return json.load(f)

# 保存用户数据
def save_users(users):
    with open(USER_DATA_FILE, "w") as f:
        json.dump(users, f)

# 计算密码哈希（简单SHA256）
def hash_password(password):
    return hashlib.sha256(password.encode()).hexdigest()
warnings.filterwarnings("ignore")

# ========================== 初始化状态 ==========================
if 'logged_in' not in st.session_state:
    st.session_state.logged_in = False
if 'current_page' not in st.session_state:
    st.session_state.current_page = "首页"

# ========================== Predictor类定义 ==========================
class Predictor:
    def __init__(self, scaler_path, svc_path):
        self.scaler = joblib.load(scaler_path)
        self.model = joblib.load(svc_path)
        
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

    def create_features(self, data):
        # 静态特征处理
        static_features = data[self.static_cols]
        static_features = self.imputer.fit_transform(static_features)
        
        # 时间序列特征工程
        time_series = data[self.time_series_cols]
        features = []
        for idx, row in time_series.iterrows():
            seq = row.values
            seq = seq[~np.isnan(seq)]
            
            # 统计特征
            max_val = np.max(seq)
            min_val = np.min(seq)
            mean_val = np.mean(seq)
            std_val = np.std(seq)
            trend_val = (seq[-1] - seq[0]) / len(seq)
            range_val = max_val - min_val
            autocorr_val = np.correlate(seq - mean_val, seq - mean_val, mode='full')[len(seq)-1] / (std_val**2 * len(seq))
            
            # 组合特征
            features.append([
                len(seq), max_val, mean_val, min_val,
                std_val, trend_val, range_val, autocorr_val
            ])
        
        return np.hstack([static_features, np.array(features)])

    def predict_one(self, sample):
        sample_df = pd.DataFrame([sample], columns=self.static_cols + self.time_series_cols)
        features = self.create_features(sample_df)
        scaled_features = self.scaler.transform(features)
        return self.model.predict(scaled_features)[0]

# ========================== 全局配置和样式 ==========================
def image_to_base64(image_path, quality=95):
    img = Image.open(image_path)
    if img.width != 1000:
        img = img.resize((500, int(img.height * (1000 / img.width))), 
                        resample=Image.LANCZOS)
    buffered = io.BytesIO()
    img.save(buffered, format="PNG", quality=quality, optimize=True)
    return base64.b64encode(buffered.getvalue()).decode()

image_path = "图片1.jpg"
icon_base64 = image_to_base64(image_path)

st.set_page_config(
    page_title="阻燃聚合物复合材料智能设计平台",
    layout="wide",
    page_icon=f"data:image/png;base64,{icon_base64}"
)

st.markdown(f"""
<style>
    .global-header {{
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
    }}
    
    .feature-grid {{
        display: grid;
        grid-template-columns: repeat(auto-fit, minmax(280px, 1fr));
        gap: 1.5rem;
        margin: 2rem 0;
    }}
    
    .login-container {{
        max-width: 400px;
        margin: 2rem auto;
        padding: 2rem;
        background: #f8f9fa;
        border-radius: 10px;
        box-shadow: 0 2px 8px rgba(0,0,0,0.1);
    }}
    
    .feature-card {{
        padding: 1.5rem;
        background: #ffffff;
        border-radius: 10px;
        box-shadow: 0 2px 8px rgba(0,0,0,0.1);
        transition: transform 0.2s;
    }}
    
    .feature-card:hover {{
        transform: translateY(-5px);
    }}
    
    .card-title {{
        color: #1e3d59;
        margin-bottom: 0.8rem;
    }}
    
    .section-title {{
        color: #1e3d59;
        font-size: 1.5rem;
        margin: 2rem 0 1rem;
        padding-bottom: 0.5rem;
        border-bottom: 2px solid #1e3d59;
    }}
    
    .quote-section {{
        background: #f8f9fa;
        padding: 1.5rem;
        border-left: 4px solid #1e3d59;
        margin: 1.5rem 0;
    }}
</style>
""", unsafe_allow_html=True)

# ========================== 页面组件 ==========================
def show_header():
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

def navigation():
    st.sidebar.title("🔧 导航菜单")

    # 决定显示哪些页面
    if st.session_state.get("logged_in", False):
        pages = ["首页", "性能预测", "配方建议", "退出登录"]
    else:
        pages = ["首页", "用户登录"]

    # 当前页面初始设置（第一次运行）
    if "current_page" not in st.session_state:
        st.session_state.current_page = "首页"

    # 用户点击的页面
    selection = st.sidebar.radio("选择页面", pages, index=pages.index(st.session_state.current_page))

    # 如果用户选择的是退出登录，先设置状态，再重新加载
    if selection == "退出登录":
        st.session_state.logged_in = False
        st.session_state.current_page = "首页"
        st.rerun()
        return  # 避免继续执行后续逻辑

    # 切换页面，更新当前页并刷新
    if selection != st.session_state.current_page:
        st.session_state.current_page = selection
        st.experimental_rerun()


# ========================== 页面内容 ==========================
def home_page():
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
    
        st.markdown("""
        <style>
            .feature-list {
                list-style: none; /* 移除默认列表符号 */
                padding-left: 0;  /* 移除默认左内边距 */
            }
            .feature-list li:before {
                content: "•";
                color: var(--secondary);
                font-size: 1.5em;
                position: relative;
                left: -0.8em;    /* 微调定位 */
                vertical-align: middle;
            }
            .feature-list li {
                margin-left: 1.2em;  /* 给符号留出空间 */
                text-indent: -1em;   /* 文本缩进对齐 */
            }
        </style>
        
        <div class="section-title">核心功能</div>
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
def login_page():
    users = load_users()

    mode = st.radio("选择操作", ["登录", "注册", "忘记密码"])

    if mode == "登录":
        with st.container():
            col1, col2 = st.columns([1, 1])
            with col1:
                with st.form("login_form"):
                    st.markdown("""
                    <div class="login-container">
                        <h2 class="login-title">🔐 用户登录</h2>
                    """, unsafe_allow_html=True)

                    username = st.text_input("用户名")
                    password = st.text_input("密码", type="password")
                    login_button = st.form_submit_button("登录")

                    if login_button:
                        pw_hash = hash_password(password)
                        if username in users and users[username] == pw_hash:
                            st.session_state.logged_in = True
                            st.session_state.current_page = "首页"
                            st.rerun()
                        else:
                            st.error("用户名或密码错误")

                    st.markdown("</div>", unsafe_allow_html=True)

            with col2:
                st.markdown("""
                <div style="padding: 2rem; background: #f8f9fa; border-radius: 10px;">
                    <h3>📢 使用说明</h3>
                    <p>1. 登录后可访问完整功能</p>
                    <p>2. 注册后即可登录</p>
                </div>
                """, unsafe_allow_html=True)

    elif mode == "注册":
        st.subheader("📝 用户注册")
        with st.form("register_form"):
            new_username = st.text_input("新用户名")
            new_password = st.text_input("新密码", type="password")
            confirm_password = st.text_input("确认密码", type="password")
            register_button = st.form_submit_button("注册")

            if register_button:
                if not new_username or not new_password:
                    st.error("用户名和密码不能为空")
                elif new_username in users:
                    st.error("用户名已存在，请选择其他用户名")
                elif new_password != confirm_password:
                    st.error("两次密码输入不一致")
                else:
                    users[new_username] = hash_password(new_password)
                    save_users(users)
                    st.success(f"用户 {new_username} 注册成功！请返回登录。")

    else:  # 忘记密码
        st.subheader("🔑 忘记密码")
        with st.form("forgot_form"):
            forget_username = st.text_input("请输入用户名")
            reset_button = st.form_submit_button("重置密码")

            if reset_button:
                if not forget_username:
                    st.error("请输入用户名")
                elif forget_username not in users:
                    st.error("用户名不存在")
                else:
                    # 简单重置为默认密码 "123456" 并保存
                    users[forget_username] = hash_password("123456")
                    save_users(users)
                    st.success(f"用户 {forget_username} 的密码已重置为默认密码：123456，请登录后尽快修改密码。")


def prediction_page():
    st.subheader("🔮 性能预测：基于配方预测LOI和TS")
    
    # 加载模型
    models = {
        "loi_model": joblib.load("loi_model.pkl"),
        "ts_model": joblib.load("ts_model.pkl"),
        "loi_scaler": joblib.load("loi_scaler.pkl"),
        "ts_scaler": joblib.load("ts_scaler.pkl"),
        "loi_features": ["PP", "PAPP", "ZS", "Anti-drip-agent", "MPP", "wollastonite"],
        "ts_features": ["PP", "PAPP", "ZS", "Anti-drip-agent", "MPP", "wollastonite"]
    }

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
    
    input_values[selected_matrix] = st.number_input(
        f"选择 {selected_matrix} ({unit_matrix})", min_value=0.0, max_value=100.0, value=50.0, step=0.1
    )
    
    for fr in selected_flame_retardants:
        input_values[fr] = st.number_input(
            f"选择 {fr}({unit_flame_retardant})", min_value=0.0, max_value=100.0, value=10.0, step=0.1
        )
    
    for ad in selected_additives:
        input_values[ad] = st.number_input(
            f"选择 {ad} ({unit_additive})", min_value=0.0, max_value=100.0, value=10.0, step=0.1
        )
    
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
    # 🧪 模型精度验证（添加在开始预测按钮之前）
    with st.expander("📊 模型精度验证"):
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
    
        col1, col2, col3 = st.columns(3)
        all_features = set(models["loi_features"]) | set(models["ts_features"])
    
        for i, sample in enumerate(samples):
            with [col1, col2, col3][i]:
                st.markdown(f"### {sample['name']}")
                st.write("配方：")
                for ingredient, value in sample["配方"].items():
                    st.write(f"  - {ingredient}: {value} wt %")
    
        for sample in samples:
            input_vector = {feature: 0.0 for feature in all_features}
            for k, v in sample["配方"].items():
                if k not in input_vector:
                    st.warning(f"检测到样本中存在模型未定义的特征: {k}")
                input_vector[k] = v
    
            try:
                loi_input = np.array([[input_vector[f] for f in models["loi_features"]]])
                loi_scaled = models["loi_scaler"].transform(loi_input)
                loi_pred = models["loi_model"].predict(loi_scaled)[0]
            except KeyError as e:
                st.error(f"LOI模型特征缺失: {e}，请检查模型配置")
                st.stop()
    
            try:
                ts_input = np.array([[input_vector[f] for f in models["ts_features"]]])
                ts_scaled = models["ts_scaler"].transform(ts_input)
                ts_pred = models["ts_model"].predict(ts_scaled)[0]
            except KeyError as e:
                st.error(f"TS模型特征缺失: {e}，请检查模型配置")
                st.stop()
    
            loi_error = abs(sample["LOI_真实值"] - loi_pred) / sample["LOI_真实值"] * 100
            ts_error = abs(sample["TS_真实值"] - ts_pred) / sample["TS_真实值"] * 100
            loi_color = "green" if loi_error < 15 else "red"
            ts_color = "green" if ts_error < 15 else "red"
    
            with [col1, col2, col3][samples.index(sample)]:
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
            # 👉 如果是体积分数，转质量分数（默认处理）
            if fraction_type == "体积分数":
                vol_values = np.array(list(input_values.values()))
                mass_values = vol_values
                total_mass = mass_values.sum()
                input_values = {
                    k: (v / total_mass * 100) for k, v in zip(input_values.keys(), mass_values)
                }
    
            # ✅ 如果是质量分数，自动换算成质量（默认总质量为100g）
            if fraction_type == "质量分数":
                total_mass = 100.0  # 默认总质量
                input_values = {
                    k: v / 100.0 * total_mass for k, v in input_values.items()
                }
    
            # 🧠 填充模型所需的缺失特征
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

def formula_page():
    # 子页面选择
    sub_page = st.sidebar.selectbox(
        "🔧 子功能选择",
        ["配方优化", "添加剂推荐"],
        key="sub_nav"
    )

    if sub_page == "配方优化":
        st.subheader("🧪 配方优化建议")

        # 加载模型
        models = {
            "loi_model": joblib.load("loi_model.pkl"),
            "ts_model": joblib.load("ts_model.pkl"),
            "loi_scaler": joblib.load("loi_scaler.pkl"),
            "ts_scaler": joblib.load("ts_scaler.pkl"),
            "loi_features": ["PP", "PAPP", "ZS", "Anti-drip-agent", "MPP", "wollastonite"],
            "ts_features": ["PP", "PAPP", "ZS", "Anti-drip-agent", "MPP", "wollastonite"]
        }

        # 单位类型选择
        fraction_type = st.sidebar.radio(
            "📐 单位类型",
            ["质量", "质量分数", "体积分数"],
            key="unit_type"
        )

        # 材料选择
        matrix_materials = ["PP", "PA", "PC/ABS", "POM", "PBT", "PVC", "其他"]
        flame_retardants = [
            "AHP", "ammonium octamolybdate", "Al(OH)3", "CFA", "APP", "Pentaerythritol", "DOPO",
            "EPFR-1100NT", "XS-FR-8310", "ZS", "XiuCheng", "ZHS", "ZnB", "antimony oxides",
            "Mg(OH)2", "TCA", "MPP", "PAPP", "其他"
        ]
        additives = [
            "Anti-drip-agent", "wollastonite", "M-2200B", "ZBS-PV-OA", "FP-250S",
            "silane coupling agent", "antioxidant", "SiO2", "其他"
        ]

        col1, col2 = st.columns(2)
        with col1:
            selected_matrix = st.selectbox("选择基体材料", matrix_materials, index=0)
        with col2:
            selected_flame_retardants = st.multiselect("选择阻燃剂", flame_retardants, default=["ZS"])

        selected_additives = st.multiselect("选择助剂", additives, default=["wollastonite"])

        # 目标值输入
        target_loi = st.number_input("目标LOI值（%）", min_value=0.0, max_value=100.0, value=30.0)
        target_ts = st.number_input("目标TS值（MPa）", min_value=0.0, value=40.0)

        if st.button("🚀 开始优化"):
            all_features = [selected_matrix] + selected_flame_retardants + selected_additives

            creator.create("FitnessMin", base.Fitness, weights=(-1.0, -1.0))
            creator.create("Individual", list, fitness=creator.FitnessMin)
            toolbox = base.Toolbox()

            def repair_individual(individual):
                individual = [max(0.0, x) for x in individual]
                total = sum(individual)
                if total <= 1e-6:
                    return [100.0 / len(individual)] * len(individual)

                scale = 100.0 / total
                individual = [x * scale for x in individual]

                try:
                    matrix_idx = all_features.index(selected_matrix)
                    matrix_value = individual[matrix_idx]
                    other_max = max([v for i, v in enumerate(individual) if i != matrix_idx], default=0)

                    if matrix_value <= other_max:
                        delta = other_max - matrix_value + 0.01
                        others_total = sum(v for i, v in enumerate(individual) if i != matrix_idx)
                        if others_total > 0:
                            deduction_ratio = delta / others_total
                            for i in range(len(individual)):
                                if i != matrix_idx:
                                    individual[i] *= (1 - deduction_ratio)
                            individual[matrix_idx] += delta * others_total / others_total

                        total = sum(individual)
                        scale = 100.0 / total
                        individual = [x * scale for x in individual]
                except ValueError:
                    pass

                return individual

            def generate_individual():
                try:
                    matrix_idx = all_features.index(selected_matrix)
                except ValueError:
                    matrix_idx = 0

                matrix_range = (60, 100) if selected_matrix == "PP" else (30, 50)
                matrix_percent = random.uniform(*matrix_range)

                remaining = 100 - matrix_percent
                n_others = len(all_features) - 1
                if n_others == 0:
                    return [matrix_percent]

                others = np.random.dirichlet(np.ones(n_others) * 0.5) * remaining
                individual = [0.0] * len(all_features)
                individual[matrix_idx] = matrix_percent

                other_idx = 0
                for i in range(len(all_features)):
                    if i != matrix_idx:
                        individual[i] = others[other_idx]
                        other_idx += 1

                return repair_individual(individual)

            toolbox.register("individual", tools.initIterate, creator.Individual, generate_individual)
            toolbox.register("population", tools.initRepeat, list, toolbox.individual)

            def evaluate(individual):
                try:
                    input_values = dict(zip(all_features, individual))
                    loi_input = np.array([[input_values.get(f, 0.0) for f in models["loi_features"]]])
                    loi_scaled = models["loi_scaler"].transform(loi_input)
                    loi_pred = models["loi_model"].predict(loi_scaled)[0]

                    ts_input = np.array([[input_values.get(f, 0.0) for f in models["ts_features"]]])
                    ts_scaled = models["ts_scaler"].transform(ts_input)
                    ts_pred = models["ts_model"].predict(ts_scaled)[0]

                    return (abs(target_loi - loi_pred), abs(target_ts - ts_pred))
                except Exception as e:
                    print(f"Error in evaluate: {e}")
                    return (float('inf'), float('inf'))

            def cxBlendWithConstraint(ind1, ind2, alpha):
                tools.cxBlend(ind1, ind2, alpha)
                ind1[:] = repair_individual(ind1)
                ind2[:] = repair_individual(ind2)
                return ind1, ind2

            def mutGaussianWithConstraint(individual, mu, sigma, indpb):
                tools.mutGaussian(individual, mu, sigma, indpb)
                individual[:] = repair_individual(individual)
                return individual,

            toolbox.register("evaluate", evaluate)
            toolbox.register("mate", cxBlendWithConstraint, alpha=0.5)
            toolbox.register("mutate", mutGaussianWithConstraint, mu=0, sigma=3, indpb=0.2)
            toolbox.register("select", tools.selNSGA2)

            population = toolbox.population(n=150)
            algorithms.eaMuPlusLambda(
                population, toolbox,
                mu=150, lambda_=300,
                cxpb=0.7, mutpb=0.3,
                ngen=250, verbose=False
            )

            valid_individuals = [ind for ind in population if not np.isinf(ind.fitness.values[0])]
            best_individuals = tools.selBest(valid_individuals, k=5)

            results = []
            for ind in best_individuals:
                normalized = [round(x, 2) for x in repair_individual(ind)]
                matrix_value = normalized[all_features.index(selected_matrix)]

                if not all(v <= matrix_value for i, v in enumerate(normalized) if i != all_features.index(selected_matrix)):
                    continue

                input_dict = dict(zip(all_features, normalized))
                loi_input = [[input_dict.get(f, 0) for f in models["loi_features"]]]
                loi_scaled = models["loi_scaler"].transform(loi_input)
                loi_pred = models["loi_model"].predict(loi_scaled)[0]

                ts_input = [[input_dict.get(f, 0) for f in models["ts_features"]]]
                ts_scaled = models["ts_scaler"].transform(ts_input)
                ts_pred = models["ts_model"].predict(ts_scaled)[0]

                if abs(target_loi - loi_pred) > 10 or abs(target_ts - ts_pred) > 10:
                    continue

                results.append({
                    **{f: normalized[i] for i, f in enumerate(all_features)},
                    "LOI预测值 (%)": round(loi_pred, 2),
                    "TS预测值 (MPa)": round(ts_pred, 2),
                })

            if results:
                df = pd.DataFrame(results)
                unit = "wt%" if "质量分数" in fraction_type else "vol%" if "体积分数" in fraction_type else "g"
                df.columns = [f"{col} ({unit})" if col in all_features else col for col in df.columns]

                st.dataframe(
                    df.style.apply(lambda x: ["background: #e6ffe6" if x["LOI预测值 (%)"] >= target_loi and
                                              x["TS预测值 (MPa)"] >= target_ts else "" for _ in x], axis=1),
                    height=400
                )
            else:
                st.warning("未找到符合要求的配方，请尝试调整目标值")
    elif sub_page == "添加剂推荐":
        st.subheader("🧪 PVC添加剂智能推荐")
        predictor = Predictor("scaler_fold_1.pkl", "svc_fold_1.pkl")
        with st.expander("点击查看参考样本"):
            st.markdown("""
                ### 参考样本
                以下是一些参考样本，展示了不同的输入数据及对应的推荐添加剂类型：
            """)
            
            # 参考样本数据
            sample_data = [
                ["样本1", "无添加剂", 
                    {"Sn%": 19.2, "添加比例": 0, "一甲%": 32, "黄度值_3min": 5.36, "黄度值_6min": 6.29, "黄度值_9min": 7.57, "黄度值_12min": 8.57, "黄度值_15min": 10.26, "黄度值_18min": 13.21, "黄度值_21min": 16.54, "黄度值_24min": 27.47}],
                ["样本2", "氯化石蜡", 
                    {"Sn%": 18.5, "添加比例": 3.64, "一甲%": 31.05, "黄度值_3min": 5.29, "黄度值_6min": 6.83, "黄度值_9min": 8.00, "黄度值_12min": 9.32, "黄度值_15min": 11.40, "黄度值_18min": 14.12, "黄度值_21min": 18.37, "黄度值_24min": 30.29}],
                ["样本3", "EA15（市售液体钙锌稳定剂）", 
                    {"Sn%": 19, "添加比例": 1.041666667, "一甲%": 31.88, "黄度值_3min": 5.24, "黄度值_6min": 6.17, "黄度值_9min": 7.11, "黄度值_12min": 8.95, "黄度值_15min": 10.33, "黄度值_18min": 13.21, "黄度值_21min": 17.48, "黄度值_24min": 28.08}]
            ]
            
            # 为每个样本创建一个独立的表格
            for sample in sample_data:
                sample_name, additive, features = sample
                st.markdown(f"#### {sample_name} - {additive}")
                
                # 将数据添加到表格
                features["推荐添加剂"] = additive  # 显示样本推荐的添加剂
                features["推荐添加量 (%)"] = features["添加比例"]  # 使用已提供的添加比例
                
                # 转换字典为 DataFrame
                df_sample = pd.DataFrame(list(features.items()), columns=["特征", "值"])
                st.table(df_sample)  # 显示为表格形式
    
        # 修改黄度值输入为独立输入
        with st.form("additive_form"):
            st.markdown("### 基础参数")
            col_static = st.columns(3)
            with col_static[0]:
                add_ratio = st.number_input("添加比例 (%)", 
                                            min_value=0.0,
                                            max_value=100.0,
                                            value=3.64,
                                            step=0.1)
            with col_static[1]:
                sn_percent = st.number_input("Sn含量 (%)", 
                                            min_value=0.0, 
                                            max_value=100.0,
                                            value=18.5,
                                            step=0.1,
                                            help="锡含量范围0%~100%")
            with col_static[2]:
                yijia_percent = st.number_input("一甲含量 (%)",
                                               min_value=0.0,
                                               max_value=100.0,
                                               value=31.05,
                                               step=0.1,
                                               help="一甲胺含量范围15.1%~32%")
            
            st.markdown("### 黄度值")
            yellow_values = {}
            col1, col2, col3, col4 = st.columns(4)
            yellow_values["3min"] = st.number_input("3min 黄度值", min_value=0.0, max_value=100.0, value=5.29, step=0.1)
            yellow_values["6min"] = st.number_input("6min 黄度值", min_value=yellow_values["3min"], max_value=100.0, value= 6.83, step=0.1)
            yellow_values["9min"] = st.number_input("9min 黄度值", min_value=yellow_values["6min"], max_value=100.0, value=8.00, step=0.1)
            yellow_values["12min"] = st.number_input("12min 黄度值", min_value=yellow_values["9min"], max_value=100.0, value=9.32, step=0.1)
            yellow_values["15min"] = st.number_input("15min 黄度值", min_value=yellow_values["12min"], max_value=100.0, value=11.40, step=0.1)
            yellow_values["18min"] = st.number_input("18min 黄度值", min_value=yellow_values["15min"], max_value=100.0, value=14.12, step=0.1)
            yellow_values["21min"] = st.number_input("21min 黄度值", min_value=yellow_values["18min"], max_value=100.0, value=18.37, step=0.1)
            yellow_values["24min"] = st.number_input("24min 黄度值", min_value=yellow_values["21min"], max_value=100.0, value=30.29, step=0.1)
        
            submit_btn = st.form_submit_button("生成推荐方案")
        
        # 如果提交了表单，进行数据验证和预测
        if submit_btn:
            # 验证比例是否符合要求：每个黄度值输入必须满足递增条件
            if any(yellow_values[t] > yellow_values[next_time] for t, next_time in zip(yellow_values.keys(), list(yellow_values.keys())[1:])):
                st.error("错误：黄度值必须随时间递增！请检查输入数据")
                st.stop()
            
            # 构建输入样本
            sample = [
                sn_percent, add_ratio, yijia_percent,
                yellow_values["3min"], yellow_values["6min"],
                yellow_values["9min"], yellow_values["12min"],
                yellow_values["15min"], yellow_values["18min"],
                yellow_values["21min"], yellow_values["24min"]
            ]
        
            # 进行预测
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
        
            # 动态确定添加量和显示名称
            additive_amount = 0.0 if prediction == 1 else add_ratio
            additive_name = result_map[prediction]
        
            # 构建配方表
            formula_data = [
                ["PVC份数", 100.00],
                ["加工助剂ACR份数", 1.00],
                ["外滑剂70S份数", 0.35],
                ["MBS份数", 5.00],
                ["316A份数", 0.20],
                ["稳定剂份数", 1.00]
            ]
        
            if prediction != 1:
                formula_data.append([f"{additive_name}含量（wt%）", additive_amount])
            else:
                formula_data.append([additive_name, additive_amount])
        
            # 创建格式化表格
            df = pd.DataFrame(formula_data, columns=["材料名称", "含量"])
            styled_df = df.style.format({"含量": "{:.2f}"})\
                                  .hide(axis="index")\
                                  .set_properties(**{'text-align': 'left'})
        
            # 展示推荐结果
            col1, col2 = st.columns([1, 2])
            with col1:
                st.success(f"**推荐添加剂类型**  \n{additive_name}")
                st.metric("建议添加量", 
                         f"{additive_amount:.2f}%",
                         delta="无添加" if prediction == 1 else None)
            with col2:
                st.markdown("**完整配方表（基于PVC 100份）**")
           


# ========================== 主程序 ==========================
def main():
    show_header()
    navigation()
    
    if st.session_state.current_page == "首页":
        home_page()
    elif st.session_state.current_page == "用户登录":
        login_page()
    elif st.session_state.current_page == "性能预测":
        prediction_page()
    elif st.session_state.current_page == "配方建议":
        formula_page()
    
    # 页脚
    st.markdown("""
    <hr>
    <footer style="text-align: center;">
        <p>© 2025 阻燃聚合物复合材料智能设计平台</p>
        <p>声明：本平台仅供学术研究、技术验证等非营利性科研活动使用，严禁用于任何商业用途。</p>
    </footer>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()
