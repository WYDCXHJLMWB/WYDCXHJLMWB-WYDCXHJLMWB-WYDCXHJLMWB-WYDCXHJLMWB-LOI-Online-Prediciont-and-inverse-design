import numpy as np
import pandas as pd
import random
from sklearn.preprocessing import StandardScaler
import joblib

# 加载模型和缩放器
data = joblib.load("model_and_scaler_loi.pkl")
model = data["model"]
scaler = data["scaler"]

# 配方成分列表
feature_names = ["PP", "PAPP", "TCA", "ZHS", "ZS", "Mg(OH)2", "DOPO", "MPP", "Pentaerythritol", "APP",
                 "EBS", "Anti-drip-agent", "silane coupling agent", "Al(OH)3", "ZnB", "XiuCheng", "wollastonite", 
                 "XS-FR-8310", "XS-HFFR-8332", "ZBS-PV-OA", "M-2200B", "CFA", "ammonium octamolybdate", 
                 "antimony oxides", "antioxidant"]

def generate_recipe():
    # 初始随机生成配方
    recipe = np.random.uniform(0.01, 0.5, len(feature_names))  # 设置每个成分的范围为0.01到0.5，避免负数
    recipe = np.abs(recipe)  # 确保所有成分为正数

    # 确保总和为100
    total = np.sum(recipe)
    recipe = (recipe / total) * 100  # 将总和归一化为100

    # 输出配方：如果出现负值，确保变为0并重新调整
    recipe = {name: max(0, round(amount, 2)) for name, amount in zip(feature_names, recipe)}
    
    # 计算配方总和以确保其为100
    recipe_sum = sum(recipe.values())
    if recipe_sum != 100:
        # 如果总和不为100，归一化为100
        scale_factor = 100 / recipe_sum
        recipe = {k: round(v * scale_factor, 2) for k, v in recipe.items()}

    return recipe

# 示例：生成一个符合要求的配方
recipe = generate_recipe()
print("生成的配方：")
print(recipe)

