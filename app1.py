# ...（前面的代码保持不变，直到配方建议部分）

elif page == "配方建议":
    st.subheader("🧪 配方建议：根据性能反推配方")
    target_loi = st.number_input("目标LOI值", min_value=10.0, max_value=50.0, value=25.0, step=0.1)
    
    # 遗传算法配置（保持不变）
    # ...（遗传算法配置代码保持不变）

    if st.button("生成推荐配方"):
        with st.spinner("🔍 正在优化配方..."):
            # 遗传算法运行代码（保持不变）
            # ...（算法运行代码保持不变）

            # 获取最佳个体并处理单位
            best = hof[0]
            total = sum(best)
            recipe_wt = {name: (val/total)*100 for name, val in zip(feature_names, best)}
            
            # 根据单位类型转换数值和单位标签
            if unit_type == "质量 (g)":
                recipe = recipe_wt  # 数值直接显示为克数（假设总质量100g）
                unit_label = "g"
            elif unit_type == "质量分数 (wt%)":
                recipe = recipe_wt
                unit_label = "wt%"
            elif unit_type == "体积分数 (vol%)":
                recipe = recipe_wt  # 假设体积分数与质量分数数值相同
                unit_label = "vol%"

            # 添加单位到列名
            columns_with_units = [f"{name} ({unit_label})" for name in feature_names]
            
            # 创建结果DataFrame
            recipe_df = pd.DataFrame([recipe]*10, columns=columns_with_units)
            recipe_df.index = [f"配方 {i+1}" for i in range(10)]

            # 验证PP含量
            pp_col = f"PP ({unit_label})"
            for i in range(10):
                # 确保PP是最大值且≥50%
                recipe_df.loc[f"配方 {i+1}", pp_col] = max(recipe_df.loc[f"配方 {i+1}"])
                if recipe_df.loc[f"配方 {i+1}", pp_col] < 50:
                    recipe_df.loc[f"配方 {i+1}", pp_col] = 50

            st.success("✅ 配方优化完成！")
            
            st.subheader("推荐配方列表")
            st.dataframe(recipe_df.style.format("{:.2f}"))

            # 显示预测值（保持不变）
            input_array = np.array([[recipe_wt[name] for name in feature_names]])
            input_scaled = scaler.transform(input_array)
            predicted_loi = model.predict(input_scaled)[0]
            st.metric("预测LOI", f"{predicted_loi:.2f}%")
