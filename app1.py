elif page == "配方建议":
    target_loi = st.number_input("🎯 请输入目标 LOI 值 (%)", value=20.0, step=0.1, min_value=10.0, max_value=40.0)
    output_mode = st.selectbox("📦 请选择输出形式", ["质量分数（wt%）", "质量（g）", "体积分数（vol%）"])

    if target_loi < 10 or target_loi > 40:
        st.warning("⚠️ 目标 LOI 值必须在 10 到 40 之间，请重新输入。")
    else:
        st.write("🔄 正在进行逆向设计，请稍等...")

        pp_index = feature_names.index("PP")
        num_features = len(feature_names)

        creator.create("FitnessMin", base.Fitness, weights=(-1.0,))
        creator.create("Individual", list, fitness=creator.FitnessMin)

        def make_valid_individual():
            ind = np.random.uniform(0.1, 1, num_features)
            ind[pp_index] = max(ind) + 0.1
            ind = np.clip(ind, 0, None)
            return creator.Individual(ind)

        def evaluate(ind):
            ind = np.clip(ind, 0, None)
            if ind[pp_index] <= max([x for i, x in enumerate(ind) if i != pp_index]):
                return 1e6,
            norm = ind / np.sum(ind) * 100  # 确保加和为100
            X_scaled = scaler.transform([norm])
            y_pred = model.predict(X_scaled)[0]
            return abs(y_pred - target_loi),

        toolbox = base.Toolbox()
        toolbox.register("individual", make_valid_individual)
        toolbox.register("population", tools.initRepeat, list, toolbox.individual)
        toolbox.register("evaluate", evaluate)
        toolbox.register("mate", tools.cxBlend, alpha=0.5)
        toolbox.register("mutate", tools.mutGaussian, mu=0.0, sigma=0.1, indpb=0.3)
        toolbox.register("select", tools.selTournament, tournsize=3)

        pop = toolbox.population(n=100)
        hof = tools.HallOfFame(20)

        algorithms.eaSimple(pop, toolbox, cxpb=0.7, mutpb=0.3, ngen=60, halloffame=hof, verbose=False)

        results = []
        for ind in hof:
            ind = np.clip(ind, 0, None)
            norm = ind / np.sum(ind) * 100
            if norm[pp_index] <= max([x for i, x in enumerate(norm) if i != pp_index]):
                continue
            pred_loi = model.predict(scaler.transform([norm]))[0]
            results.append(list(norm) + [pred_loi])

        if len(results) == 0:
            st.error("❌ 未能生成符合条件的配方，请尝试调整目标值或放宽条件。")
        else:
            df_result = pd.DataFrame(results[:10], columns=feature_names + ["预测 LOI"])

            if output_mode == "质量（g）":
                df_result.iloc[:, :-1] = df_result.iloc[:, :-1] * 1.0  # 总质量100g
                df_result.columns = [f"{col} (g)" if col != "预测 LOI" else col for col in df_result.columns]
            elif output_mode == "质量分数（wt%）":
                df_result.columns = [f"{col} (wt%)" if col != "预测 LOI" else col for col in df_result.columns]
            elif output_mode == "体积分数（vol%）":
                volume_fractions = df_result.iloc[:, :-1].div(df_result.iloc[:, :-1].sum(axis=1), axis=0) * 100
                df_result.iloc[:, :-1] = volume_fractions
                df_result.columns = [f"{col} (vol%)" if col != "预测 LOI" else col for col in df_result.columns]

            st.markdown("### 📋 推荐配方")
            st.dataframe(df_result.round(2))
