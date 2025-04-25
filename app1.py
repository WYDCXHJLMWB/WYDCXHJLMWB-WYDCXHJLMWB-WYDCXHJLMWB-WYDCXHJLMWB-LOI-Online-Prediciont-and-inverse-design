# 配方建议部分修改
elif page == "配方建议":
    st.subheader("🧪 配方建议：根据性能反推配方")

    # 用户输入目标LOI值并确保范围在10到50之间
    target_loi = st.number_input("请输入目标极限氧指数 (LOI)", min_value=10.0, max_value=50.0, value=25.0, step=0.1)

    # 如果用户输入的目标LOI超出范围，提醒用户
    if target_loi < 10 or target_loi > 50:
        st.warning("⚠️ 目标LOI应在10到50之间，请重新输入。")

    # 添加遗传算法的部分
    creator.create("FitnessMin", base.Fitness, weights=(-1.0,))  # 最小化目标
    creator.create("Individual", list, fitness=creator.FitnessMin)

    # 示例：用遗传算法生成配方
    toolbox = base.Toolbox()
    toolbox.register("attr_float", np.random.uniform, 0.01, 0.5)  # 设置最小值为0.01，避免负数
    toolbox.register("individual", tools.initRepeat, creator.Individual, toolbox.attr_float, n=len(feature_names))
    toolbox.register("population", tools.initRepeat, list, toolbox.individual)

    def evaluate(individual):
        # 将个体（配方）转换为字典形式
        user_input = dict(zip(feature_names, individual))
        
        # 保证配方总和为100，必要时进行调整
        total = sum(user_input.values())
        if total != 100:
            user_input = {k: (v / total) * 100 for k, v in user_input.items()}  # 归一化为质量分数

        # 使用模型进行LOI预测
        input_array = np.array([list(user_input.values())])
        input_scaled = scaler.transform(input_array)
        predicted_loi = model.predict(input_scaled)[0]
        
        # 返回LOI与目标LOI之间的差异，作为目标函数值
        return abs(predicted_loi - target_loi),  # 返回元组，符合遗传算法的要求

    toolbox.register("mate", tools.cxBlend, alpha=0.5)
    toolbox.register("mutate", tools.mutGaussian, mu=0, sigma=0.1, indpb=0.2)
    toolbox.register("select", tools.selTournament, tournsize=3)
    toolbox.register("evaluate", evaluate)

    population = toolbox.population(n=50)
    
    # 开始推荐配方按钮
    if st.button("开始推荐配方"):
        # 使用遗传算法生成配方
        for gen in range(10):  # 10代
            offspring = toolbox.select(population, len(population))
            offspring = list(map(toolbox.clone, offspring))
            for child1, child2 in zip(offspring[::2], offspring[1::2]):
                if np.random.rand() < 0.7:
                    toolbox.mate(child1, child2)
                    del child1.fitness.values
                    del child2.fitness.values
            for mutant in offspring:
                if np.random.rand() < 0.2:
                    toolbox.mutate(mutant)
                    del mutant.fitness.values
            invalid_individuals = [ind for ind in offspring if not ind.fitness.valid]
            fitnesses = list(map(toolbox.evaluate, invalid_individuals))
            for ind, fit in zip(invalid_individuals, fitnesses):
                ind.fitness.values = fit
            population[:] = offspring

        # 从最后一代中选出最好的配方
        best_individual = tools.selBest(population, 1)[0]
        st.write("最佳配方:", dict(zip(feature_names, best_individual)))
