# -*- coding: utf-8 -*-
"""
Created on Mon Oct 28 19:46:55 2024

@author: DELL
"""

#加载包
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import random
import seaborn as sns
plt.rcParams['font.sans-serif']=['SimHei'] # 中文支持
plt.rcParams['axes.unicode_minus']=False

#读取数据
results = pd.read_csv("E:/桌面/results/月库存量+销量.csv", 
                      encoding = 'gbk')
inventory_forecast = results.iloc[:, : -3]
sales_forecast = results.drop(columns = ['7月库存量', 
                                         '8月库存量', 
                                         '9月库存量'])
warehouse_data= pd.read_csv("E:/桌面/附件3.csv",encoding='gbk')
warehouses = warehouse_data['仓库']
categories = results['品类']

#计算每个品类最大库存量
inventory_forecast['I_i'] = inventory_forecast[['7月库存量', 
                                                '8月库存量', 
                                                '9月库存量']].max(axis=1)

# 计算每个品类的平均每日销量S_i
sales_forecast['S_i'] = sales_forecast[['7月销量', 
                                        '8月销量', 
                                        '9月销量']].sum(axis=1) / 3

# 参数提取
# 每种品类的库存需求量
I_i_dict = inventory_forecast.set_index('品类')['I_i'].to_dict()
# 每种品类的销售需求量  
S_i_dict = sales_forecast.set_index('品类')['S_i'].to_dict()  
# 每个仓库的最大仓容
C_j_dict = warehouse_data.set_index('仓库')['仓容上限'].to_dict() 
# 每个仓库的最大产能 
P_j_dict = warehouse_data.set_index('仓库')['产能上限'].to_dict() 
# 每个仓库的日租赁成本 
R_j_dict = warehouse_data.set_index('仓库')['仓租日成本'].to_dict() 

# 遗传算法参数设置
population_size = 100  # 种群大小
generations = 3000  # 遗传算法迭代代数
mutation_rate = 0.01  # 基因突变率
tournament_size = 5  # 选择操作中锦标赛的参赛个体数量
penalty_factor = 1e6  # 违反约束时的惩罚系数

# 适应度函数

def fitness(individual):
    # 计算总仓租成本（乘以90表示为90天成本）
    total_rental_cost = sum(R_j_dict[warehouses[ind]] * 90 for ind in individual)
    
    # 计算总仓容利用率倒数形式
    total_capacity_utilization = sum(I_i_dict[categories[i]] / C_j_dict[warehouses[individual[i]]] for i in range(len(individual)))
    
    # 计算总产能利用率倒数形式
    total_production_utilization = sum(S_i_dict[categories[i]] / P_j_dict[warehouses[individual[i]]] for i in range(len(individual)))
    
    # 检查是否违反约束条件，并在违反时加入惩罚项
    penalty = 0
    
    # 1. 一品一仓约束：每个品类只能被分配到一个仓库
    if len(set(individual)) != len(individual):
        penalty += penalty_factor  

    # 2. 仓容约束：每个仓库的总库存量不能超过其仓容上限
    for j in range(len(warehouses)):
        allocated_inventory = sum(I_i_dict[categories[i]] for i in range(len(individual)) if individual[i] == j)
        if allocated_inventory > C_j_dict[warehouses[j]]:
            penalty += penalty_factor  

    # 3. 产能约束：每个仓库的出库能力不能超过其产能上限
    for j in range(len(warehouses)):
        allocated_sales = sum(S_i_dict[categories[i]] for i in range(len(individual)) if individual[i] == j)
        if allocated_sales > P_j_dict[warehouses[j]]:
            penalty += penalty_factor  

    # 4. 逻辑关系约束：产品能否分配到某个仓库取决于该仓库是否租赁
    for i in range(len(individual)):
        if individual[i] not in R_j_dict:
            penalty += penalty_factor  

    # 避免出现分母为零的情况，保证目标函数的可计算性
    if total_capacity_utilization == 0 or total_production_utilization == 0:
        return float('inf')

    # 计算适应度值，加入惩罚项以优先满足约束条件
    return (total_rental_cost * (1 / total_capacity_utilization) * (1 / total_production_utilization)) + penalty

# 初始化种群
def create_individual():
    # 创建个体表示对每个品类的仓库分配情况
    return [random.randint(0, len(warehouses) - 1) for _ in range(len(categories))]

def create_population():
    # 创建种群
    return [create_individual() for _ in range(population_size)]

# 选择操作（锦标赛）
def tournament_selection(pop):
    selected = random.sample(pop, tournament_size)
    selected.sort(key=lambda x: fitness(x))
    return selected[0]

# 交叉操作
def crossover(parent1, parent2):
    crossover_point = random.randint(1, len(parent1) - 2)
    child1 = parent1[:crossover_point] + parent2[crossover_point:]
    child2 = parent2[:crossover_point] + parent1[crossover_point:]
    return child1, child2

# 变异操作
def mutate(individual):
    if random.random() < mutation_rate:
        mutate_point = random.randint(0, len(individual) - 1)
        individual[mutate_point] = random.randint(0, len(warehouses) - 1)

# 主算法
population = create_population()  # 初始化种群
fitness_history = []  # 记录每代最佳适应度值

# 遗传算法迭代
for gen in range(generations):
    population = sorted(population, key=lambda x: fitness(x))
    next_population = population[:10]  # 精英选择保留前10个解
    
    # 生成下一代
    while len(next_population) < population_size:
        parent1 = tournament_selection(population)
        parent2 = tournament_selection(population)
        child1, child2 = crossover(parent1, parent2)
        mutate(child1)
        mutate(child2)
        next_population.extend([child1, child2])

    population = next_population  # 更新种群
    best_fitness = fitness(population[0])  # 记录当前代的最优适应度值
    fitness_history.append(best_fitness)
    print(f"Generation {gen}: Best Fitness = {best_fitness}")

# 绘制收敛曲线
plt.figure(figsize=(10, 6))
plt.plot(fitness_history, marker='o')
plt.xlabel('迭代此时')
plt.ylabel('目标函数')
plt.title('遗传算法的收敛图')
plt.grid(True)
plt.show()

# 输出最优解
best_individual = population[0]
assignment = [{'品类': categories[i], 'warehouse': warehouses[best_individual[i]]} for i in range(len(categories))]
assignment_df = pd.DataFrame(assignment)

assignment_df.to_csv('一品一仓结果.csv',index=None)




























































































