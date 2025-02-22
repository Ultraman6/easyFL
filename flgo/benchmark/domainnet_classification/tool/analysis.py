import pandas as pd
import numpy as np
from scipy.stats import entropy
import itertools
import openpyxl
from tqdm import tqdm


# 读取Excel文件
def read_xlsx(file_path):
    # 使用pandas读取所有工作表
    xls = pd.ExcelFile(file_path)
    categories = {}
    for sheet_name in xls.sheet_names:
        df = pd.read_excel(xls, sheet_name=sheet_name)
        categories[sheet_name] = df
    return categories

# 计算不平衡度
def compute_imbalance(selected_combination, categories, weight_category=6 / 30, weight_domain=24 / 30):
    # 计算选择方案的类别不平衡度和领域不平衡度
    category_distributions = []
    domain_distributions = []

    for category_name, selected_class in tqdm(selected_combination.items()):
        # 获取该类别的小类选择的领域分布
        selected_row = categories[category_name].loc[categories[category_name]['Item'] == selected_class]

        # 计算类别不平衡度（使用方差）
        category_distribution = selected_row['Category Distribution'].values[0]
        category_distributions.append(category_distribution)

        # 计算领域不平衡度（使用领域分布的熵）
        domain_distribution = selected_row['Domain Distribution'].values[0]
        domain_distributions.append(domain_distribution)

    # 计算类别不平衡度的平均值（假设方差）
    category_imbalance = np.mean([np.var(dist) for dist in category_distributions])

    # 计算领域不平衡度的平均值（使用熵）
    domain_imbalance = np.var(domain_distributions)

    # 计算加权不平衡度
    total_imbalance = weight_domain * domain_imbalance + weight_category * category_imbalance
    return total_imbalance


# 生成所有可能的选择组合
def generate_combinations(categories):
    return list(itertools.product(*[list(category['Item']) for category in categories.values()]))


# 找出最优方案
def find_best_combination(categories):
    combinations = generate_combinations(categories)
    best_combination = None
    best_imbalance = float('inf')

    for comb in combinations:
        selected_combination = {category: comb[i] for i, category in enumerate(categories.keys())}
        imbalance = compute_imbalance(selected_combination, categories)

        if imbalance < best_imbalance:
            best_imbalance = imbalance
            best_combination = selected_combination

    return best_combination, best_imbalance


# 示例使用
file_path = '/mnt/d/GitHub/easyFL/flgo/benchmark/domainnet_classification/DomainNet_Dataset.xlsx'  # 请替换成你的文件路径
categories = read_xlsx(file_path)
# 计算最优组合
best_combination, best_imbalance = find_best_combination(categories)

# 输出最优方案和不平衡度
print("Best combination:", best_combination)
print("Best imbalance:", best_imbalance)
