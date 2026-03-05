# -*- coding: utf-8 -*-
"""
Created on Wed Dec  3 10:13:45 2025

@author: 雷建虎
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pymoo.core.problem import ElementwiseProblem
from pymoo.algorithms.moo.nsga2 import NSGA2
from pymoo.operators.crossover.sbx import SBX
from pymoo.operators.mutation.pm import PM
from pymoo.operators.sampling.rnd import FloatRandomSampling
from pymoo.optimize import minimize
import time
import warnings
import os

warnings.filterwarnings('ignore')

# 设置中文字体支持
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False


def get_non_dominated_indices(F):
    """
    简单非支配筛选函数：
    F: (n_samples, n_obj)，这里是 [f1, f2, f3]，全部是“要最小化”的目标
       注意：f3 = -profit（已经是最小化形式）
    返回：第一前沿（non-dominated）解的索引数组
    """
    F = np.asarray(F)
    n = F.shape[0]
    is_dominated = np.zeros(n, dtype=bool)

    for i in range(n):
        if is_dominated[i]:
            continue
        for j in range(n):
            if i == j or is_dominated[j]:
                continue
            # 如果 j 支配 i：所有目标 <= 且至少一个 <（全部是要“越小越好”）
            if np.all(F[j] <= F[i]) and np.any(F[j] < F[i]):
                is_dominated[i] = True
                break

    return np.where(~is_dominated)[0]


class CropOptimizationProblem(ElementwiseProblem):
    def __init__(self, Z, J_total, J_ufy,
                 UTC, TFI, UFY, EIA, CL, WA, M, PA_min, UAY,
                 TOTAL_AREA_MIN, TCD_Cur, TC_Cur, TNP_Cur,
                 max_additional_crops):
        """
        max_additional_crops: 每个区域除前 J_ufy 种作物外，最多允许额外种植的“高效作物”种类数
        """
        self.Z = Z
        self.J_total = J_total
        self.J_ufy = J_ufy      # 前 J_ufy 种为必须种植的作物（默认 j=0: 小麦, j=1: 玉米）
        self.UTC = UTC
        self.TFI = TFI
        self.UFY = UFY
        self.EIA = EIA
        self.CL = CL
        self.WA = WA
        self.M = M
        self.PA_min = PA_min
        self.UAY = UAY
        self.TOTAL_AREA_MIN = TOTAL_AREA_MIN  # 灌区总面积下限（不低于）
        self.TCD_Cur = TCD_Cur
        self.TC_Cur = TC_Cur      # 仅用于对比，不设刚性约束
        self.TNP_Cur = TNP_Cur
        self.max_additional_crops = max_additional_crops

        # ========= 1) 为每个村预先选出 “高效额外作物” =========
        # extra_allowed[z, j] = True 表示在村 z 作物 j 允许作为“额外作物”种植
        self.extra_allowed = np.zeros((Z, J_total), dtype=bool)

        for z in range(Z):
            # 候选额外作物：j = J_ufy ... J_total-1（这里是 11 种）
            extra_indices = np.arange(self.J_ufy, self.J_total)
            extra_tfi = self.TFI[z, self.J_ufy:]  # 该村对 11 种作物的单位面积效益

            k = min(self.max_additional_crops, len(extra_indices))
            if k > 0:
                # 按 TFI 从大到小选出前 k 个
                top_rel = np.argsort(extra_tfi)[::-1][:k]
                top_js = extra_indices[top_rel]
                self.extra_allowed[z, top_js] = True
            # 没选中的额外作物在该村永远不允许种植（后面会把上下界都设为 0）

        # ========= 2) 设置变量边界 =========
        #   前 J_ufy 种作物：下界 = PA_min（必须种植）
        #   额外“高效作物”：下界 = 0，上界按土地/水约束给
        #   非高效作物：上下界 = 0（永远不种）
        xl = np.zeros(Z * J_total)
        xu = np.zeros(Z * J_total)

        for z in range(Z):
            for j in range(J_total):
                idx = z * J_total + j

                if j < J_ufy:
                    # 两个主粮：必须种
                    xl[idx] = PA_min[z, j]
                else:
                    if self.extra_allowed[z, j]:
                        # 选中的高效额外作物：可以种，面积下界为 0
                        xl[idx] = 0.0
                    else:
                        # 没选中的额外作物：禁止种植
                        xl[idx] = 0.0
                        xu[idx] = 0.0
                        continue  # 直接下一个作物

                # 上界：受耕地面积、可灌溉面积和水资源限制
                max_by_land = min(EIA[z], CL[z])
                if M[z, j] > 0:
                    max_by_water = WA[z] / M[z, j]
                    xu[idx] = min(max_by_land, max_by_water)
                else:
                    xu[idx] = max_by_land

                if xu[idx] < xl[idx] - 1e-6:
                    print(f"警告: 区域{z}作物{j} - 上界({xu[idx]:.2f}) < 下界({xl[idx]:.2f})")
                    xu[idx] = xl[idx]

        # ========= 3) 约束数 =========
        # 1) 土地面积上界: Z
        # 2) 轮作约束(仅前 J_ufy 种): Z*J_ufy
        # 3) 耕地面积约束: Z*J_total
        # 4) 水资源约束: Z
        # 5) 每村额外作物种类数约束: Z
        # 6) 粮食安全约束: 1
        # 7) 经济效益约束: 1
        # 8) 灌区总种植面积下限: 1
        n_constr = (3 * Z) + (Z * J_ufy) + (Z * J_total) + 3

        super().__init__(n_var=Z * J_total,
                         n_obj=3,
                         n_constr=n_constr,
                         xl=xl,
                         xu=xu)

    def _evaluate(self, x, out, *args, **kwargs):
        PA = x.reshape(self.Z, self.J_total)

        # ========= 目标函数 =========
        # 1) 最小化灌溉水量
        f1 = np.sum(PA * self.M)

        # 2) 最小化单位产量碳排放
        total_carbon = np.sum(PA * self.UTC)
        total_yield = np.sum(PA * self.UAY)
        if total_yield > 0:
            f2 = total_carbon / total_yield
        else:
            f2 = 1e6

        # 3) 最大化经济效益 → 最小化 -经济效益
        f3 = -np.sum(PA * self.TFI)

        out["F"] = [f1, f2, f3]

        # ========= 约束 =========
        g = []

        # 1. 土地面积上界约束: 每个区域总面积 ≤ EIA[z]
        for z in range(self.Z):
            total_area = np.sum(PA[z, :])
            g.append(total_area - self.EIA[z])

        # 2. 轮作约束: 仅前 J_ufy 种作物要求 PA_zj ≥ PA_min_zj
        for z in range(self.Z):
            for j in range(self.J_ufy):
                g.append(self.PA_min[z, j] - PA[z, j])

        # 3. 耕地面积约束: 每个作物在每个区域 PA_zj ≤ CL[z]
        for z in range(self.Z):
            for j in range(self.J_total):
                g.append(PA[z, j] - self.CL[z])

        # 4. 水资源约束: 每个区域用水量 ≤ WA[z]
        for z in range(self.Z):
            water_usage = np.sum(PA[z, :] * self.M[z, :])
            g.append(water_usage - self.WA[z])

        # 5. 每村额外作物种类数约束（冗余检查，理论上 always ≤ max_additional_crops）
        epsilon = 1e-6
        for z in range(self.Z):
            num_extra = np.sum(PA[z, self.J_ufy:] > epsilon)
            g.append(float(num_extra) - float(self.max_additional_crops))

        # 6. 粮食安全约束：前 J_ufy 种作物总产量 ≥ TCD_Cur
        total_food_production = 0.0
        for z in range(self.Z):
            for j in range(self.J_ufy):
                total_food_production += PA[z, j] * self.UFY[z, j]
        g.append(self.TCD_Cur - total_food_production)

        # 7. 经济效益约束：总经济效益 ≥ TNP_Cur
        total_economic_benefit = np.sum(PA * self.TFI)
        g.append(self.TNP_Cur - total_economic_benefit)

        # 8. 灌区总面积约束（下限）：∑∑ PA_zj ≥ TOTAL_AREA_MIN
        total_area_all = np.sum(PA)
        g.append(self.TOTAL_AREA_MIN - total_area_all)

        out["G"] = g

    def evaluate_constraints(self, x):
        """评估单个解的约束违反情况（与 _evaluate 中 g 的顺序保持一致）"""
        PA = x.reshape(self.Z, self.J_total)

        g = []

        # 1. 土地面积上界
        for z in range(self.Z):
            total_area = np.sum(PA[z, :])
            g.append(total_area - self.EIA[z])

        # 2. 轮作约束（仅前 J_ufy 种）
        for z in range(self.Z):
            for j in range(self.J_ufy):
                g.append(self.PA_min[z, j] - PA[z, j])

        # 3. 耕地面积约束
        for z in range(self.Z):
            for j in range(self.J_total):
                g.append(PA[z, j] - self.CL[z])

        # 4. 水资源约束
        for z in range(self.Z):
            water_usage = np.sum(PA[z, :] * self.M[z, :])
            g.append(water_usage - self.WA[z])

        # 5. 每村额外作物种类数约束（冗余检查）
        epsilon = 1e-6
        for z in range(self.Z):
            num_extra = np.sum(PA[z, self.J_ufy:] > epsilon)
            g.append(float(num_extra) - float(self.max_additional_crops))

        # 6. 粮食安全约束
        total_food_production = 0.0
        for z in range(self.Z):
            for j in range(self.J_ufy):
                total_food_production += PA[z, j] * self.UFY[z, j]
        g.append(self.TCD_Cur - total_food_production)

        # 7. 经济效益约束
        total_economic_benefit = np.sum(PA * self.TFI)
        g.append(self.TNP_Cur - total_economic_benefit)

        # 8. 灌区总面积下限约束
        total_area_all = np.sum(PA)
        g.append(self.TOTAL_AREA_MIN - total_area_all)

        return np.array(g)


class EfficientFeasibleSampling(FloatRandomSampling):
    def _do(self, problem, n_samples, **kwargs):
        X = np.zeros((n_samples, problem.n_var))
        for i in range(n_samples):
            X[i] = self.generate_efficient_feasible(problem)
        return X

    def generate_efficient_feasible(self, problem):
        """
        生成高效的可行初始解：
        - 前 J_ufy 种作物按 PA_min 分配；
        - 额外作物：只在 extra_allowed=True 的作物里分配面积，
          相当于“每村从 11 个里选出的高效作物”。
        """
        x = np.zeros(problem.n_var)

        for z in range(problem.Z):
            start_idx = z * problem.J_total

            # 1) 为前 J_ufy 种“必种作物”分配最小面积
            total_min_area = 0.0
            for j in range(problem.J_total):
                if j < problem.J_ufy:
                    x[start_idx + j] = problem.PA_min[z, j]
                    total_min_area += problem.PA_min[z, j]
                else:
                    x[start_idx + j] = 0.0

            # 如果最小面积超过总面积，按比例缩减
            if total_min_area > problem.EIA[z] and total_min_area > 0:
                scale = problem.EIA[z] / total_min_area
                for j in range(problem.J_ufy):
                    x[start_idx + j] *= scale
                total_min_area = problem.EIA[z]

            # 2) 剩余面积按经济效益优先级分配
            remaining_area = problem.EIA[z] - total_min_area
            if remaining_area > 0:
                priorities = problem.TFI[z, :].copy()
                if np.sum(priorities) > 0:
                    priorities = priorities / np.sum(priorities)
                else:
                    priorities = np.ones(problem.J_total) / problem.J_total

                # 粮食作物优先顺序（前 J_ufy 种）
                cereal_indices = list(range(problem.J_ufy))
                cereal_indices_sorted = sorted(cereal_indices,
                                               key=lambda j: priorities[j],
                                               reverse=True)

                # 额外作物候选：只在 extra_allowed=True 的作物中选
                extra_all = [j for j in range(problem.J_ufy, problem.J_total)
                             if problem.extra_allowed[z, j]]
                chosen_extra = extra_all.copy()

                # 额外作物在自身内部按收益从高到低排序
                extra_sorted = sorted(chosen_extra,
                                      key=lambda j: priorities[j],
                                      reverse=True)

                # 最终分配顺序：高收益粮食 → 高收益额外作物
                allocation_order = cereal_indices_sorted + extra_sorted

                for j in allocation_order:
                    if remaining_area <= 0:
                        break

                    allocation = remaining_area * priorities[j]

                    current_water = np.sum(
                        x[start_idx:start_idx + problem.J_total] * problem.M[z, :]
                    )
                    remaining_water = problem.WA[z] - current_water

                    if problem.M[z, j] > 0:
                        max_by_water = remaining_water / problem.M[z, j]
                        allocation = min(allocation, max_by_water)

                    allocation = min(allocation, problem.CL[z] - x[start_idx + j])

                    if allocation < 0:
                        allocation = 0.0

                    x[start_idx + j] += allocation
                    remaining_area -= allocation

        return x


def plot_pareto_front_3d(objectives_real, output_dir, title_suffix=""):
    fig = plt.figure(figsize=(16, 12))

    water_usage = objectives_real[:, 0]
    carbon_intensity = objectives_real[:, 1]
    profit = objectives_real[:, 2]

    # 3D 帕累托
    ax1 = fig.add_subplot(231, projection='3d')
    scatter = ax1.scatter(water_usage, carbon_intensity, profit,
                          c=profit, cmap='viridis', s=50, alpha=0.7)
    ax1.set_xlabel('灌溉水量 (立方米)', fontsize=12, labelpad=10)
    ax1.set_ylabel('单位产量碳排放 (kg CO₂/kg)', fontsize=12, labelpad=10)
    ax1.set_zlabel('经济效益 (元)', fontsize=12, labelpad=10)
    ax1.set_title(f'三维帕累托前沿{title_suffix}', fontsize=14, pad=20)
    cbar = plt.colorbar(scatter, ax=ax1, shrink=0.6)
    cbar.set_label('经济效益 (元)', rotation=270, labelpad=15, fontsize=12)

    # 用水 vs 碳强度
    ax2 = fig.add_subplot(232)
    scatter2 = ax2.scatter(water_usage, carbon_intensity, c=profit,
                           cmap='viridis', s=50, alpha=0.7)
    ax2.set_xlabel('灌溉水量 (立方米)', fontsize=12)
    ax2.set_ylabel('单位产量碳排放 (kg CO₂/kg)', fontsize=12)
    ax2.set_title('用水量 vs 碳强度', fontsize=14)
    ax2.grid(True, alpha=0.3)
    cbar2 = plt.colorbar(scatter2, ax=ax2)
    cbar2.set_label('经济效益 (元)', rotation=270, labelpad=15, fontsize=12)

    # 用水 vs 经济效益
    ax3 = fig.add_subplot(233)
    scatter3 = ax3.scatter(water_usage, profit, c=carbon_intensity,
                           cmap='plasma', s=50, alpha=0.7)
    ax3.set_xlabel('灌溉水量 (立方米)', fontsize=12)
    ax3.set_ylabel('经济效益 (元)', fontsize=12)
    ax3.set_title('用水量 vs 经济效益', fontsize=14)
    ax3.grid(True, alpha=0.3)
    cbar3 = plt.colorbar(scatter3, ax=ax3)
    cbar3.set_label('单位产量碳排放 (kg CO₂/kg)', rotation=270, labelpad=15, fontsize=12)

    # 碳强度 vs 经济效益
    ax4 = fig.add_subplot(234)
    scatter4 = ax4.scatter(carbon_intensity, profit, c=water_usage,
                           cmap='cool', s=50, alpha=0.7)
    ax4.set_xlabel('单位产量碳排放 (kg CO₂/kg)', fontsize=12)
    ax4.set_ylabel('经济效益 (元)', fontsize=12)
    ax4.set_title('碳强度 vs 经济效益', fontsize=14)
    ax4.grid(True, alpha=0.3)
    cbar4 = plt.colorbar(scatter4, ax=ax4)
    cbar4.set_label('灌溉水量 (立方米)', rotation=270, labelpad=15, fontsize=12)

    # 用水分布
    ax5 = fig.add_subplot(235)
    ax5.hist(water_usage, bins=20, alpha=0.7, color='skyblue', edgecolor='black')
    ax5.set_xlabel('灌溉水量 (立方米)', fontsize=12)
    ax5.set_ylabel('解的数量', fontsize=12)
    ax5.set_title('灌溉水量分布', fontsize=14)
    ax5.grid(True, alpha=0.3)

    # 收益分布
    ax6 = fig.add_subplot(236)
    ax6.hist(profit, bins=20, alpha=0.7, color='lightgreen', edgecolor='black')
    ax6.set_xlabel('经济效益 (元)', fontsize=12)
    ax6.set_ylabel('解的数量', fontsize=12)
    ax6.set_title('经济效益分布', fontsize=14)
    ax6.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(f'{output_dir}/pareto_front_3d_and_projections{title_suffix}.png',
                dpi=300, bbox_inches='tight')
    plt.show()
    plt.close()
    print("✓ 三维帕累托前沿图和二维投影图已保存")


def plot_additional_analysis(objectives_real, output_dir, title_suffix=""):
    fig = plt.figure(figsize=(16, 12))

    water_usage = objectives_real[:, 0]
    carbon_intensity = objectives_real[:, 1]
    profit = objectives_real[:, 2]

    # 相关性矩阵
    ax1 = fig.add_subplot(221)
    targets = np.column_stack([water_usage, carbon_intensity, profit])
    corr_matrix = np.corrcoef(targets.T)
    im = ax1.imshow(corr_matrix, cmap='coolwarm', vmin=-1, vmax=1)
    ax1.set_xticks([0, 1, 2])
    ax1.set_yticks([0, 1, 2])
    ax1.set_xticklabels(['用水量', '碳强度', '经济效益'], fontsize=12)
    ax1.set_yticklabels(['用水量', '碳强度', '经济效益'], fontsize=12)
    ax1.set_title('目标函数相关性矩阵', fontsize=14)
    for i in range(3):
        for j in range(3):
            ax1.text(j, i, f'{corr_matrix[i, j]:.2f}',
                     ha='center', va='center',
                     color='white' if abs(corr_matrix[i, j]) > 0.5 else 'black',
                     fontsize=14, fontweight='bold')
    cbar1 = plt.colorbar(im, ax=ax1)
    cbar1.set_label('相关系数', rotation=270, labelpad=15, fontsize=12)

    # 平行坐标图
    ax2 = fig.add_subplot(222)
    nw = (water_usage - np.min(water_usage)) / (np.max(water_usage) - np.min(water_usage))
    nc = (carbon_intensity - np.min(carbon_intensity)) / (np.max(carbon_intensity) - np.min(carbon_intensity))
    npf = (profit - np.min(profit)) / (np.max(profit) - np.min(profit))
    for i in range(len(objectives_real)):
        ax2.plot([0, 1, 2], [nw[i], nc[i], npf[i]],
                 alpha=0.3, linewidth=1, color='blue')
    ax2.set_xticks([0, 1, 2])
    ax2.set_xticklabels(['用水量', '碳强度', '经济效益'], fontsize=12)
    ax2.set_ylabel('归一化值', fontsize=12)
    ax2.set_title('平行坐标图', fontsize=14)
    ax2.grid(True, alpha=0.3)

    # 理想点距离分布
    ax3 = fig.add_subplot(223)
    ideal_point = np.min(objectives_real, axis=0)
    distances = np.linalg.norm(objectives_real - ideal_point, axis=1)
    ax3.hist(distances, bins=20, alpha=0.7, color='orange', edgecolor='black')
    ax3.set_xlabel('到理想点的距离', fontsize=12)
    ax3.set_ylabel('解的数量', fontsize=12)
    ax3.set_title('解的质量分布', fontsize=14)
    ax3.grid(True, alpha=0.3)

    # 箱线图
    ax4 = fig.add_subplot(224)
    box_data = [water_usage, carbon_intensity, profit]
    ax4.boxplot(box_data, labels=['用水量', '碳强度', '经济效益'])
    ax4.set_ylabel('目标函数值', fontsize=12)
    ax4.set_title('目标函数值分布箱线图', fontsize=14)
    ax4.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(f'{output_dir}/additional_analysis{title_suffix}.png',
                dpi=300, bbox_inches='tight')
    plt.show()
    plt.close()
    print("✓ 附加分析图表已保存")


def load_parameters(excel_path):
    try:
        print("正在加载参数...")
        UTC = pd.read_excel(excel_path, sheet_name='Sheet1', header=None).values
        TFI = pd.read_excel(excel_path, sheet_name='Sheet2', header=None).values
        M = pd.read_excel(excel_path, sheet_name='Sheet7', header=None).values

        EIA = pd.read_excel(excel_path, sheet_name='Sheet6', header=None).values.flatten()
        CL = pd.read_excel(excel_path, sheet_name='Sheet4', header=None).values.flatten()
        WA = pd.read_excel(excel_path, sheet_name='Sheet8', header=None).values.flatten()

        UFY = pd.read_excel(excel_path, sheet_name='Sheet5', header=None).values
        UAY = pd.read_excel(excel_path, sheet_name='Sheet13', header=None).values

        TCD_Cur_df = pd.read_excel(excel_path, sheet_name='Sheet12', header=None)
        TCD_Cur = TCD_Cur_df.values[0, 0] if TCD_Cur_df.size > 0 else 0
        print(f"粮食安全约束值 TCD_Cur = {TCD_Cur}")

        TC_Cur_df = pd.read_excel(excel_path, sheet_name='Sheet11', header=None)
        TC_Cur = TC_Cur_df.values[0, 0] if TC_Cur_df.size > 0 else 0
        print(f"现状总碳排放 TC_Cur = {TC_Cur}（仅用于对比）")

        TNP_Cur_df = pd.read_excel(excel_path, sheet_name='Sheet16', header=None)
        TNP_Cur = TNP_Cur_df.values[0, 0] if TNP_Cur_df.size > 0 else 0
        print(f"经济效益约束值 TNP_Cur = {TNP_Cur}")

        # 灌区总面积下限，从 Sheet17 第一格读取
        TOTAL_AREA_df = pd.read_excel(excel_path, sheet_name='Sheet17', header=None)
        TOTAL_AREA_MIN = TOTAL_AREA_df.values[0, 0] if TOTAL_AREA_df.size > 0 else 0
        print(f"灌区总种植面积下限 TOTAL_AREA_MIN = {TOTAL_AREA_MIN}")

        # 现状种植结构：Sheet_PA
        PA_cur = pd.read_excel(excel_path, sheet_name='Sheet_PA', header=None).values
        print("✓ 现状种植结构 PA_cur 已从 Sheet_PA 读取")

        Z = 272
        J_total = 13
        J_ufy = 2

        print("✓ 参数加载完成")
        return (UTC, TFI, PA_cur, M, EIA, CL, WA,
                UFY, UAY, TOTAL_AREA_MIN,
                Z, J_total, J_ufy, TCD_Cur, TC_Cur, TNP_Cur)

    except Exception as e:
        print(f"✗ 加载参数失败: {e}")
        return None


def extract_feasible_solutions(res, problem,
                               tol_single=1e-5,
                               tol_sum=1e-4):
    """
    从 res.X 中提取“近似可行”的帕累托解：
    - 对每个解 x，计算 G = problem.evaluate_constraints(x)
    - 单个约束违反度 max(G_pos) <= tol_single
    - 总违反度 sum(G_pos) <= tol_sum
    若一个都没有，则返回违反度最小的解（近似可行解）
    """
    if res.X is None or len(res.X) == 0:
        print("未找到任何解（res.X 为空）")
        return None, None

    print(f"从 res.X（帕累托/最小违反解） 中提取候选解，共 {len(res.X)} 个。")

    feasible_solutions = []
    feasible_objectives = []

    min_violation_sum = float('inf')
    min_violation_single = float('inf')
    best_solution = None
    best_objective = None

    for x, F in zip(res.X, res.F):
        G = problem.evaluate_constraints(x)
        G_pos = np.maximum(0, G)

        v_sum = G_pos.sum()
        v_single = G_pos.max() if len(G_pos) > 0 else 0.0

        # 更新“最小违反度”的解
        if v_sum < min_violation_sum:
            min_violation_sum = v_sum
            min_violation_single = v_single
            best_solution = x
            best_objective = F

        # 满足容差 → 认为是“可行解”
        if (v_single <= tol_single) and (v_sum <= tol_sum):
            feasible_solutions.append(x)
            feasible_objectives.append(F)

    if len(feasible_solutions) == 0:
        print("未找到满足容差的可行解。")
        print(f"→ 当前候选集中最小总违反度 = {min_violation_sum:.3e}, "
              f"最小单约束最大违反度 = {min_violation_single:.3e}")
        print("将返回违反度最小的一个解用于后续分析（近似可行解）。")
        if best_solution is not None:
            return np.array([best_solution]), np.array([best_objective])
        else:
            return None, None

    print(f"找到 {len(feasible_solutions)} 个满足容差的可行解。")
    return np.array(feasible_solutions), np.array(feasible_objectives)


def main():
    print("=" * 60)
    print("多目标作物优化模型 - 最小化单位产量碳排放版本（多次运行 + 聚合非支配前沿）")
    print("前两种作物在每个区域必须种，其它 11 种作物中，每村只允许若干高效作物可种")
    print("=" * 60)

    excel_path = r"E:\calculation\Matlab\MATLAB_Projects\test2_稳定性.xlsx"
    parameters = load_parameters(excel_path)
    if parameters is None:
        print("参数加载失败")
        return

    (UTC, TFI, PA_cur, M, EIA, CL, WA,
     UFY, UAY, TOTAL_AREA_MIN,
     Z, J_total, J_ufy,
     TCD_Cur, TC_Cur, TNP_Cur) = parameters

    # ====== 由你控制小麦 / 玉米释放比例 ======  （单个情景，多次运行）
    try:
        inp_w = input("请输入小麦释放比例 r_wheat (0-0.3，回车默认 0)：").strip()
        r_wheat = float(inp_w) if inp_w != "" else 0.0
    except Exception:
        r_wheat = 0.0

    try:
        inp_m = input("请输入玉米释放比例 r_maize (0-0.3，回车默认 0)：").strip()
        r_maize = float(inp_m) if inp_m != "" else 0.0
    except Exception:
        r_maize = 0.0

    # 简单约束在 [0, 0.5] 之间，避免乱输
    r_wheat = max(0.0, min(0.5, r_wheat))
    r_maize = max(0.0, min(0.5, r_maize))

    print(f"→ 本次情景：r_wheat = {r_wheat:.3f}, r_maize = {r_maize:.3f}")

    # ====== 由你控制“每个村最多额外种几种高效作物” ======
    try:
        user_input = input(
            f"请输入每个村最多额外种植的高效作物种类数（1-{J_total - J_ufy}，回车默认 {J_total - J_ufy}）："
        ).strip()
        if user_input == "":
            max_additional_crops = J_total - J_ufy
        else:
            max_additional_crops = int(user_input)
            if max_additional_crops < 1:
                max_additional_crops = 1
            if max_additional_crops > J_total - J_ufy:
                max_additional_crops = J_total - J_ufy
    except Exception:
        max_additional_crops = J_total - J_ufy

    print(f"→ 每个区域除前{J_ufy}种主粮外，最多允许额外种植 {max_additional_crops} 种“高效作物”")
    print()

    # ========= 根据 r_wheat、r_maize 自动生成 PA_min =========
    # 假定 j=0 为小麦, j=1 为玉米
    PA_min = np.zeros_like(PA_cur)
    PA_min[:, 0] = (1.0 - r_wheat) * PA_cur[:, 0]
    PA_min[:, 1] = (1.0 - r_maize) * PA_cur[:, 1]
    # 其他作物的最小面积设为 0（完全由模型决定）   PA_min[:, 2:] 已经是 0

    print("已根据 r_wheat, r_maize 生成各村的小麦/玉米最小种植面积。")
    print(f"小麦 PA_min 范围: {PA_min[:, 0].min():.2f} - {PA_min[:, 0].max():.2f}")
    print(f"玉米 PA_min 范围: {PA_min[:, 1].min():.2f} - {PA_min[:, 1].max():.2f}")

    print(f"问题规模: {Z}个区域 × {J_total}种作物 = {Z * J_total}个决策变量")
    print(f"粮食安全约束: 前{J_ufy}种作物的总产量 ≥ {TCD_Cur}")
    print(f"现状总碳排放: {TC_Cur}（本模型不设置总碳排放刚性约束，仅在目标中考虑碳强度）")
    print(f"经济效益约束: 所有作物的总经济效益 ≥ {TNP_Cur}")
    print(f"灌区总面积约束: 所有村所有作物总面积 ≥ {TOTAL_AREA_MIN}")

    # 参数形状检查
    try:
        assert UTC.shape == (Z, J_total)
        assert TFI.shape == (Z, J_total)
        assert PA_cur.shape == (Z, J_total)
        assert PA_min.shape == (Z, J_total)
        assert M.shape == (Z, J_total)
        assert EIA.shape == (Z,)
        assert CL.shape == (Z,)
        assert WA.shape == (Z,)
        assert UFY.shape == (Z, J_ufy)
        assert UAY.shape == (Z, J_total)
        print("✓ 所有参数形状验证通过")
    except AssertionError as e:
        print(f"✗ 参数验证失败: {e}")
        return

    # ========== 多次运行设置 ==========
    pop_size = 200
    n_gen = 300
    n_runs = 7   # ←←← 你要做几次重复，就改这里的 7

    print("\n==== 开始多次运行 NSGA-II ====")
    print(f"配置: 种群大小={pop_size}, 最大代数={n_gen}, 重复运行次数={n_runs}")
    print("=" * 60)

    X_all = []
    F_all = []
    total_start_time = time.time()

    try:
        for run_idx in range(n_runs):
            print(f"\n>>> 运行 {run_idx + 1}/{n_runs} ...")
            start_time = time.time()

            problem = CropOptimizationProblem(
                Z, J_total, J_ufy,
                UTC, TFI, UFY, EIA, CL, WA, M, PA_min, UAY,
                TOTAL_AREA_MIN, TCD_Cur, TC_Cur, TNP_Cur,
                max_additional_crops
            )

            algorithm = NSGA2(
                pop_size=pop_size,
                sampling=EfficientFeasibleSampling(),
                crossover=SBX(prob=0.8, eta=15),
                mutation=PM(prob=0.3, eta=20),
                eliminate_duplicates=True
            )

            res = minimize(
                problem,
                algorithm,
                ('n_gen', n_gen),
                verbose=False,
                seed=None
            )

            run_time = time.time() - start_time
            print(f"运行 {run_idx + 1} 完成，用时 {run_time / 60:.2f} 分钟")

            pareto_solutions, pareto_objectives = extract_feasible_solutions(res, problem)

            if pareto_solutions is None or len(pareto_solutions) == 0:
                print(f"运行 {run_idx + 1}: 无可行/近似可行解，跳过。")
                continue

            X_all.append(pareto_solutions)
            F_all.append(pareto_objectives)
            print(f"运行 {run_idx + 1}: 收集到 {len(pareto_solutions)} 个可行或近似可行的帕累托解。")

        total_time = time.time() - total_start_time
        print("\n==============================================")
        print(f"✓ 全部 {n_runs} 次运行完成! 总耗时: {total_time / 60:.1f} 分钟")

        if len(X_all) == 0:
            print("✗ 所有运行中均未找到可行/近似可行解。")
            return

        # 堆叠所有运行的可行解
        X_all = np.vstack(X_all)
        F_all = np.vstack(F_all)
        n_feasible_total = X_all.shape[0]
        print(f"\n★ 累计可行/近似可行解数量（所有运行合计）: {n_feasible_total}")

        # 再做一次非支配筛选 → 得到 aggregated Pareto front
        front_idx = get_non_dominated_indices(F_all)
        pareto_solutions_agg = X_all[front_idx]
        pareto_objectives_agg = F_all[front_idx]

        print(f"★ 聚合后的非支配前沿解数量: {len(pareto_solutions_agg)}")
        if len(pareto_solutions_agg) == 0:
            print("✗ 聚合后没有非支配解，退出。")
            return

        # 将 F 转换为真实指标（经济效益取负号）
        objectives_real = []
        for F in pareto_objectives_agg:
            water_real = F[0]
            carbon_intensity_real = F[1]
            profit_real = -F[2]
            objectives_real.append([water_real, carbon_intensity_real, profit_real])
        objectives_real = np.array(objectives_real)

        print("\n聚合帕累托前沿分析:")
        print(f"灌溉水量范围: {np.min(objectives_real[:, 0]):.0f} - {np.max(objectives_real[:, 0]):.0f}")
        print(f"单位产量碳排放范围: {np.min(objectives_real[:, 1]):.4f} - {np.max(objectives_real[:, 1]):.4f}")
        print(f"经济效益范围: {np.min(objectives_real[:, 2]):.0f} - {np.max(objectives_real[:, 2]):.0f}")

        print("\n约束验证（抽查前 3 个聚合解）:")
        for i, sol in enumerate(pareto_solutions_agg[:min(3, len(pareto_solutions_agg))]):
            PA = sol.reshape(Z, J_total)
            total_food_production = 0.0
            for z in range(Z):
                for j in range(J_ufy):
                    total_food_production += PA[z, j] * UFY[z, j]
            total_carbon_emission = np.sum(PA * UTC)
            total_economic_benefit = np.sum(PA * TFI)
            total_area_all = np.sum(PA)

            epsilon = 1e-6
            max_over_all = 0
            for z in range(Z):
                num_extra = np.sum(PA[z, J_ufy:] > epsilon)
                if num_extra > max_over_all:
                    max_over_all = num_extra

            food_status = "✓ 满足" if total_food_production >= TCD_Cur else "✗ 违反"
            economic_status = "✓ 满足" if total_economic_benefit >= TNP_Cur else "✗ 违反"
            extra_status = "✓" if max_over_all <= max_additional_crops else "✗"
            area_status = "✓ 满足" if total_area_all + 1e-6 >= TOTAL_AREA_MIN else "✗ 违反"

            print(f"解{i}: 粮食产量 = {total_food_production:.2f} {food_status}, "
                  f"经济效益 = {total_economic_benefit:.2f} {economic_status}, "
                  f"碳排放 = {total_carbon_emission:.2f} (现状 {TC_Cur:.2f}), "
                  f"最大额外作物种类数 = {max_over_all} (上限 {max_additional_crops}) {extra_status}, "
                  f"灌区总面积 = {total_area_all:.2f} (下限 {TOTAL_AREA_MIN}) {area_status}")

        idx_min_water = np.argmin(objectives_real[:, 0])
        idx_min_carbon_intensity = np.argmin(objectives_real[:, 1])
        idx_max_profit = np.argmax(objectives_real[:, 2])

        print("\n代表性聚合解:")
        print(f"最少用水解 (索引{idx_min_water}): 用水量={objectives_real[idx_min_water, 0]:.0f}, "
              f"碳强度={objectives_real[idx_min_water, 1]:.4f}, 效益={objectives_real[idx_min_water, 2]:.0f}")
        print(f"最低碳强度解 (索引{idx_min_carbon_intensity}): 用水量={objectives_real[idx_min_carbon_intensity, 0]:.0f}, "
              f"碳强度={objectives_real[idx_min_carbon_intensity, 1]:.4f}, 效益={objectives_real[idx_min_carbon_intensity, 2]:.0f}")
        print(f"最高效益解 (索引{idx_max_profit}): 用水量={objectives_real[idx_max_profit, 0]:.0f}, "
              f"碳强度={objectives_real[idx_max_profit, 1]:.4f}, 效益={objectives_real[idx_max_profit, 2]:.0f}")

        # 输出目录
        output_dir = r"E:\calculation\Matlab\Python结果\最小化单位产量碳排放优化结果_多次聚合"
        os.makedirs(output_dir, exist_ok=True)

        # 保存每个聚合非支配解的面积矩阵 + summary
        solutions_info = []
        for i, (sol, obj) in enumerate(zip(pareto_solutions_agg, objectives_real)):
            PA = sol.reshape(Z, J_total)
            total_carbon_emission = np.sum(PA * UTC)
            total_yield = np.sum(PA * UAY)
            total_economic_benefit = np.sum(PA * TFI)
            total_area_all = np.sum(PA)

            total_food_production = 0.0
            for z in range(Z):
                for j in range(J_ufy):
                    total_food_production += PA[z, j] * UFY[z, j]

            epsilon = 1e-6
            max_extra = 0
            for z in range(Z):
                num_extra = np.sum(PA[z, J_ufy:] > epsilon)
                if num_extra > max_extra:
                    max_extra = num_extra

            solutions_info.append({
                'solution_id': i,
                'water_usage': obj[0],
                'carbon_intensity': obj[1],
                'profit': obj[2],
                'total_carbon_emission': total_carbon_emission,
                'total_yield': total_yield,
                'total_economic_benefit': total_economic_benefit,
                'total_food_production': total_food_production,
                'total_area_all': total_area_all,
                'max_extra_crops_per_village': max_extra,
                'food_constraint_satisfied': total_food_production >= TCD_Cur,
                'economic_constraint_satisfied': total_economic_benefit >= TNP_Cur,
                'area_constraint_satisfied': total_area_all + 1e-6 >= TOTAL_AREA_MIN,
                'n_feasible_total_all_runs': n_feasible_total,
                'r_wheat': r_wheat,
                'r_maize': r_maize
            })

            df = pd.DataFrame(PA)
            df.to_excel(os.path.join(output_dir, f"solution_agg_{i:03d}.xlsx"),
                        index=False, header=False)

        info_df = pd.DataFrame(solutions_info)
        info_df.to_excel(os.path.join(output_dir, "solutions_summary_aggregated.xlsx"),
                         index=False)
        print(f"\n✓ 聚合后的 {len(pareto_solutions_agg)} 个非支配解及其方案已保存到: {output_dir}")

        print("正在绘制聚合帕累托前沿图...")
        title_suffix = f"(r_wheat={r_wheat:.2f}, r_maize={r_maize:.2f}, runs={n_runs})"
        plot_pareto_front_3d(objectives_real, output_dir, title_suffix=title_suffix)
        plot_additional_analysis(objectives_real, output_dir, title_suffix=title_suffix)

        print(f"\n代表性解索引（聚合）:", [idx_min_water, idx_min_carbon_intensity, idx_max_profit])
        print("如需进一步查看具体某个聚合解，可以自己打开对应的 solution_agg_xxx.xlsx 文件。")

    except KeyboardInterrupt:
        print("\n✗ 优化被用户中断")
    except Exception as e:
        print(f"✗ 优化过程中出错: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
