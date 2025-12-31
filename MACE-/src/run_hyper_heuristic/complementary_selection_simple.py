"""
Complementary Portfolio Selection using Gurobi MILP
纯MILP版本 - 实现论文公式(9)
"""

import gurobipy as gp
from gurobipy import GRB
import numpy as np
from typing import List, Dict


def complementary_selection_milp(
    pool: List[Dict],
    n: int,
    time_limit: float = 600.0,
    verbose: bool = True
) -> List[Dict]:
    """
    使用Gurobi MILP求解互补种群选择
    
    论文公式(9):
    min η
    s.t. Σ x_h = n
         Σ y_ih = 1, ∀i
         y_ih ≤ x_h, ∀i,h
         z_i = Σ f_i(h)*y_ih, ∀i
         z_i ≤ η, ∀i
         x_h, y_ih ∈ {0,1}
    
    Args:
        pool: 候选启发式池 [{'name': ..., 'performance_vector': [...], 'avg_performance': ...}, ...]
        n: 选择数量
        time_limit: 时间限制(秒)
        verbose: 是否打印详细信息
    
    Returns:
        selected: 选中的n个启发式
    """
    if len(pool) <= n:
        return pool
    
    H = len(pool)  # 候选数量
    m = len(pool[0]['performance_vector'])  # 实例数量
    
    # 性能矩阵 f[i][h]
    f = np.zeros((m, H))
    for h_idx, h in enumerate(pool):
        for i in range(m):
            f[i][h_idx] = h['performance_vector'][i]
    
    if verbose:
        print(f"\n{'='*80}")
        print(f"🔧 Gurobi MILP 互补选择")
        print(f"{'='*80}")
        print(f"候选启发式: {H}, 实例: {m}, 选择: {n}")
    
    # 创建模型
    model = gp.Model("ComplementarySelection")
    model.setParam('OutputFlag', 1 if verbose else 0)
    model.setParam('TimeLimit', time_limit)
    model.setParam('MIPGap', 1e-4)
    
    # 变量
    x = model.addVars(H, vtype=GRB.BINARY, name="x")
    y = model.addVars(m, H, vtype=GRB.BINARY, name="y")
    z = model.addVars(m, vtype=GRB.CONTINUOUS, lb=0, name="z")
    eta = model.addVar(vtype=GRB.CONTINUOUS, lb=0, name="eta")
    
    # 目标: min η
    model.setObjective(eta, GRB.MINIMIZE)
    
    # 约束
    model.addConstr(gp.quicksum(x[h] for h in range(H)) == n)  # 选n个
    
    for i in range(m):
        model.addConstr(gp.quicksum(y[i, h] for h in range(H)) == 1)  # 每个实例分配一个
        
        for h in range(H):
            model.addConstr(y[i, h] <= x[h])  # 只能分配给选中的
        
        model.addConstr(z[i] == gp.quicksum(f[i][h] * y[i, h] for h in range(H)))  # 性能
        model.addConstr(z[i] <= eta)  # worst-case
    
    # 求解
    if verbose:
        print("开始求解...")
    
    model.optimize()
    
    # 提取解
    if model.status in [GRB.OPTIMAL, GRB.TIME_LIMIT]:
        if verbose:
            if model.status == GRB.OPTIMAL:
                print(f"✅ 最优解! η* = {eta.X:.4f}, 时间: {model.Runtime:.2f}秒")
            else:
                print(f"⚠️  时间限制, Gap: {model.MIPGap*100:.2f}%")
        
        selected_indices = [h for h in range(H) if x[h].X > 0.5]
        selected = [pool[h] for h in selected_indices]
        
        if verbose:
            print(f"\n选中的启发式:")
            for i, h_idx in enumerate(selected_indices, 1):
                print(f"  {i}. {pool[h_idx]['name']}")
            print(f"{'='*80}\n")
        
        return selected
    else:
        raise RuntimeError(f"MILP求解失败: status = {model.status}")