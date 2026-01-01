"""
Complementary Portfolio Selection using Gurobi MILP
Pure MILP version - Implements paper Equation (9)
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
    Solve complementary portfolio selection using Gurobi MILP.
    
    Paper Equation (9):
    min η
    s.t. Σ x_h = n
         Σ y_ih = 1, ∀i
         y_ih ≤ x_h, ∀i,h
         z_i = Σ f_i(h)*y_ih, ∀i
         z_i ≤ η, ∀i
         x_h, y_ih ∈ {0,1}
    
    Args:
        pool: Candidate heuristic pool 
              [{'name': ..., 'performance_vector': [...], 'avg_performance': ...}, ...]
        n: Number of heuristics to select
        time_limit: Time limit (seconds)
        verbose: Whether to print detailed information
    
    Returns:
        selected: Selected n heuristics
    """
    if len(pool) <= n:
        return pool
    
    H = len(pool)
    m = len(pool[0]['performance_vector'])
    
    # Performance matrix f[i][h]
    f = np.zeros((m, H))
    for h_idx, h in enumerate(pool):
        for i in range(m):
            f[i][h_idx] = h['performance_vector'][i]
    
    if verbose:
        print(f"\n{'='*80}")
        print(f"🔧 Gurobi MILP Complementary Selection")
        print(f"{'='*80}")
        print(f"Candidates: {H}, Instances: {m}, Selecting: {n}")
    
    # Create model
    model = gp.Model("ComplementarySelection")
    model.setParam('OutputFlag', 1 if verbose else 0)
    model.setParam('TimeLimit', time_limit)
    model.setParam('MIPGap', 1e-4)
    
    # Variables
    x = model.addVars(H, vtype=GRB.BINARY, name="x")
    y = model.addVars(m, H, vtype=GRB.BINARY, name="y")
    z = model.addVars(m, vtype=GRB.CONTINUOUS, lb=0, name="z")
    eta = model.addVar(vtype=GRB.CONTINUOUS, lb=0, name="eta")
    
    # Objective: min η
    model.setObjective(eta, GRB.MINIMIZE)
    
    # Constraints
    model.addConstr(gp.quicksum(x[h] for h in range(H)) == n)
    
    for i in range(m):
        model.addConstr(gp.quicksum(y[i, h] for h in range(H)) == 1)
        
        for h in range(H):
            model.addConstr(y[i, h] <= x[h])
        
        model.addConstr(z[i] == gp.quicksum(f[i][h] * y[i, h] for h in range(H)))
        model.addConstr(z[i] <= eta)
    
    # Solve
    if verbose:
        print("Starting optimization...")
    
    model.optimize()
    
    # Extract solution
    if model.status in [GRB.OPTIMAL, GRB.TIME_LIMIT]:
        if verbose:
            if model.status == GRB.OPTIMAL:
                print(f"✅ Optimal solution! η* = {eta.X:.4f}, Time: {model.Runtime:.2f}s")
            else:
                print(f"⚠️  Time limit reached, Gap: {model.MIPGap*100:.2f}%")
        
        selected_indices = [h for h in range(H) if x[h].X > 0.5]
        selected = [pool[h] for h in selected_indices]
        
        if verbose:
            print(f"\nSelected heuristics:")
            for i, h_idx in enumerate(selected_indices, 1):
                print(f"  {i}. {pool[h_idx]['name']}")
            print(f"{'='*80}\n")
        
        return selected
    else:
        raise RuntimeError(f"MILP solving failed: status = {model.status}")