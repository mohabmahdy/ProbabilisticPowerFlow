from probabilisticFunction import sum_lognormal, square_lognormal, ratio_lognormal, power_lognormal, diff_lognormal, scalar_lognormal
from lognormal import lognormal_RV
import numpy as np
def depth_first_search(branches, node, visited):
    visited.add(node)
    order = [node]
    for (i, j, _, _) in branches:
        if i == node and j not in visited:
            order.extend(depth_first_search(branches, j, visited))
    return order


def order_branch(branches, first_node=0):
    visited = set()
    ordered_nodes = depth_first_search(branches, first_node, visited)
    # Build branches in DFS order
    ordered_branches = []
    for u in ordered_nodes:
        for branch in branches:
            if branch[0] == u:
                ordered_branches.append(branch)

    return ordered_nodes, ordered_branches
def distflow_lognormal_iterative(branches:list, P:dict, Q:dict, slack_bus:int, max_iter:int = 100, slack_value:float = 1.0, order_branch_bool:bool=True, nodes:list = None, initial_sol:dict = None, tol:float=1e-6, calculate_losses:bool=False, view_iterations=True)->dict:
    # Correlation between P and Q is set to 1.0
    slack_value = slack_value**2
    solution_found = False
    if order_branch_bool or nodes is None:
        ordered_nodes, ordered_branches = order_branch(branches,slack_bus)
    else:
        ordered_nodes = nodes
        ordered_branches = branches

    I2 = {}
    if initial_sol is not None:
        V_i = initial_sol["V_mag"]
        P_ij = initial_sol["P"]
        Q_ij = initial_sol["Q"]
        I = initial_sol["I"]
        R_ij = initial_sol["R_ij"]
        X_ij = initial_sol["X_ij"]
        for i,j,_,_ in ordered_branches:
            I2[(i,j)] = square_lognormal(I[(i,j)])

    else:
        P_ij = {}; Q_ij = {}; I = {}; V_i = {}
        R_ij = {}; X_ij = {}

        for (i,j,z,_) in branches:
            P_ij[(i,j)] = None
            Q_ij[(i,j)] = None
            I2[(i,j)] = None
            R_ij[(i,j)] = z.real
            X_ij[(i,j)] = z.imag
        
        for i in ordered_nodes:
            V_i[i] = None
        
        for branch in reversed(ordered_branches):
            i,j,_,_= branch
            P_ij[(i,j)] = P[j]
            Q_ij[(i,j)] = Q[j]

            for (a,b,_,_) in ordered_branches:
                if a == j:
                    P_ij[(i,j)] = sum_lognormal(P_ij[(i,j)],P_ij[(a,b)])
                    Q_ij[(i,j)] = sum_lognormal(Q_ij[(i,j)],Q_ij[(a,b)])

    for iter in range(max_iter):
        if view_iterations:
            print(f"\r current iteration : {iter} of {max_iter}", end = '', flush=True)
        P_ij_old, Q_ij_old, V_i_old, I2_old = P_ij.copy(), Q_ij.copy(), V_i.copy(), I2.copy()
        for branch in reversed(ordered_branches):
            i,j,_,_= branch
            p_bus = None
            q_bus = None

            for (a,b,_,_) in ordered_branches:
                if a == j:
                    if p_bus is None:
                        p_bus = P_ij[(a,b)]
                        q_bus = Q_ij[(a,b)]
                    else:
                        p_bus = sum_lognormal(p_bus,P_ij[(a,b)])
                        q_bus = sum_lognormal(q_bus,Q_ij[(a,b)])

            p_load = P[j]
            q_load = Q[j]

            p2 = square_lognormal(P_ij[(i,j)])
            q2 = square_lognormal(Q_ij[(i,j)])
            nom = sum_lognormal(p2, q2,corr=1.0)
            term3 = ratio_lognormal(nom,V_i[i], add_one=True, corr=0.0, v_value=slack_value)

            P_ij[(i,j)] = sum_lognormal(p_load, p_bus)
            P_ij[(i,j)] = sum_lognormal(P_ij[(i,j)], term3, [1,R_ij[(i,j)]])

            Q_ij[(i,j)] = sum_lognormal(q_load, q_bus)
            Q_ij[(i,j)] = sum_lognormal(Q_ij[(i,j)], term3, [1,X_ij[(i,j)]])

        for branch in ordered_branches:
            i,j,_,_= branch
            p2 = square_lognormal(P_ij[(i,j)])
            q2 = square_lognormal(Q_ij[(i,j)])
            nom = sum_lognormal(p2, q2, corr=1.0)
            term4 = ratio_lognormal(nom,V_i[i], add_one=True, corr = 0.0, v_value=slack_value)

            I2[(i,j)] = term4 

            temp = sum_lognormal(P_ij[(i,j)], Q_ij[(i,j)],[2*R_ij[(i,j)], 2*X_ij[(i,j)]], corr = 1.0)
            if i == slack_bus:
                V_i[j] = temp 
            else:
                V_i[j] = sum_lognormal(V_i[i], temp, corr=1.0)

            V_i[j] = sum_lognormal(V_i[j], term4,[1,-(R_ij[(i,j)]**2+X_ij[(i,j)]**2)], corr=0.0)

        max_chg = 0.0
        for (i,j,_,_) in ordered_branches:
            max_chg = max(max_chg, diff_lognormal(P_ij[(i,j)], P_ij_old[(i,j)]),
                            diff_lognormal(Q_ij[(i,j)], Q_ij_old[(i,j)]),
                            diff_lognormal(I2[(i,j)], I2_old[(i,j)]),
                            diff_lognormal(V_i[j], V_i_old[j]))
            if max_chg >= tol:
                break

        if max_chg < tol:
            solution_found = True
            break
    
    S_loss = {}
    I = {}

    if solution_found:        
        for i in ordered_nodes[1:]:
            V_i[i] = lognormal_RV(V_i[i].get_mu(), V_i[i].get_sigma(), slack_value, negative=True)
        for (i,j,_,_) in ordered_branches:
            I[(i,j)] = power_lognormal(I2[(i,j)],0.5)
            if calculate_losses:
                p_loss2 = scalar_lognormal(I2[(i,j)], R_ij[(i,j)])
                q_loss2 = scalar_lognormal(I2[(i,j)], X_ij[(i,j)])
                S_loss[(i,j)] = power_lognormal(sum_lognormal(p_loss2, q_loss2, corr=1.0),0.5) 
        all_apparent_power = None
        for i,j,_,_ in ordered_branches:
            if i == slack_bus:
                s_n = power_lognormal(sum_lognormal(square_lognormal(P_ij[(i,j)]), square_lognormal(Q_ij[(i,j)]), corr=1.0),0.5)
                all_apparent_power = sum_lognormal(all_apparent_power, s_n)


    return {
        "V_mag": V_i,
        "P": P_ij,
        "Q": Q_ij,
        "S_loss": S_loss,
        "I": I,
        "R_ij": R_ij,
        "X_ij": X_ij,
        "S_slack": all_apparent_power,
        "iterations": iter+1,
        "converged": solution_found
    }