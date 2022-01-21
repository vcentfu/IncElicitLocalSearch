import random as r
import numpy as np
import gurobipy as gp
from gurobipy import GRB


def true_sol(d, dm, items_key = 'i', weight_bag = 'W'):
    
    """ dict[str:{int, list[int]}] * decision_maker -> list[float]
        
        d : le dictionnaire contenant les données extraits.
        dm : le décideur.
        
        Retourne la solution exacte préférée pour le décideur. """
        
    m = gp.Model("exact")
    m.setParam(GRB.Param.OutputFlag, 0)
    m.setParam(GRB.Param.LogToConsole, 0)
    m.setParam(GRB.Param.IntFeasTol, 1e-09)
    x = []
    
    for i in range(len(d[items_key])):
        x.append(m.addVar(vtype = GRB.BINARY, name = "x%d" % (i + 1)))
    
    o = []
    
    for i in range(len(dm.omega)):
        o.append(sum([d[items_key][j][i + 1] * x[j] for j in range(len(x))]))
        
    m.addConstr(sum([d[items_key][j][0] * x[j] for j in range(len(x))]) <= d[weight_bag], "bag_c")
    
    if dm.type_a == "LW":
        m.setObjective(np.sum(np.array(dm.omega) * np.array(o)), GRB.MAXIMIZE) 
    elif dm.type_a == "OWA":
        y = []
        
        for i in range(len(dm.omega)):
            y.append(m.addVar(vtype = GRB.INTEGER, name = "y%d" % (i + 1)))
            #m.addConstr(y[-1] > 0, "yc0_%d" % (i + 1))
            
        b = []
        
        for i in range(len(dm.omega)):
            bt = []
            
            for j in range(len(dm.omega)):
                bt.append(m.addVar(vtype = GRB.BINARY, name = "b_%d%d" % (i + 1, j + 1)))
                
            b.append(bt)
            
        b = np.array(b)
        m.setObjective(np.sum(np.array(dm.omega) * np.array(y)), GRB.MAXIMIZE) #
        M = []
        
        for l in range(len(dm.omega)):
            for u in range(l + 1, len(dm.omega)):
                M.append(sum([abs(d[items_key][k][l + 1] - d[items_key][k][u + 1]) for k in range(len(d[items_key]))]))
        
        if len(M) > 0:
            M = max(M)
        else:
            M = 10 ** 6
                
        for i in range(len(dm.omega)):
            for j in range(len(dm.omega)):
                m.addConstr(y[i] <= o[j] + M * b[i, j], "owa1_%d%d" % (i + 1, j + 1))
            
        for j in range(len(dm.omega)):
            m.addConstr(np.sum(b[:, j]) <= j, "owa2_%d" % (j + 1))
    
    m.optimize()
    
    """
    print("tf =", m.ObjVal)
    print(y)
    print(M)
    print(b)
    """
    
    return [v.getValue() for v in o]






