import random as r
import numpy as np
import gurobipy as gp
from gurobipy import GRB


def true_sol(d, dm, object_key = 'i', weight_bag = 'W'):
    
    """ dict[str:{int, list[int]}] * decision_maker -> list[float]
        
        d : le dictionnaire contenant les données extraits.
        dm : le décideur.
        
        Retourne la solution exacte préférée pour le décideur. """
        
    m = gp.Model("exact")
    m.setParam(GRB.Param.OutputFlag, 0)
    x = []
    
    for i in range(len(d[object_key])):
        x.append(m.addVar(vtype = GRB.BINARY, name = "x%d" % (i + 1)))
    
    o = []
    
    for i in range(len(d[object_key][-1]) - 1):
        o.append(sum([d[object_key][j][i + 1] * x[j] for j in range(len(x))]))
        
    m.setObjective(np.sum(np.array(dm.omega) * np.array(o)), GRB.MAXIMIZE)
    m.addConstr(sum([d[object_key][j][0] * x[j] for j in range(len(x))]) <= d[weight_bag])
    m.optimize()
    
    return [v.getValue() for v in o]






