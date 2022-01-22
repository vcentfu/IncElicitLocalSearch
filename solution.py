import random as r
import numpy as np
import gurobipy as gp
from gurobipy import GRB
import sys, os


stdout = sys.stdout
sys.stdout = open(os.devnull, 'w')
env = gp.Env()
sys.stdout = stdout


def true_sol(d, dm, items_key = 'i', weight_bag = 'W'):
    
    """ dict[str:{int, list[int]}] * decision_maker -> list[float]
        
        d : le dictionnaire contenant les données extraits.
        dm : le décideur.
        
        Retourne la solution exacte préférée pour le décideur. """
        
    m = gp.Model("exact", env = env)
    m.setParam(GRB.Param.OutputFlag, 0)
    m.setParam(GRB.Param.LogToConsole, 0)
    m.setParam(GRB.Param.IntFeasTol, 1e-09)
    
    x = []
    
    for i in range(len(d[items_key])):
        x.append(m.addVar(vtype = GRB.BINARY, name = "x%d" % (i + 1)))
    
    o = []
    
    for i in range(len(d[items_key][-1]) - 1):
        o.append(sum([d[items_key][j][i + 1] * x[j] for j in range(len(x))]))
        
    m.addConstr(sum([d[items_key][j][0] * x[j] for j in range(len(x))]) <= d[weight_bag], "bag_c")
    
    if dm.type_a == "LW":
        m.setObjective(np.sum(np.array(dm.omega) * np.array(o)), GRB.MAXIMIZE) 
    elif dm.type_a == "OWA":
        r = []
        
        for i in range(len(dm.omega)):
            r.append(m.addVar(vtype = GRB.CONTINUOUS, name = "r%d" % (i + 1)))
            
        b = []
        
        for i in range(len(dm.omega)):
            bt = []
            
            for j in range(len(dm.omega)):
                bt.append(m.addVar(vtype = GRB.CONTINUOUS, name = "b_%d%d" % (i + 1, j + 1)))
                m.addConstr(bt[-1] >= 0, "b_c_%d%d" % (i + 1, j + 1))
            
            b.append(bt)
            
        b = np.array(b)
        
        m.setObjective(np.sum([(dm.omega[i] - dm.omega[i + 1]) * ((i + 1) * r[i] - np.sum(b[:, i])) for i in range(len(dm.omega) - 1)]) + dm.omega[-1] * (len(dm.omega) * r[-1] - np.sum(b[:, -1])), GRB.MAXIMIZE)
        
        for i in range(len(dm.omega)):
            for j in range(len(dm.omega)):
                m.addConstr(r[i] <= o[j] + b[j, i])
    elif dm.type_a == "CHOQ":
        vd = dict()
        
        for key in dm.omega:
            vd[key] = m.addVar(vtype = GRB.CONTINUOUS, name = "d_%s" % (str(key)))
            m.addConstr(vd[key] >= 0, "dc_%s" % (str(key)))
        
        m.setObjective(sum([dm.omega[a] * vd[a] for a in vd]), GRB.MAXIMIZE)
        
        for i in range(len(o)):
            m.addConstr(sum([vd[a] for a in vd if i + 1 in set(a)]) <= o[i], "choq_%d" % (i + 1))
            
    m.optimize()
    
    """
    print("tf =", m.ObjVal)
    print(y)
    print(M)
    print(b)
    """
    
    return [v.getValue() for v in o]






