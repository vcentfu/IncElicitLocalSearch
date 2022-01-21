import random as r
import numpy as np
import gurobipy as gp
from gurobipy import GRB


def omega_sum(nb_crit):
    
    """ int -> list[float]
    
        nb_crit : nombres de critères.
        
        Retourne un vecteur de poids aléatoires sommant à 1. """
    
    res = np.array([r.random() for i in range(nb_crit)])
    res = res / np.sum(res)
    
    while np.sum(res) != 1:
        res = np.array([r.random() for i in range(nb_crit)])
        res = res / np.sum(res)
    
    return res


def omega_owa(nb_crit):
    
    """ int -> list[float]
    
        nb_crit : nombres de critères.
        
        Retourne un vecteur de poids décroissant aléatoires sommant à 1. """
        
    res = np.array([r.random() for i in range(nb_crit)])
    res = np.sort(res)[::-1]
    res = res / np.sum(res)
    
    while np.sum(res) != 1:
        res = np.array([r.random() for i in range(nb_crit)])
        res = np.sort(res)[::-1]
        res = res / np.sum(res)
    
    return res
    

class decision_maker:
    def __init__(self, omega, type_a = "LW"):
        
        """ omega : vecteur de poids.
            pref : liste des paires comparées [préférée, moins préférée].
        
            Constructeur du décideur pour un problème de maximisation. """
        
        self.omega = omega
        self.pref = []
        self.type_a = type_a
        
    def choose(self, x, y, verbose = False):
        
        """ list[int] * list[int] * list[float] -> bool
        
            x : une solution dans l'espace des critères.
            y : une solution dans l'espace des critères.
            
            Retourne True si le décideur préfère x à y, False sinon. """
            
        h = [list(map(list, p)) for p in self.pref]
        
        if [list(x), list(y)] in h or [list(y), list(x)] in h:
            if verbose:
                print("Question already asked")
            return -1
        
        if self.type_a == "LW":
            if np.sum(np.array(x) * np.array(self.omega)) > np.sum(np.array(y) * np.array(self.omega)):
                self.pref.append([x, y])
                
                return True
            
            self.pref.append([y, x])
        elif self.type_a == "OWA":
            if np.sum(np.sort(np.array(x)) * np.array(self.omega)) > np.sum(np.sort(np.array(y)) * np.array(self.omega)):
                self.pref.append([x, y])
                
                return True
            
            self.pref.append([y, x])
        
        return False

        
def pmr(x, y, pref, type_a = "LW"):
    
    """ list[int] * list[int] * list[list[int]] -> float
        
        Retourne le PMR de x, y selon les préférences pref. """
        
    m = gp.Model("PMR")
    m.setParam(GRB.Param.OutputFlag, 0)
    m.setParam(GRB.Param.LogToConsole, 0)
    w = []
    
    for i in range(len(x)):
        w.append(m.addVar(vtype = GRB.CONTINUOUS, name = "w_%d" % (i + 1)))
        m.addConstr(0 <= w[-1], "weightI0_%d" % (i + 1))
        m.addConstr(1 >= w[-1], "weightI1_%d" % (i + 1))
    
    w = np.array(w)
    m.addConstr(np.sum(w) == 1, "weight_sum")
    
    if type_a == "LW":
        m.setObjective(np.sum(np.array(y) * w) - np.sum(np.array(x) * w), GRB.MAXIMIZE)
        
        for i in range(len(pref)):
            m.addConstr(np.sum(np.array(pref[i][0]) * w) >= np.sum(np.array(pref[i][1]) * w), "pref_%d" % (i + 1))
    elif type_a == "OWA":
        m.setObjective(np.sum(np.sort(np.array(y)) * w) - np.sum(np.sort(np.array(x)) * w), GRB.MAXIMIZE)
        
        for i in range(len(w) - 1):
            m.addConstr(w[i] >= w[i + 1], "weightOwa_%d" % (i + 1))
            
        for i in range(len(pref)):
            m.addConstr(np.sum(np.sort(np.array(pref[i][0])) * w) >= np.sum(np.sort(np.array(pref[i][1])) * w), "pref_%d" % (i + 1))
            
    m.optimize()
    
    return m.ObjVal


def mr(x, X, pref, type_a = "LW"):
    
    """ list[int] * list[list[int]] -> float * int
    
        Retourne le MR de x par rapport à l'ensemble des choix X et des préférences pref. """
    
    t = [pmr(x, y, pref, type_a = type_a) for y in X]
    
    return max(t), np.argmax(t)


def mmr(X, pref, type_a = "LW"):
    
    """ list[list[int]] * list[list[int]] -> float
    
        Retourne le MMR de l'ensemble X par rapport aux préférences pref. """
    
    t = []
    qu = []
    
    for x in X:
        tv, q = mr(x, X, pref, type_a = type_a)
        t.append(tv)
        qu.append(q)
    
    return min(t), qu


def om_dominance(y, X, pref, type_a = "LW"):
    
    """ list[int] * list[list[int]] * list[list[int]] -> bool 
    
        Retourne True si y est omega-dominée par X pour une maximisation. """
        
    m = gp.Model("Omega-dominance")
    m.setParam(GRB.Param.OutputFlag, 0)
    m.setParam(GRB.Param.LogToConsole, 0)
    w = []
    
    for i in range(len(y)):
        w.append(m.addVar(vtype = GRB.CONTINUOUS, name = "w_%d" % (i + 1)))
        m.addConstr(0 <= w[-1], "weightI0_%d" % (i + 1))
        m.addConstr(1 >= w[-1], "weightI1_%d" % (i + 1))
    
    w = np.array(w)
    z = m.addVar(vtype = GRB.CONTINUOUS, name = "z")
    m.setObjective(z, GRB.MINIMIZE)
    m.addConstr(np.sum(w) == 1, "weight_sum")
    
    if type_a == "LW": 
        for i in range(len(pref)):
            m.addConstr(np.sum(np.array(pref[i][0]) * w) >= np.sum(np.array(pref[i][1]) * w), "pref_%d" % (i + 1))
            
        for i in range(len(X)):
            m.addConstr(z >= np.sum(np.array(X[i]) * w) - np.sum(np.array(y) * w), "dom_%d" % (i + 1))
    elif type_a == "OWA":
        for i in range(len(w) - 1):
            m.addConstr(w[i] >= w[i + 1], "weightOwa_%d" % (i + 1))
            
        for i in range(len(pref)):
            m.addConstr(np.sum(np.sort(np.array(pref[i][0])) * w) >= np.sum(np.sort(np.array(pref[i][1])) * w), "weight_%d" % (i + 1))
            
        for i in range(len(X)):
            m.addConstr(z >= np.sum(np.sort(np.array(X[i])) * w) - np.sum(np.sort(np.array(y)) * w), "dom_%d" % (i + 1))
            
            
    m.optimize()
    #print(m.ObjVal)
    #print(w)
    #print((np.sum(np.array(X[i]) * w) - np.sum(np.array(y) * w)).getValue())
    
    return m.ObjVal > 0


def om_filter(X, pref, type_a = "LW"):
    
    """ list[list[int]] -> list[list[int]]
    
        Retourne l'ensemble X filtré par l'omega-dominance selon les préférences pref. """
        
    res = X.copy()
    
    for i in range(len(X)):
        if om_dominance(X[i], res, pref, type_a = type_a):
            nres = []
            
            for j in range(len(res)):
                if np.all(X[i] == res[j]):
                    nres = res[:j] + res[j + 1:]
                    break
            
            res = nres
            
    return res


def elicitation(X, dm, strategy = "RANDOM", verbose = False):
    
    """ list[list[int]] * decision maker -> 
    
        Applique l'élicitation incrémentale. """
        
    Xc = X.copy()
    nbans = 0 
    mmrp = True
    vmmr, qu = mmr(Xc, dm.pref, type_a = dm.type_a)
    tmmr = [vmmr]
    tans = [0]
    
    while len(Xc) > 1 and mmrp:
        a = r.randint(0, len(Xc) - 1)
        
        if strategy == "RANDOM":
            b = r.randint(0, len(Xc) - 1)
        
            while a == b:
                b = r.randint(0, len(Xc) - 1)
        elif strategy == "CSS":
            b = qu[a]
            
        nbans += 1
        ans = dm.choose(Xc[a], Xc[b], verbose = verbose) 
        
        if ans:
            Xc = Xc[:b] + Xc[b + 1:]
        elif not ans:
            Xc = Xc[:a] + Xc[a + 1:]
        else:
            nbans -= 1
         
        Xc = om_filter(Xc, dm.pref, type_a = dm.type_a)
        vmmr, qu = mmr(Xc, dm.pref, type_a = dm.type_a)
        
        if vmmr <= 0:
            mmrp = False
        
        if verbose:     
            print("elicitation : size :", len(Xc), "answ :", nbans, "mmr :", vmmr)
        else:
            print("+", end = "")
            
        tans.append(nbans)
        
    if len(Xc) > 1:
        #print(dm.pref)
        #print(Xc)
        nbans +=  1
        
        if dm.type_a == "LW":
            Xc = [Xc[np.argmax([np.sum(x * dm.omega) for x in Xc])]]
        elif dm.type_a == "OWA":
            Xc = [Xc[np.argmax([np.sum(np.sort(x) * dm.omega) for x in Xc])]]
    
    """
    if dm.type_a == "LW":
        print("f =", np.sum(dm.omega * Xc[-1]))
    elif dm.type_a == "OWA":
        print("f =", np.sum(dm.omega * np.sort(Xc[-1])))
    """
    
    return Xc, nbans, tmmr, tans


if __name__ == "__main__": 
    pref = [[[6497.0, 4444.0, 6118.0], [6552.0, 4506.0, 5659.0]], [[5931.0, 4215.0, 6659.0], [5954.0, 5310.0, 5763.0]], [[5931.0, 4215.0, 6659.0], [6497.0, 4444.0, 6118.0]], [[6499.0, 4605.0, 7459.0], [5720.0, 4753.0, 7479.0]], [[7136.0, 5468.0, 7384.0], [6087.0, 4147.0, 7633.0]], [[6629.0, 5314.0, 8097.0], [7569.0, 5984.0, 7115.0]], [[7329.0, 6088.0, 7854.0], [6629.0, 5314.0, 8097.0]], [[6629.0, 5314.0, 8097.0], [7571.0, 5758.0, 7250.0]]]
    Xc = [[7243.0, 6486.0, 7912.0], [6543.0, 5712.0, 8155.0]]
    print(om_filter(Xc, pref))
            
        



    
    
    
    
    
    
    
    
    
        
    
    
    
    
























