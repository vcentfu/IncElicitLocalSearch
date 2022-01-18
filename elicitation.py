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


class decision_maker:
    def __init__(self, omega):
        
        """ omega : vecteur de poids.
            pref : liste des paires comparées [préférée, moins préférée].
        
            Constructeur du décideur pour un problème de maximisation. """
        
        self.omega = omega
        self.pref = []
        
    def choose(self, x, y):
        
        """ list[int] * list[int] * list[float] -> bool
        
            x : une solution dans l'espace des critères.
            y : une solution dans l'espace des critères.
            
            Retourne True si le décideur préfère x à y, False sinon. """
            
        
        if np.sum(np.array(x) * np.array(self.omega)) > np.sum(np.array(y) * np.array(self.omega)):
            self.pref.append([x, y])
            
            return True
        
        self.pref.append([y, x])
        
        return False

        
def pmr(x, y, pref):
    
    """ list[int] * list[int] * list[list[int]] -> float
        
        Retourne le PMR de x, y selon les préférences pref. """
        
    m = gp.Model("PMR")
    m.setParam(GRB.Param.OutputFlag, 0)
    w = []
    
    for i in range(len(x)):
        w.append(m.addVar(vtype = GRB.CONTINUOUS, name = "j%d" % (i + 1)))
        m.addConstr(0 <= w[-1], "h0%d" % (i + 1))
        m.addConstr(1 >= w[-1], "h1%d" % (i + 1))
    
    w = np.array(w)
    m.setObjective(np.sum(np.array(y) * w) - np.sum(np.array(x) * w), GRB.MAXIMIZE)
    m.addConstr(np.sum(w) == 1, "c0")
    
    for i in range(len(pref)):
        m.addConstr(np.sum(np.array(pref[i][0]) * w) >= np.sum(np.array(pref[i][1]) * w), "c%d" % (i + 1))
        
    m.optimize()
    
    return m.ObjVal


def mr(x, X, pref):
    
    """ list[int] * list[list[int]] -> float
    
        Retourne le MR de x par rapport à l'ensemble des choix X et des préférences pref. """
    
    return max([pmr(x, y, pref) for y in X])


def mmr(X, pref):
    
    """ list[list[int]] * list[list[int]] -> float
    
        Retourne le MMR de l'ensemble X par rapport aux préférences pref. """
        
    return min([mr(x, X, pref) for x in X])


def om_dominance(y, X, pref):
    
    """ list[int] * list[list[int]] * list[list[int]] -> bool 
    
        Retourne True si y est omega-dominée par X pour une maximisation. """
        
    m = gp.Model("Omega-dominance")
    m.setParam(GRB.Param.OutputFlag, 0)
    w = []
    
    for i in range(len(y)):
        w.append(m.addVar(vtype = GRB.CONTINUOUS, name = "w%d" % (i + 1)))
        m.addConstr(0 <= w[-1], "h0%d" % (i + 1))
        m.addConstr(1 >= w[-1], "h1%d" % (i + 1))
    
    w = np.array(w)
    z = m.addVar(vtype = GRB.CONTINUOUS, name = "z")
    m.setObjective(z, GRB.MINIMIZE)
    m.addConstr(np.sum(w) == 1, "c0")
    
    for i in range(len(pref)):
        m.addConstr(np.sum(np.array(pref[i][0]) * w) >= np.sum(np.array(pref[i][1]) * w), "c%d" % (i + 1))
        
    for i in range(len(X)):
        m.addConstr(z >= np.sum(np.array(X[i]) * w) - np.sum(np.array(y) * w), "d%d" % (i + 1))
        
    m.optimize()
    
    return m.ObjVal > 0


def om_filter(X, pref):
    
    """ list[list[int]] -> list[list[int]]
    
        Retourne l'ensemble X filtré par l'omega-dominance selon les préférences pref. """
        
    res = X.copy()
    
    for i in range(len(X)):
        if om_dominance(X[i], res, pref):
            nres = []
            
            for x in res:
                if not np.all(X[i] == x):
                    nres.append(x)
            
            res = nres
            
    return res


def elicitation(X, dm):
    
    """ list[list[int]] * decision maker -> 
    
        Applique l'élicitation incrémentale. """
        
    Xc = X.copy()
    nbans = 0 
    mmrp = True
    tmmr = [mmr(Xc, dm.pref)]
    tans = [0]
    
    while len(Xc) > 1 and mmrp:
        a = r.randint(0, len(Xc) - 1)
        b = r.randint(0, len(Xc) - 1)
        
        while a == b:
            a = r.randint(0, len(Xc) - 1)
            b = r.randint(0, len(Xc) - 1)
            
        nbans += 1
        
        if dm.choose(Xc[a], Xc[b]):
            Xc = Xc[:b] + Xc[b + 1:]
        else:
            Xc = Xc[:a] + Xc[a + 1:]
         
        vmmr = mmr(Xc, dm.pref)
        
        if vmmr <= 0:
            mmrp = False
            
        Xc = om_filter(Xc, dm.pref)
        print("elicitation : size :", len(Xc), "answ :", nbans, "mmr :", vmmr)
        tmmr.append(vmmr)
        tans.append(nbans)
        
    return Xc, nbans, tmmr, tans
            
        



    
    
    
    
    
    
    
    
    
        
    
    
    
    
























