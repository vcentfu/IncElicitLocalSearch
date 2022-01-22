import random as r
import numpy as np
import gurobipy as gp
from gurobipy import GRB
import sys, os
from itertools import *
 

stdout = sys.stdout
sys.stdout = open(os.devnull, 'w')
env = gp.Env()
sys.stdout = stdout


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


def v_choquet(nb_crit):
    
    """ int -> list[float]
    
        nb_crit : nombres de critères.
        
        Retourne une fonction de choquet super-modulaire. """
    
    def g_choquet(nb_crit):
        v = dict()
        v[()] = 0
        val = 0
        
        for i in range(1, nb_crit + 1):
            for arg in list(combinations([j + 1 for j in range(nb_crit)], i)): 
                if len(arg) == nb_crit:
                    v[arg] = 1
                else:
                    v[arg] = r.uniform(max([v[k] for j in range(len(arg)) for k in list(combinations(arg, j))]), 1)
                    
        return v
    
    def test_convex(v):
        i = 0
        convex = True
        keys = list(v.keys())
        
        while i < len(keys) and convex:
            j = 0
            
            while j < len(keys) and convex:
                union = tuple(sorted(list(set(keys[i]) | set(keys[j]))))
                inter = tuple(sorted(list(set(keys[i]) & set(keys[j]))))
                
                if v[union] + v[inter] < v[keys[i]] + v[keys[j]]:
                    convex = False
                
                j = j + 1
            
            i = i + 1
        
        return convex
                
    res = g_choquet(nb_crit)
    
    while not test_convex(res):
        res = g_choquet(nb_crit)
        
    return res


def mobius_inv(t, v):
    
    """ tuple(int) * dict[tuple : float -> float
    
        t : le tuple à caluler.
        v : le dictionnaire représentant une fonction de choquet super-modulaire.
        
        Retourne l'inverse de mobius de t pour capacité v. """
    
    res = 0
    
    for i in range(len(t) + 1):
        for arg in list(combinations(t, i)):
           res = res + ((-1) ** len(set(t) - set(arg))) * v[arg]
           
    return res


def choq_integ(x, v):
    
    """ list[int] * dict[tuple: float] -> float
    
        x : La liste des valeurs objectifs.
        v : le dictionnaire représentant une fonction de choquet super-modulaire.
        
        Retourne l'intégral de Choquet de x avec v. """
        
    xt = np.sort(np.array([0] + list(x)))
    w = [xt[i] - xt[i - 1] for i in range(1, len(xt))]
    u = []
    
    for obj in xt[1:]:
        ui = []
        
        for i in range(len(x)):
            if x[i] >= obj:
                ui.append(i + 1)
        
        u.append(tuple(ui))
    
    return np.sum(np.array(w) * np.array([v[k] for k in u]))


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
        elif self.type_a == "CHOQ":
            if choq_integ(x, self.omega) > choq_integ(y, self.omega):
                self.pref.append([x, y])
                
                return True
            
            self.pref.append([y, x])
        
        return False

        
def pmr(x, y, pref, type_a = "LW"):
    
    """ list[int] * list[int] * list[list[int]] -> float
        
        Retourne le PMR de x, y selon les préférences pref. """
        
    m = gp.Model("PMR", env = env)
    m.setParam(GRB.Param.OutputFlag, 0)
    m.setParam(GRB.Param.LogToConsole, 0)
    
    if type_a != "CHOQ":
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
    elif type_a == "CHOQ":
        w = dict()
        
        for i in range(len(x) + 1):
            for key in list(combinations([j + 1 for j in range(len(x))], i)):
                w[key] = m.addVar(vtype = GRB.CONTINUOUS, name = "w_%s" % (str(key)))
                m.addConstr(0 <= w[key], "weightC0_%s" % (str(key)))
                m.addConstr(1 >= w[key], "weightC1_%s" % (str(key)))
        
                
                if len(key) == len(x):
                    m.addConstr(w[key] == 1, "wc_1")
                    
        
        m.setObjective(choq_integ(y, w) - choq_integ(x, w), GRB.MAXIMIZE)
        m.addConstr(w[()] == 0, "wc_0")
        
        for key in w:
            for i in range(len(key) - 1):
                for s_key in list(combinations(key, i)):
                    m.addConstr(w[s_key] <= w[key], "wc_%s_%s" % (str(s_key), str(key)))
        
        keys = list(w.keys())
    
        for i in range(len(w)):
            for j in range(i + 1, len(w)):
                union = tuple(sorted(list(set(keys[i]) | set(keys[j]))))
                inter = tuple(sorted(list(set(keys[i]) & set(keys[j]))))
                m.addConstr(w[union] + w[inter] >= w[keys[i]] + w[keys[j]], "sm_%s_%s" % (str(keys[i]), str(keys[j])))
        
                
        for i in range(len(pref)):
            m.addConstr(choq_integ(pref[i][0], w) >= choq_integ(pref[i][1], w), "pref_%d" % (i + 1))
        
        
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
        
    m = gp.Model("Omega-dominance", env = env)
    m.setParam(GRB.Param.OutputFlag, 0)
    m.setParam(GRB.Param.LogToConsole, 0)
    
    z = m.addVar(vtype = GRB.CONTINUOUS, name = "z")
    m.setObjective(z, GRB.MINIMIZE)
    
    if type_a != "CHOQ":
        w = []
        
        for i in range(len(y)):
            w.append(m.addVar(vtype = GRB.CONTINUOUS, name = "w_%d" % (i + 1)))
            m.addConstr(0 <= w[-1], "weightI0_%d" % (i + 1))
            m.addConstr(1 >= w[-1], "weightI1_%d" % (i + 1))
        
        w = np.array(w)
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
    elif type_a == "CHOQ":
        w = dict()
        
        for i in range(len(y) + 1):
            for key in list(combinations([j + 1 for j in range(len(y))], i)):
                w[key] = m.addVar(vtype = GRB.CONTINUOUS, name = "w_%s" % (str(key)))
                m.addConstr(0 <= w[key], "weightC0_%s" % (str(key)))
                m.addConstr(1 >= w[key], "weightC1_%s" % (str(key)))
                
                if len(key) == len(y):
                    m.addConstr(w[key] == 1, "wc_1")
                    
        
        m.addConstr(w[()] == 0, "wc_0")
        
        for key in w:
            for i in range(len(key) - 1):
                for s_key in list(combinations(key, i)):
                    m.addConstr(w[s_key] <= w[key], "wc_%s_%s" % (str(s_key), str(key)))
        
        keys = list(w.keys())
        
        
        for i in range(len(w)):
            for j in range(i + 1, len(w)):
                union = tuple(sorted(list(set(keys[i]) | set(keys[j]))))
                inter = tuple(sorted(list(set(keys[i]) & set(keys[j]))))
                m.addConstr(w[union] + w[inter] >= w[keys[i]] + w[keys[j]], "sm_%s_%s" % (str(keys[i]), str(keys[j])))
    
        
        for i in range(len(pref)):
            m.addConstr(choq_integ(pref[i][0], w) >= choq_integ(pref[i][1], w), "pref_%d" % (i + 1))
            
        for i in range(len(X)):
            m.addConstr(z >= choq_integ(X[i], w) - choq_integ(y, w), "dom_%d" % (i + 1))
        
    m.optimize()
    
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
        tmmr.append(vmmr)
        
    if len(Xc) > 1:
        nbans +=  1
        
        if dm.type_a == "LW":
            Xc = [Xc[np.argmax([np.sum(x * dm.omega) for x in Xc])]]
        elif dm.type_a == "OWA":
            Xc = [Xc[np.argmax([np.sum(np.sort(x) * dm.omega) for x in Xc])]]
        elif dm.type_a == "CHOQ":
            Xc = [Xc[np.argmax([choq_integ(x, dm.omega) for x in Xc])]]
    
    return Xc, nbans, tmmr, tans
        



    
    
    
    
    
    
    
    
    
        
    
    
    
    
























