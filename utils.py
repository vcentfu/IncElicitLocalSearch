import random as r
import numpy as np
import matplotlib.pyplot as plt


def read_data(path, keys = ['n', 'i', 'W'], types = ['i', 'l', 'i']):
    
    """ str * list[str]  * list[str] -> dict[str:{int, list[int]}]
        
        path : Le nom du fichier à lire.
        keys : Les clés du dictionnaires.
        types : Les formats des donnees à extraire.
            'i' -> int
            'l' -> list
        
        Retourne un dictionnaire contenant les données du fichier. """
        
    f = open('./' + path, 'r')
    lines = f.readlines()
    d = dict()
    
    for line in lines:
        sline = line.split()
        key = sline[0]
        
        if key in keys:
            if types[keys.index(key)] == 'i':
                d[key] = int(sline[1])
            elif types[keys.index(key)] == 'l':
                if key not in d:
                    d[key] = [list(map(int, sline[1:]))]
                else:
                    d[key].append(list(map(int, sline[1:])))
    
    return d


def read_eff(path):
    
    """ str -> list[list[int]]
    
        path : le nom du fichier eff à lire.
        
        Retourne la liste des points non dominées. """
        
    f = open("./" + path)
    lines = f.readlines()
    res = []
    
    for line in lines:
        tline = line.split()
        res.append(list(map(int, tline)))
        
    return res


def p_quality(yn_t, yn):
    
    """ list[list[int]] * list[list[int]] -> float
        
        yn_t : une approximation des points non-dominés. 
        yn : l'ensemble des points non-dominés.
        
        Retourne la proportion des points approximés intersectant les points non dominés. """

    tyn_t = list(map(tuple, yn_t))
    tyn = list(map(tuple, yn))
    
    return len(set(tyn_t) & set(tyn)) / len(set(tyn))
        

def cut_data(d, n, p, items_key = 'i', weight_bag = 'W', size_key = 'n'):
    
    """ dict[str:{int, list[int]}] * int -> dict[str:{int, list[int]}]
    
        d : le dictionnaire contenant les données extraits.
        n : le nombre des premiers objets extraits.
        p : le nombre d'objectifs.
        
        
        Retourne le dictionnaire avec les n premiers objets et p objectifs. """
        
    res = dict()
    res[size_key] = n
    
    for key in d:
        res[key] = d[key]
    
    res[size_key] = n
    t = []
    
    for i in range(n):
        t.append(d[items_key][i][:p + 1])
    
    res[items_key] = t
   
    if len(d[items_key]) != n:
        res[weight_bag] = sum([res[items_key][i][0] for i in range(len(res[items_key]))]) / 2
    else:
        res[weight_bag] = d[weight_bag]
    
    return res


def initial_bag(d, items_key = 'i', w_index = 0, bag_c = 'W'):
    
    """ dict[str:{int, list[int]}] -> list[int] 
    
        d : le dictionnaire contenant les données extraits.
    
        Retourne les objets sélectionnées respectant la contrainte du sac à dos. """
        
    res = []
    inds = [i for i in range(len(d[items_key]))]
    w = 0
    
    while len(inds) != 0:
        pick = r.choice(inds)
        
        if w + d[items_key][pick][w_index] <= d[bag_c]:
            w += d[items_key][pick][w_index] 
            res.append(pick)
        
        inds.remove(pick)

    return res
        

def neighborhood(d, bag, items_key = 'i', w_index = 0, bag_c = 'W'):
    
    """ dict[str:{int, list[int]}] * list[int] -> list[list[int]]
    
        d : le dictionnaire contenant les données extraits.
        bag : Une liste d'indices des objets sélectionnés.
        
        Retourne le voisinage de bag par un échange 1-1. """
        
    res = []
    inds = [j for j in range(len(d[items_key])) if j not in bag]
    i_obj = []
    
    for i in range(len(bag)):
        for pick in inds:
            c_bag = bag.copy()
            c_bag.remove(bag[i])
            w = sum([d[items_key][k][w_index] for k in c_bag])
            
            if w + d[items_key][pick][w_index] <= d[bag_c]:
                c_bag.append(pick)
                res.append(c_bag)
                i_obj.append(bag[i])
                
    return res, i_obj
                
                
def fulfill(d, bags, i_obj, items_key = 'i', w_index = 0, bag_c = 'W'):
    
    """ dict[str:{int, list[int]}] * list[list[int]] * list[int] -> list[list[int]]
    
        d : le dictionnaire contenant les données extraits.
        bags : La liste des listes d'indices des objets sélectionnés.
        
        Rempli les sacs à dos si possible. """
        
    res = []
    
    for bag, i_o in zip(bags, i_obj):
        c_bag = bag.copy()
        inds = [j for j in range(len(d[items_key])) if j not in c_bag and j != i_o]
        w = sum([d[items_key][k][w_index] for k in c_bag])
        
        while len(inds) != 0:
            pick = r.choice(inds)
            
            if w + d[items_key][pick][w_index] <= d[bag_c]:
                w += d[items_key][pick][w_index]
                c_bag.append(pick)
            
            inds.remove(pick)
        
        res.append(c_bag)
    
    return res


def objective_values_w(d, bag, items_key = 'i', w_index = 0, bag_c = 'W'):
    
    """ dict[str:{int, list[int]}] * list[int] -> list[int] 
    
        d : le dictionnaire contenant les données extraits.
        bag : Une liste d'indices des objets sélectionnés.
    
        Retourne le poids du sac à dos avec ses valeurs objectives. """
    
    res = np.zeros(len(d[items_key][0]))
    
    for ind in bag:
        res += np.array(d[items_key][ind])
        
    return res