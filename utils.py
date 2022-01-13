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
        

def cut_data(d, n, p, object_key = 'i', weight_bag = 'W'):
    
    """ dict[str:{int, list[int]}] * int -> dict[str:{int, list[int]}]
    
        d : le dictionnaire contenant les données extraits.
        n : le nombre des premiers objets extraits.
        p : le nombre d'objectifs.
        
        
        Retourne le dictionnaire avec les n premiers objets et p objectifs. """
        
    res = dict()
    
    for key in d:
        res[key] = d[key]
        
    t = []
    
    for i in range(n):
        t.append(d[object_key][i][:p + 1])
    
    res[object_key] = t
    res[weight_bag] = sum([res[object_key][i][0] for i in range(len(res[object_key]))]) / 2
    
    return res


def initial_bag(d, object_key = 'i', w_index = 0, bag_c = 'W'):
    
    """ dict[str:{int, list[int]}] -> list[int] 
    
        d : le dictionnaire contenant les données extraits.
    
        Retourne les objets sélectionnées respectant la contrainte du sac à dos. """
        
    res = []
    inds = [i for i in range(len(d[object_key]))]
    w = 0
    
    while len(inds) != 0:
        pick = r.choice(inds)
        
        if w + d[object_key][pick][w_index] <= d[bag_c]:
            w += d[object_key][pick][w_index] 
            res.append(pick)
        
        inds.remove(pick)

    return res
        

def neighborhood(d, bag, object_key = 'i', w_index = 0, bag_c = 'W'):
    
    """ dict[str:{int, list[int]}] * list[int] -> list[list[int]]
    
        bag : Une liste d'indices des objets sélectionnés.
        
        Retourne le voisinage de bag par un échange 1-1. """
        
    res = []
    inds = [j for j in range(len(d[object_key])) if j not in bag]
    i_obj = []
    
    for i in range(len(bag)):
        for pick in inds:
            c_bag = bag.copy()
            c_bag.remove(bag[i])
            w = sum([d[object_key][k][w_index] for k in c_bag])
            
            if w + d[object_key][pick][w_index] <= d[bag_c]:
                c_bag.append(pick)
                res.append(c_bag)
                i_obj.append(bag[i])
                
    return res, i_obj
                
                
def fulfill(d, bags, i_obj, object_key = 'i', w_index = 0, bag_c = 'W'):
    
    """ dict[str:{int, list[int]}] * list[list[int]] * list[int] -> list[list[int]]
    
        bags : La liste des listes d'indices des objets sélectionnés.
        
        Rempli les sacs à dos si possible. """
        
    res = []
    
    for bag, i_o in zip(bags, i_obj):
        c_bag = bag.copy()
        inds = [j for j in range(len(d[object_key])) if j not in c_bag and j != i_o]
        w = sum([d[object_key][k][w_index] for k in c_bag])
        
        while len(inds) != 0:
            pick = r.choice(inds)
            
            if w + d[object_key][pick][w_index] <= d[bag_c]:
                w += d[object_key][pick][w_index]
                c_bag.append(pick)
            
            inds.remove(pick)
        
        res.append(c_bag)
    
    return res


def objective_values_w(d, bag, object_key = 'i', w_index = 0, bag_c = 'W'):
    
    """ Retourne le poids du sac à dos avec ses valeurs objectives. """
    
    res = np.zeros(len(d[object_key][0]))
    
    for ind in bag:
        res += np.array(d[object_key][ind])
        
    return res


# TME précédent pour évaluer la performance de PLS


def read_non_dominated_pts(path):
    """ Returns non dominated points from inst with nb objects. """
    
    f = open('./' + path, 'r')
    lines = f.readlines()
    res = []
    
    for line in lines:
        tline = line.split()
        res.append((int(tline[0]), int(tline[1])))
        
    return set(res)
            

def convert_set(obags):
    
    res = [(k[1], k[2]) for k in obags]
    
    return set(res)


def compute_P_YN(YN_t, YN):
    """ Computes P_YN. """
    
    return len(set(YN_t) & set(YN)) / len(set(YN)) 


def draw_pts(all_pts):
    """ Draws all_pts into a graph. """
    
    _x = []
    _y = []
    
    for x, y in all_pts:
        _x.append(x)
        _y.append(y)
        
    plt.xlim(25000, 100000)
    plt.ylim(25000, 100000)
    plt.scatter(_x, _y, marker ='+')
    plt.grid()
    plt.show()
    
    return
        
        
    
    



