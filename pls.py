from tree import *
from utils import *
import time
from elicitation import *


def pls(d, initial_bag, neigh_func, fulfill_func):
    
    """ dict[str:{int, list[int]}] * list[int] * function -> list[list[int]]
    
        d : le dictionnaire contenant les données extraits.
        inital_bag : un sac à dos initial.
        neigh_func : la fonction de voisinage.
    
        Retourne les solutions Pareto-optimale par PLS. """
    
    p = [initial_bag]
    res = QuadTree(initial_bag, objective_values_w(d, initial_bag))
    
    while len(p) != 0:
        print("remaining exploration :", len(p))
        tv_bags = []
        to_bags = []
        
        for bag in p:
            vi_bags, i_obj = neigh_func(d, bag)
            v_bags = fulfill_func(d, vi_bags, i_obj)
            o_bags = [objective_values_w(d, sbag) for sbag in v_bags]
            tv_bags += v_bags
            to_bags += o_bags
        
        ln = QuadTree(tv_bags[0], to_bags[0])
        
        for sbag, obag in zip(tv_bags[1:], to_bags[1:]):
            ln.add(sbag, obag)
                
        pn = ln.get_all_bags()
        po = ln.get_all_i()
        
        nn = []
            
        for sbag, obag in zip(pn, po):
            if res.add(sbag, obag):
                nn.append(sbag)
        
        p = nn
        
    return res

if __name__ == "__main__":       
    d = read_data('2KP200-TA-0.dat')
    d = cut_data(d, 200, 2)
    ini = initial_bag(d)
    start = time.process_time()
    t = pls(d, ini, neighborhood, fulfill)
    last = time.process_time() - start
    print("time :", last)
    ob = [o[1:] for o in t.get_all_i()]
    dm = decision_maker(omega_sum(2))
    print("initial size of choices:", len(ob))
    print(elicitation(ob, dm))
    
    
    
    """
	sr = convert_set(t.get_all_i())
	YN = read_non_dominated_pts('2KP100-TA-0.eff')
	draw_pts(YN)
	print(compute_P_YN(sr, YN))


	while compute_P_YN(sr, YN) < 0.8:
	    d = read_data('2KP100-TA-0.dat')
	    ini = initial_bag(d)
	    t = pls(d, ini, neighborhood, fulfill)
	    sr = convert_set(t.get_all_i())
	    YN = read_non_dominated_pts('2KP100-TA-0.eff')
	    draw_pts(YN)
	    print(compute_P_YN(sr, YN))
	"""
                
            
            
            



























