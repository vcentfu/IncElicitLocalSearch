from tree import *
from utils import *
import time
from elicitation import *
from pl import *


def pls(d, initial_bag, neigh_func, fulfill_func, elit = False, deci_m = None, strategy = "RANDOM", verbose = False):
    
    """ dict[str:{int, list[int]}] * list[int] * function -> list[list[int]]
    
        d : le dictionnaire contenant les données extraits.
        inital_bag : un sac à dos initial.
        neigh_func : la fonction de voisinage.
        fulfill_func : la fonction de remplissage de sac à dos.
        elit : booléen indiquant si l'élicitation incrémentale est activée.
        deci_m : le décideur sous forme de classe.
    
        Retourne les solutions Pareto-optimale par PLS. """
    
    p = [initial_bag]
    ini_o = objective_values_w(d, initial_bag)
    res = QuadTree(initial_bag, ini_o)
    sol_c = ini_o[1:]
    best_sol = None
    nbt = 0
        
    while len(p) != 0 or elit:
        if verbose:
            print("remaining exploration :", len(p))
        else:
            print(".", end = "")
            
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
        
        if elit :
            print()
                
            X = [list(o[1:]) for o in res.get_all_i()]
            bX = res.get_all_bags()
            best_sol, nbans, _, _ = elicitation(X, deci_m, strategy = strategy)
            nbt += nbans
            indX = X.index(list(best_sol[-1]))
            p = [bX[indX]]
            res = QuadTree(p[-1], objective_values_w(d, p[-1]))
            
            if list(sol_c) == list(best_sol[-1]):
                break
            else:
                sol_c = list(best_sol[-1])
    
    if not verbose and not elit:
        print()
        
    return res, nbt


if __name__ == "__main__":
    """
    print("Test PLS for 2KP100-TA-0.dat with 2 objectives & all objects")
    d = read_data("2KP100-TA-0.dat")
    d = cut_data(d, 100, 2)
    yn = read_eff("2KP100-TA-0.eff")
    ini = initial_bag(d)
    t, _ = pls(d, ini, neighborhood, fulfill, verbose = True)
    yn_t = [i[1:] for i in t.get_all_i()]
    p = p_quality(yn_t, yn)
    print("proportion :", p)
    print("---------------------------")
    """
    
    nb_obj = 10
    nb_crit = 6
    type_a = "OWA"
    strategy = "CSS"
    
    print("Test PLS then %s elicitation for 2KP200-TA-0.dat with %d objectives & %d objects" % (type_a, nb_crit, nb_obj))   
    d = read_data("2KP200-TA-0.dat")
    d = cut_data(d, nb_obj, nb_crit)
    ini = initial_bag(d)
    start = time.process_time()
    t, nbt = pls(d, ini, neighborhood, fulfill, strategy = strategy)
    ob = [i[1:] for i in t.get_all_i()]
    
    if type_a == "LW":
        om = omega_sum(nb_crit)
    elif type_a == "OWA":
        om = omega_owa(nb_crit)
        
    dm = decision_maker(om, type_a = type_a)
    print("initial size of choices:", len(ob))
    print("Start elicitation")
    best_sol, nbans, tmmr, tans = elicitation(ob, dm, strategy = strategy)
    print("best solution :", best_sol, "total answers :", nbans)
    #print("queries :", tans)
    #print("mmr list :", tmmr)
    last = time.process_time() - start
    print("time :", last)
    print("---------------------------")
    
    print("Test %s incremental elicitation for 2KP200-TA-0.dat with %d objectives %d objects" % (type_a, nb_crit, nb_obj))
    start = time.process_time()
    t, nbt = pls(d, ini, neighborhood, fulfill, elit = True, deci_m = dm, strategy = strategy)
    last = time.process_time() - start
    print("best solution :", t.get_all_i()[-1][1:], "total answers :", nbt)
    print("time :", last)
    
    print("True pref value :", true_sol(d, dm))
    
            
            
            



























