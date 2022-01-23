from tree import *
from utils import *
import time
from elicitation import *
from solution import *
from visualization import *
from mpl_toolkits.mplot3d import Axes3D


def pls(d, initial_bag, neigh_func, fulfill_func, elit = False, deci_m = None, strategy = "RANDOM", verbose = False, render = False, render_stop = True):
    
    """ dict[str:{int, list[int]}] * list[int] * function -> list[list[int]]
    
        d : le dictionnaire contenant les données extraits.
        inital_bag : un sac à dos initial.
        neigh_func : la fonction de voisinage.
        fulfill_func : la fonction de remplissage de sac à dos.
        elit : booléen indiquant si l'élicitation incrémentale est activée.
        deci_m : le décideur sous forme de classe.
    
        Retourne les solutions Pareto-optimale par PLS. """
    
    if render:
        plt.show(block = False)
        plt.pause(0.001)
    
    p = [initial_bag]
    ini_o = objective_values_w(d, initial_bag)
    res = QuadTree(initial_bag, ini_o)
    sol_c = ini_o[1:]
    best_sol = None
    nbt = 0
    max_pts = objective_values_w(d, [i for i in range(len(d["i"]))])[1:]
        
    while len(p) != 0 or elit:
        if verbose and not elit:
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
        
        
        nn = []
        po = []
        pn = []
        
        if len(tv_bags) != 0:
            ln = QuadTree(tv_bags[0], to_bags[0])
        
            for sbag, obag in zip(tv_bags[1:], to_bags[1:]):
                ln.add(sbag, obag)
                
            pn = ln.get_all_bags()
            po = ln.get_all_i()

        for sbag, obag in zip(pn, po):
            if res.add(sbag, obag):
                nn.append(sbag)
        
        p = nn
        
        if elit :
            X = [list(i[1:]) for i in res.get_all_i()]
            
            if render:
                vizualize(X, max_pts, color = '#ff7f0e', clear = False)
            
            bX = res.get_all_bags()
        
            if verbose:
                print()
            
            best_sol, nbans, _, _ = elicitation(X, deci_m, strategy = strategy, verbose = verbose)
            nbt += nbans
            indX = X.index(list(best_sol[-1]))
            p = [bX[indX]]
            res = QuadTree(p[-1], objective_values_w(d, p[-1]))
            
            if list(sol_c) == list(best_sol[-1]):
                break
            else:
                sol_c = list(best_sol[-1])
        
        if render:
            pts = [list(i[1:]) for i in res.get_all_i()]
            vizualize(pts, max_pts, color = '#1f77b4')            
        
    if render:
        if elit:
            vizualize(X, max_pts, color = '#ff7f0e', clear = True)
            vizualize(pts, max_pts, color = '#1f77b4', clear = False) 
        
        if not verbose:
            print()
        
        print("End of exploration")
        
        if render_stop:
            print("Please close plot window to continue ...")
            plt.show()
    
    return res, nbt


if __name__ == "__main__":
    print("Test PLS for 2KP100-TA-0.dat with 2 objectives & all items")
    d = read_data("2KP100-TA-0.dat")
    d = cut_data(d, 100, 2)
    yn = read_eff("2KP100-TA-0.eff")
    ini = initial_bag(d)
    t, _ = pls(d, ini, neighborhood, fulfill, verbose = True, render = True)
    yn_t = [i[1:] for i in t.get_all_i()]
    p = p_quality(yn_t, yn)
    print("proportion eff:", p)
    print("---------------------------")
    
    nb_items = 20
    nb_crit = 3
    type_a = "LW"
    strategy = "RANDOM"
    verbose = True
    render = True 
    
    print("Test PLS then %s elicitation for 2KP200-TA-0.dat with %d objectives & %d items" % (type_a, nb_crit, nb_items))   
    d = read_data("2KP200-TA-0.dat")
    d = cut_data(d, nb_items, nb_crit)
    ini = initial_bag(d)
    
    if type_a == "LW":
        om = omega_sum(nb_crit)
    elif type_a == "OWA":
        om = omega_owa(nb_crit)
    elif type_a == "CHOQ":
        om = v_choquet(nb_crit)
        
    dm = decision_maker(om, type_a = type_a)
    start = time.process_time()
    
    t, nbt = pls(d, ini, neighborhood, fulfill, strategy = strategy, verbose = verbose, render = render, render_stop = False)
    
    if not verbose:
        print()
    
    ob = [i[1:] for i in t.get_all_i()]
    
    print("initial size of choices:", len(ob))
    print("start elicitation")
    best_sol, nbans, tmmr, tans = elicitation(ob, dm, strategy = strategy, verbose = verbose)
    last = time.process_time() - start
    
    if render:
        vizualize(best_sol, objective_values_w(d, [i for i in range(len(d["i"]))])[1:], '#2ca02c', clear = False)
        plt.show()
    
    if not verbose:
        print()
    
    print("best solution :", best_sol, "total answers :", nbans)
    print("time :", last)
    print("---------------------------")
    
    
    print("Test %s incremental elicitation for 2KP200-TA-0.dat with %d objectives %d items" % (type_a, nb_crit, nb_items))
    start = time.process_time()
    t, nbt = pls(d, ini, neighborhood, fulfill, elit = True, deci_m = dm, strategy = strategy, verbose = verbose, render = render)
    
    print()
    
    last = time.process_time() - start
    print("best solution :", t.get_all_i()[-1][1:], "total answers :", nbt)
    print("time :", last)
    
    print("True pref value :", true_sol(d, dm))
    
            
            
            



























