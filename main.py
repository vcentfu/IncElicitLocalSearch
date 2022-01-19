from utils import *
from tree import *
from pls import *
from elicitation import *
from pl import *
import numpy as np
import matplotlib.pyplot as plt
import multiprocessing  


def draw_graph_mmr(file_name, nb_obj, nb_crit, color = None, show = False):
    
    """ str * int * int ->
    
        nb_obj : le nombre d'objets sélectionnés.
        nb_crit : le nombre de critères.
        
        Dessine le graphe mmr en fonction du nombre de questions pour nb_obj objets et nb_crit critères. """
        
    d = read_data(file_name)
    d = cut_data(d, nb_obj, nb_crit)
    ini = initial_bag(d)
    t, nbt = pls(d, ini, neighborhood, fulfill)
    ob = [o[1:] for o in t.get_all_i()]
    dm = decision_maker(omega_sum(nb_crit))
    print("initial size of choices:", len(ob))
    best_sol, nbans, tmmr, tans = elicitation(ob, dm)
    mmr_max = np.max(tmmr)
    
    if color != None:
        plt.plot(tans, np.array(tmmr) / mmr_max, "o-", color = color, label = "max mmr = %.2f, nb obj = %d, nb crit = %d" % (mmr_max, nb_obj, nb_crit))
    else:
        plt.plot(tans, np.array(tmmr) / mmr_max, "o-", label = "max mmr = %.2f, nb obj = %d, nb crit = %d" % (mmr_max, nb_obj, nb_crit))

    if show:
        plt.legend()
        plt.grid()
        plt.xlabel("number of queries")
        plt.ylabel("relative minimax regret")
        plt.xlim(0, 50)
        plt.ylim(0, 1)
        #plt.savefig("output.png")
        #plt.show()
    
    return


if __name__ == "__main__": 
    plt.figure(figsize = (12, 10))
    proc = []
    args = [(50, 3), (100, 3), (150, 3), (50, 5), (100, 5)]
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf']
   
    for arg, color in zip(args, colors):
        nb_obj, nb_crit = arg
        p = multiprocessing.Process(target = draw_graph_mmr, args = ("2KP200-TA-0.dat", nb_obj, nb_crit, color))
        proc.append(p)
        p.start()
        
    for pro in proc:
        pro.join()
        
    """
    draw_graph_mmr("2KP200-TA-0.dat", 50, 3)
    draw_graph_mmr("2KP200-TA-0.dat", 100, 3)
    draw_graph_mmr("2KP200-TA-0.dat", 150, 3)
    draw_graph_mmr("2KP200-TA-0.dat", 50, 5)
    draw_graph_mmr("2KP200-TA-0.dat", 100, 5, show = True)
    """
    plt.savefig("output.png")
    plt.show()
    








