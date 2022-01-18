from utils import *
from tree import *
from pls import *
from elicitation import *
from pl import *
import numpy as np
import matplotlib.pyplot as plt 


def draw_graph_mmr(file_name, nb_obj, nb_crit, show = False):
    
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
    
    plt.plot(tans, np.array(tmmr) / mmr_max, "o-", label = "max mmr = %.2f, nb obj = %d" % (mmr_max, nb_obj))
    
    if show:
        plt.legend()
        plt.grid()
        plt.xlabel("number of queries")
        plt.ylabel("relative minimax regret")
        plt.xlim(0, 50)
        plt.ylim(0, 1)
        plt.title("Linear weights with %d criteria" % nb_crit)
        plt.savefig("output.png")
        plt.show()
    
    return


if __name__ == "__main__": 
    plt.figure(figsize = (12, 10))
    draw_graph_mmr("2KP200-TA-0.dat", 50, 3)
    draw_graph_mmr("2KP200-TA-0.dat", 100, 3)
    draw_graph_mmr("2KP200-TA-0.dat", 150, 3, show = True)






