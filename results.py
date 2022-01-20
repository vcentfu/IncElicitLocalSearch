from utils import *
from tree import *
from pls import *
from elicitation import *
from pl import *
import numpy as np
import matplotlib.pyplot as plt
import multiprocessing


def experiment(nb_obj, nb_crit, agreg = "LW"):
    
    """ int * int * str -> 
    
        nb_obj : le nombre d'objets sélectionnés.
        nb_crit : le nombre de critères.
        agreg : type d'agrégateur utilisé : "LW", "OWA", "C".
        
        Retourne les résultats pour 20 jeux de poids. """

    d = read_data("2KP200-TA-0.dat")
    d = cut_data(d, nb_obj, nb_crit)
    tps = [0, 0]
    anst = [0, 0]
    gap = [0, 0]   
    
    for i in range(20):
        ini = initial_bag(d)
        om = omega_sum(nb_crit)
        dm = decision_maker(om)
        tv = true_sol(d, dm)
        
        start = time.process_time()
        t, _ = pls(d, ini, neighborhood, fulfill)
        ob = [i[1:] for i in t.get_all_i()]
        best_sol, nbans, _, _ = elicitation(ob, dm)
        last = time.process_time() - start
        
        tps[0] = tps[0] + last
        anst[0] = anst[0] + nbans
        gap[0] = gap[0] + np.abs(np.sum(np.array(om) * best_sol[-1]) - np.sum(np.array(om) * np.array(tv))) / np.sum(np.array(om) * np.array(tv))
        
        start = time.process_time()
        t, nbt = pls(d, ini, neighborhood, fulfill, elit = True, deci_m = dm)
        last = time.process_time() - start
        
        tps[1] = tps[1] + last
        anst[1] = anst[1] + nbt
        gap[1] = gap[1] + np.abs(np.sum(np.array(om) * t.get_all_i()[-1][1:]) - np.sum(np.array(om) * np.array(tv))) / np.sum(np.array(om) * np.array(tv))
    
    sob = str(nb_obj)
    
    if len(sob) < 3:
        while len(sob) < 3:
            sob = "0" + sob
            
    tps, anst, gap = (np.array(tps) / 20, np.array(anst) / 20, np.array(gap) / 20)
    file_name = "./logs/" + agreg + "_" + sob + "_N" + str(nb_crit) + ".log"
    f = open(file_name, "a")
    f.write("c %.2f %.1f %.3f\n" % (tps[0], anst[0], gap[0]))
    f.write("e %.2f %.1f %.3f\n" % (tps[1], anst[1], gap[1]))
    f.close()
    
    return tps, anst, gap 


if __name__ == '__main__':
    print(experiment(50, 3))
        