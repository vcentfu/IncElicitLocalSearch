from utils import *
from tree import *
from pls import *
from elicitation import *
from pl import *
import numpy as np
import matplotlib.pyplot as plt
import multiprocessing


def experiment(nb_obj, nb_crit, strategy = "RANDOM", type_a = "LW"):
    
    """ int * int * str -> 
    
        nb_obj : le nombre d'objets sélectionnés.
        nb_crit : le nombre de critères.
        type_a : type d'agrégateur utilisé : "LW", "OWA", "CHOQ".
        
        Retourne les résultats pour 20 jeux de poids. """

    d = read_data("2KP200-TA-0.dat")
    d = cut_data(d, nb_obj, nb_crit)
    tps = [0, 0]
    anst = [0, 0]
    gap = [0, 0]   
    
    for i in range(20):
        ini = initial_bag(d)
        
        if type_a == "LW":
            om = omega_sum(nb_crit)
        elif type_a == "OWA":
            om = omega_owa(nb_crit)
            
        dm = decision_maker(om, type_a = type_a)
        tv = true_sol(d, dm)
        
        start = time.process_time()
        t, _ = pls(d, ini, neighborhood, fulfill, elit = False)
        ob = [i[1:] for i in t.get_all_i()]
        best_sol, nbans, _, _ = elicitation(ob, dm, strategy = strategy)
        last = time.process_time() - start
        
        tps[0] = tps[0] + last
        anst[0] = anst[0] + nbans
        
        if type_a == "LW":
            gap[0] = gap[0] + np.abs(np.sum(np.array(om) * best_sol[-1]) - np.sum(np.array(om) * np.array(tv))) / np.sum(np.array(om) * np.array(tv))
        elif type_a == "OWA":
            gap[0] = gap[0] + np.abs(np.sum(np.array(om) * np.sort(best_sol[-1])) - np.sum(np.array(om) * np.sort(np.array(tv)))) / np.sum(np.array(om) * np.sort(np.array(tv)))            
        
        start = time.process_time()
        t, nbt = pls(d, ini, neighborhood, fulfill, elit = True, deci_m = dm, strategy = strategy)
        last = time.process_time() - start
        
        tps[1] = tps[1] + last
        anst[1] = anst[1] + nbt
        
        if type_a == "LW":
            gap[1] = gap[1] + np.abs(np.sum(np.array(om) * t.get_all_i()[-1][1:]) - np.sum(np.array(om) * np.array(tv))) / np.sum(np.array(om) * np.array(tv))
        else:
            gap[1] = gap[1] + np.abs(np.sum(np.array(om) * np.sort(t.get_all_i()[-1][1:])) - np.sum(np.array(om) * np.sort(np.array(tv)))) / np.sum(np.array(om) * np.sort(np.array(tv)))
    
    sob = str(nb_obj)
    
    if len(sob) < 3:
        while len(sob) < 3:
            sob = "0" + sob
            
    tps, anst, gap = (np.array(tps) / 20, np.array(anst) / 20, np.array(gap) / 20)
    file_name = "./logs/" + type_a + "_" + type_a + sob + "_N" + str(nb_crit) + ".log"
    f = open(file_name, "a")
    f.write("c %5.2f %4.1f %3.3f\n" % (tps[0], anst[0], gap[0]))
    f.write("e %5.2f %4.1f %3.3f\n" % (tps[1], anst[1], gap[1]))
    f.close()
    
    return tps, anst, gap 


if __name__ == '__main__':
    proc = []
    args = [(50, 2), (50, 3), (50, 4), (50, 5), (50, 6), (100, 2), (100, 3), (100, 4), (100, 5), (100, 6)]
    for arg in args:
        nb_obj, nb_crit = arg
        print("launch multiprocessing results.py")
        p = multiprocessing.Process(target = experiment, args = (nb_obj, nb_crit, "RANDOM", "LW",))
        proc.append(p)
        p.start()
        p = multiprocessing.Process(target = experiment, args = (nb_obj, nb_crit, "RANDOM", "OWA"))
        proc.append(p)
        p.start()
        p = multiprocessing.Process(target = experiment, args = (nb_obj, nb_crit, "CSS", "LW",))
        proc.append(p)
        p.start()
        p = multiprocessing.Process(target = experiment, args = (nb_obj, nb_crit, "CSS", "OWA"))
        proc.append(p)
        p.start()
        
    for pro in proc:
        pro.join(60 * 30 * 20)
        