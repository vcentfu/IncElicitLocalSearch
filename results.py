from utils import *
from tree import *
from pls import *
from elicitation import *
from solution import *
import numpy as np
import matplotlib.pyplot as plt
import multiprocessing


def experiment(nb_items, nb_crit, strategy = "RANDOM", type_a = "LW", test_times = 20):
    
    """ int * int * str -> 
    
        nb_items : le nombre d'objets sélectionnés.
        nb_crit : le nombre de critères.
        type_a : type d'agrégateur utilisé : "LW", "OWA", "CHOQ".
        
        Retourne les résultats pour 20 jeux de poids. """
    
    d = read_data("2KP200-TA-0.dat")
    d = cut_data(d, nb_items, nb_crit)
    tps = [0, 0]
    anst = [0, 0]
    gap = [0, 0]   
    
    for i in range(test_times):
        ini = initial_bag(d)
        
        if type_a == "LW":
            om = omega_sum(nb_crit)
        elif type_a == "OWA":
            om = omega_owa(nb_crit)
        elif type_a == "CHOQ":
            om = v_choquet(nb_crit)
            
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
        elif type_a == "CHOQ":
            gap[0] = gap[0] + np.abs(choq_integ(best_sol[-1], om) - choq_integ(tv, om)) / choq_integ(tv, om)
        
        start = time.process_time()
        t, nbt = pls(d, ini, neighborhood, fulfill, elit = True, deci_m = dm, strategy = strategy)
        last = time.process_time() - start
        
        tps[1] = tps[1] + last
        anst[1] = anst[1] + nbt
        
        if type_a == "LW":
            gap[1] = gap[1] + np.abs(np.sum(np.array(om) * t.get_all_i()[-1][1:]) - np.sum(np.array(om) * np.array(tv))) / np.sum(np.array(om) * np.array(tv))
        elif type_a == "OWA":
            gap[1] = gap[1] + np.abs(np.sum(np.array(om) * np.sort(t.get_all_i()[-1][1:])) - np.sum(np.array(om) * np.sort(np.array(tv)))) / np.sum(np.array(om) * np.sort(np.array(tv)))
        elif type_a == "CHOQ":
            gap[1] = gap[1] + np.abs(choq_integ(t.get_all_i()[-1][1:], om) - choq_integ(tv, om)) / choq_integ(tv, om)
        
    sob = str(nb_items)
    
    if len(sob) < 3:
        while len(sob) < 3:
            sob = "0" + sob
            
    tps, anst, gap = np.array(tps) / test_times, np.array(anst) / test_times, np.array(gap) / test_times
    file_name = "./logs/" + type_a + "_" + strategy + "_" + sob + "_N" + str(nb_crit) + ".log"
    f = open(file_name, "a")
    f.write("c %2.d %5.2f %4.1f %3.3f\n" % (test_times, tps[0], anst[0], gap[0]))
    f.write("e %2.d %5.2f %4.1f %3.3f\n" % (test_times, tps[1], anst[1], gap[1]))
    f.close()
    
    print()
    print("Process ended for %d items %d criteria %s strategy %s aggregator" % (nb_items, nb_crit, strategy, type_a))
    
    return tps, anst, gap 


if __name__ == '__main__':
    proc = []
    args = [(20, 2), (20, 3), (20, 4), (20, 5), (20, 6), (30, 2), (30, 3), (30, 4), (30, 5), (30, 6)]
    test_times = 20
    
    for arg in args:
        nb_items, nb_crit = arg
        print("launch multiprocessing results.py for %d items %d criteria %s strategy %s aggregator" % (nb_items, nb_crit, "RANDOM", "LW"))
        p = multiprocessing.Process(target = experiment, args = (nb_items, nb_crit, "RANDOM", "LW", test_times,))
        proc.append(p)
        p.start()
        print("launch multiprocessing results.py for %d items %d criteria %s strategy %s aggregator" % (nb_items, nb_crit, "RANDOM", "OWA"))
        p = multiprocessing.Process(target = experiment, args = (nb_items, nb_crit, "RANDOM", "OWA", test_times,))
        proc.append(p)
        p.start()
        print("launch multiprocessing results.py for %d items %d criteria %s strategy %s aggregator" % (nb_items, nb_crit, "RANDOM", "CHOQ"))
        p = multiprocessing.Process(target = experiment, args = (nb_items, nb_crit, "RANDOM", "CHOQ", test_times,))
        proc.append(p)
        p.start()
        print("launch multiprocessing results.py for %d items %d criteria %s strategy %s aggregator" % (nb_items, nb_crit, "CSS", "LW"))
        p = multiprocessing.Process(target = experiment, args = (nb_items, nb_crit, "CSS", "LW", test_times,))
        proc.append(p)
        p.start()
        print("launch multiprocessing results.py for %d items %d criteria %s strategy %s aggregator" % (nb_items, nb_crit, "CSS", "OWA"))
        p = multiprocessing.Process(target = experiment, args = (nb_items, nb_crit, "CSS", "OWA", test_times,))
        proc.append(p)
        p.start()
        print("launch multiprocessing results.py for %d items %d criteria %s strategy %s aggregator" % (nb_items, nb_crit, "CSS", "CHOQ"))
        p = multiprocessing.Process(target = experiment, args = (nb_items, nb_crit, "CSS", "CHOQ", test_times,))
        proc.append(p)
        p.start()
        
    for pro in proc:
        pro.join(60 * 30 * 20)
        