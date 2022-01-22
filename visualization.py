from utils import *
from tree import *
from pls import *
from elicitation import *
from solution import *
from mpl_toolkits.mplot3d import Axes3D


def vizualize(pts, max_pts = [], color = '#1f77b4', clear = True, marker = "."):
    
    """ list[list[int]] * list[int] ->
    
        pts : Le front de pareto.
        max_pts : La grille maximale pour l'affichage plot.
        
        Affiche graphiquement le front de Pareto. """
        
    if clear:
        plt.clf()
        
        if len(max_pts) == 2:
            plt.grid()
        elif len(max_pts) > 3:
            for i in range(1, len(max_pts) - 1):
                plt.axvline(x = i, color = 'grey', linewidth = 1)
    
    if len(max_pts) == 2:
        plt.xlim(0, max_pts[0])
        plt.ylim(0, max_pts[1])
        t = np.array(pts)
        plt.scatter(t[:, 0], t[:, 1], marker = marker, color = color)
        plt.draw()
        plt.show(block = False)
        plt.pause(0.001)
    elif len(max_pts) == 3:
        ax = plt.axes(projection='3d')
        ax.set_xlim(0, max_pts[0])
        ax.set_ylim(0, max_pts[1])
        ax.set_zlim(0, max_pts[2])
        t = np.array(pts)
        ax.scatter(t[:, 0], t[:, 1], t[:, 2], marker = marker, color = color)
        plt.draw()
        plt.show(block = False)
        plt.pause(0.001)
    else:
        plt.xlim(0, len(max_pts) - 1)
        plt.ylim(0, max(max_pts))
        
        for t in pts:
            plt.plot([i for i in range(len(max_pts))], t, marker = marker, color = color, linewidth = 0.5)
        
        plt.draw()
        plt.show(block = False)
        plt.pause(0.001)
        
        
        