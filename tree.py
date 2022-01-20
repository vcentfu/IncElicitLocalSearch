import numpy as np


class QuadTree:
    def __init__(self, bag, i, binary = chr(ord('0') - 1), succ = [], flag = False):
        
        """ list[int] * list[int] * str * list[QuadTree] * bool
        
            bag : La liste des objets sélectionnés associée au noeud (la liste d'indice).
            i : La liste des caractéristiques de l'objet (poids du sac et les valeurs objectifs du sac).
            binary : La chaine de caractères binaires decrivant le succesorship (0 si l'objectif est <= à l'objectif du précédent, 1 sinon)
            succ : La liste des QuadTree fils.
            flag : Un booléen pour indiquer la suppression dans le cadre de l'algorithme quad-tree 2.
            
            Constructeur d'un quad-tree pour une maximisation. """
        
        self.bag = bag
        self.i = i
        self.binary = binary
        self.succ = succ
        self.flag = flag
        
    
    def get_bag(self):
        
        """ -> int
        
            Retourne l'identifiant de l'objet associé au noeud. """
            
        return self.bag
    
    
    def get_i(self):
        
        """ -> list[int]
        
            Retourne la liste des caractéristiques de l'object. """
            
        return self.i
    
    
    def get_succ(self):
        
        """ -> list[QuadTree]

            Retourne les fils du noeud. """
            
        return self.succ
    
    def get_succ_binaries(self):
        
        """ -> list[str]

            Retourne les binaries des noeuds fils. """
            
        return [tree.binary for tree in self.succ]
    
    
    def get_all_bags(self):
        
        """ -> list[int]
        
            Retourne tous les identifiants des objets de l'arbre. """
            
        res = [self.bag]
        
        for tree in self.succ:
            res += tree.get_all_bags()
            
        return res
    
    
    def get_all_i(self):
        
        """ -> list[list[int]]
        
            Retourne la liste des listes des caractéristiques des objets de l'arbre. """
            
        res = [self.i]
        
        for tree in self.succ:
            res += tree.get_all_i()
            
        return res
    
    
    def get_binary(self):
        
        """ -> str
        
            Retourne le binary du noeud. """
            
        return self.binary
    
    
    def computes_binary(self, i):
        
        """ list[int] -> str
        
            i : La liste des caractéristiques de l'objet (poids du sac et les valeurs des objectifs du sac).
        
            Retourne le binary de l'objet ayant comme caractéristiques i par rapport au noeud. """
            
        binary = ''
        
        for v, nv in zip(i[1:], self.i[1:]):
            if v < nv:
                binary += '0'
            elif v > nv:
                binary += '1'
            else:
                binary += '*'
                    
        if binary == ('*' * len(binary)):
            return ('0' * len(binary))
        elif '0' in binary and '1' in binary:
            binaryt = ''
            
            for k in binary:
                if k == '*':
                    binaryt += '1'
                else:
                    binaryt += k
                    
            return binaryt
        else:
            c = ''
            p = 0
            
            while binary[p] == '*':
                p += 1
                
            c = binary[p]
            
            binaryt = c * len(binary)
            
            return binaryt
    
    
    def parallel_binary(self, binary, c):
        
        """ str * str -> bool
        
            binary : La chaine de caractères binaires à comparer. (0 si l'objectif est <= à l'objectif du précédent, 1 sinon)
            c : La valeur binaire à comparer.

            Retourne True si le noeud a un binary parallèle à binary avec c. """
            
        inds = [ind for ind in range(len(binary)) if binary[ind] == c]
        res = True
        i = 0
        
        while res and i < len(inds):
            if self.binary[inds[i]] != c:
                res = False
            
            i += 1
            
        return res

    
    def discard(self, i):
        
        """ list[int] -> bool
        
            i : La liste des caractéristiques de l'objet (poids du sac et les valeurs des objectifs du sac).
            
            Retourne True si l'objet à tester n'est pas à ajouter. """
        
        j = 0
        res = False
        binary = self.computes_binary(i)
        
        if binary == ('0' * (len(i) - 1)):
            return True
        
        while j < len(self.succ) and not res:
            if self.succ[j].parallel_binary(binary, '1') and self.succ[j].computes_binary(i) == ('0' * (len(i) - 1)):
                res = True
            
            if not res:
                res = self.succ[j].discard(i)
                
            j += 1
        
        return res
    
    
    def check_flag(self, i):
        
        """ list[int] -> 
        
            i : La liste des caractéristiques de l'objet (poids du sac et les valeurs des objectifs du sac).
        
            Effectue les marquages pour les suppressions (cf. quad-tree 2). """
        
        if self.computes_binary(i) == ('1' * (len(i) - 1)):
            self.flag = True
            
        for tree in self.succ:
            tree.check_flag(i)
    
    
    def get_to_insert(self, to_insert, new_root = False, b_root = True):
        
        """ list[QuadTree] * bool * bool -> bool
        
            to_insert : La liste des QuadTree à insérer.
            new_boot : booléen indiquant s'il y une nouvelle racine pour le QuadTree.
            b_root : booléen indiquant si la fonction est appelée sur le noeud racine.
        
            Recupère les noeuds non flagués à insérer dans to_insert et retourne True s'il faut une nouvelle racine, False sinon. """
        
        if b_root:
            if self.flag:
                new_root = True
            
            b_root = False
        
        
        for tree in self.succ:
            if not tree.flag and self.flag:
                to_insert += [tree]
                
            tree.get_to_insert(to_insert, new_root, b_root)
                
        return new_root
        
        
    def remove_filter(self):
        
        """ ->
        
            Supprime toutes les noeuds flagués. """
        
        succ = []
        
        for tree in self.succ:
            if not tree.flag:
                succ += [tree]
            
                tree.remove_filter()
            
        self.succ = succ
                
    
    def insert_new(self, bag, i):
        
        """ list[int] * list[int] ->
        
            bag : La liste des objets sélectionnés associée au noeud (la liste d'indice).
            i : La liste des caractéristiques de l'objet (poids du sac et les valeurs des objectifs du sac).
        
            Ajoute le nouveau objet dans l'arbre. """
        
        j = 0
        binary = self.computes_binary(i)
        
        while j < len(self.succ):
            if binary == self.succ[j].binary:
                self.succ[j].insert_new(bag, i)
                return
                
            j += 1
        
        if j == len(self.succ):
            self.succ = self.succ + [QuadTree(bag, i, binary = binary)]
    
    
    def insert(self, qt):
        
        """ QuadTree ->
        
            qt : Un quad-tree à ajouter dans l'arbre.
        
            Ajoute le quad-tree dans l'arbre. """
        
        j = 0
        binary = self.computes_binary(qt.i)
        
        while j < len(self.succ):
            if binary == self.succ[j].binary:
                self.succ[j].insert(qt)
                return
                
            j += 1
        
        if j == len(self.succ):
            qt.binary = binary
            self.succ = self.succ + [qt]
            
    
    def add(self, bag, i):
        
        """ list[int] * list[int] 
        
            object_id : L'identifiant de l'objet associé au noeud (typiquement l'indice dans la liste des objets).
            i : La liste des caractéristiques de l'objet.
            
            Ajoute l'objet et effectue une mise à jour de l'arbre. """
        
        if self.discard(i):
            return False
        
        self.check_flag(i)
        to_insert = []
        new_root = False
        b_root = True
        new_root = self.get_to_insert(to_insert, new_root, b_root)
        self.remove_filter()
        
        for tree in to_insert:
            tree.remove_filter()
        
        if not new_root:
            self.insert_new(bag, i)
        else:
            self.bag = bag
            self.i = i
            self.binary = chr(ord('0') - 1)
            self.succ = []
            self.flag = False
     
        for qt in to_insert:
            self.insert(qt)
            
        return True


def test_dominance(li):  # Pour vérifier que le Quadtree est bien un front de Pareto local
    
    """ list[list[int]] -> bool
    
        li : La liste des valeurs objectifs des sacs.
    
        Teste si li est un front de Pareto. """
    
    def cb(i, i_b):
        
        """ list[int] * list[int] -> str
        
            i : valeurs objectifs d'un sac.
            i_b : valeurs objectifs du second sac à comparer.
        
            Retourne le binary de i par rapport à i2. """
            
        binary = ''
        
        for v, nv in zip(i[1:], i_b[1:]):
            if v <= nv:
                binary += '0'
            else:
                binary += '1'
                
        return binary
    
    for i in range(len(li)):
        for i_b in range(i + 1, len(li)):
            if cb(li[i], li[i_b]) == ('1' * (len(li[0]) - 1)) or cb(li[i], li[i_b]) == ('0' * (len(li[0]) - 1)):
                print(li[i], li[i_b])
                return True
            
    return False
    
            
if __name__ == '__main__':
    print("Test QuadTree for 1000 alternatives")
    t = QuadTree(0, np.random.randint(1, 101, 5))
    verif = True

    for i in range(1, 1001):
        t.add(i, np.random.randint(1, 101, 5))
        print("i =", i)
        
        if test_dominance(t.get_all_i()):
            print(len(t.get_all_i()))
            print("non vérifié")
            verif = False
            break
    
    if verif:
        print("QuadTree fonctionne sans erreur de calculs.")
        
    


    
    
    
    
    
    
    
    
    
    
    
    
    
    
        
