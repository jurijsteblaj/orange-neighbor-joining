from itertools import chain
from math import cos, sin, pi, atan2

import numpy as np
from numpy import ma
from scipy.spatial.distance import pdist

import matplotlib.pyplot as plt


def neighbor_joining(d):    
    D = ma.array(d, copy=True)
    D.mask = ma.make_mask_none(D.shape)
    joined_nodes = []
    
    result = {}
    
    n = len(D) - len(joined_nodes)
    while n > 2:
        # Based on the current distance matrix calculate the matrix Q
        row_sums = np.sum(D, axis=1)
        Q = ((n - 2)*D - row_sums).T - row_sums
        
        Q.mask[np.diag_indices(len(Q))] = True
        
        # Find the pair of distinct taxa i and j for which Q (i, j) has its lowest value
        ix = ma.argmin(Q)
        i = ix % len(Q)
        j = int(ix/len(Q))
        
        # Calculate the distance from each of the taxa in the pair to this new node
        dist_i = D[i,j]/2 + (row_sums[i] - row_sums[j])/(2*(n - 2))
        dist_j = D[i,j] - dist_i
        
        if i not in result:
            result[i] = []
        result[i].append([len(D), dist_i])
        if j not in result:
            result[j] = []
        result[j].append([len(D), dist_j])
        if len(D) not in result:
            result[len(D)] = []
        result[len(D)].append([i, dist_i])
        result[len(D)].append([j, dist_j])
        
        # Calculate the distance from each of the taxa outside of this pair to the new node
        dist = D[i,j]
        i_col = D[:,i]
        j_col = D[:,j]
        new = (i_col + j_col - dist)/2
        
        # Add the new node to the distance matrix
        D = ma.vstack((D, new))
        D = ma.column_stack((D, ma.append(new, (0))))
        
        # Remove the two nodes which were joined
        D.mask[i,:] = True
        D.mask[j,:] = True
        D.mask[:,i] = True
        D.mask[:,j] = True

        joined_nodes.append(i)
        joined_nodes.append(j)

        D.mask[-1,joined_nodes] = True
        D.mask[joined_nodes,-1] = True
        
        
        n = len(D) - len(joined_nodes)
    
    # Join the last two remaining nodes
    ix = ma.argmax(D)
    i = ix % len(Q)
    j = int(ix/len(Q))
    
    if i not in result:
        result[i] = []
    result[i].append([j, D[i,j]])
    if j not in result:
        result[j] = []
    result[j].append([i, D[i,j]])
    
    return result


def children(tree, v):
    """Return a list of indexes of child nodes of a parent node in a tree."""
    if len(tree[v]) == 0:
        return []
    else:
        return list(zip(*tree[v]))[0]

    
def remove_backlink(t, child, parent):
    try:
        ix = children(t, child).index(parent)
        del t[child][ix]
    except ValueError:
        pass
    for c in children(t, child):
        remove_backlink(t, c, child)

        
def rooted(tree, root=0):
    t = tree.copy()
    for child in children(t, root):
        remove_backlink(t, child, root)
    return t


def distance(tree, parent, child):
    """Return distance between a parent and a child node in a tree."""
    ix = children(tree, parent).index(child)
    return tree[parent][ix][1]


def postorder_traversal(tree, v, l):
    if len(tree[v]) == 0:
        l[v] = 1
    else:
        l[v] = 0
        for w in children(tree, v):
            postorder_traversal(tree, w, l)
            l[v] += l[w]

            
def preorder_traversal(tree, v, parent, root, x, l, omega, tau):
    if v != root:
        u = parent
        angle = tau[v] + omega[v]/2
        x[v] = x[u] + distance(tree, u, v) * np.array((cos(angle), sin(angle)))
    eta = tau[v]
    for w in children(tree, v):
        omega[w] = 2*pi * l[w]/l[root]
        tau[w] = eta
        eta += omega[w]
        preorder_traversal(tree, w, v, root, x, l, omega, tau)

        
def get_points(rooted_tree, root=0):
    """See Algorithm 1: RADIAL-LAYOUT in:
    Bachmaier, Christian, Ulrik Brandes, and Barbara Schlieper.
    "Drawing phylogenetic trees." Algorithms and Computation (2005): 1110-1121.
    """
    l = {}
    x = {}
    omega = {}
    tau = {}
    
    postorder_traversal(rooted_tree, root, l)
    
    x[root] = np.array((0, 0))
    omega[root] = 2*pi
    tau[root] = 0
    preorder_traversal(rooted_tree, root, None, root, x, l, omega, tau)
    
    return x


def plot(tree, points, labels=[], classes=None):
    for v1 in tree:
        for v2 in children(tree, v1):
            plt.plot((points[v1][0], points[v2][0]), (points[v1][1], points[v2][1]), 'k')
            if v2 < len(labels):
                delta = points[v2] - points[v1]
                angle = atan2(delta[1], delta[0])*180/pi
                angle = (angle + 360) % 360
                if angle > 90 and angle < 270:
                    alignment = "right"
                else:
                    alignment = "left"
                if angle < 90:
                    pass
                elif angle < 180:
                    pass
                elif angle < 270:
                    pass
                else:
                    pass
                if angle < 45:
                    va = "center"
                elif angle < 135:
                    va = "bottom"
                elif angle < 225:
                    va = "center"
                elif angle < 315:
                    va = "top"
                else:
                    va = "center"
                
                if angle > 90 and angle < 270:
                    rotation = (angle + 180) % 360
                else:
                    rotation = angle
                plt.text(*points[v2], labels[v2], rotation=rotation,
                         va=va, clip_on=True, ha=alignment)
    
    if classes != None:
        for c in classes:
            plt.plot([points[x][0] for x in c], [points[x][1] for x in c], ".", ms=3)
    
    #plt.savefig("output.svg")
    plt.show()



def deg_r(rooted_tree, v, root):
    """Get degree of a node in a rooted tree."""
    if v == root:
        return len(children(rooted_tree, v))
    else:
        return len(children(rooted_tree, v)) + 1


def postorder_traversal_circular(tree, v, root, i, k, parent, c, d, s):
    for w in children(tree, v):
        i = postorder_traversal_circular(tree, w, root, i, k, v, c, d, s)
    if deg_r(tree, v, root) == 1:
        c[v] = 0
        d[v] = np.array((cos(2 * pi * i / k), sin(2 * pi * i / k)))
        i += 1
    else:
        S = 0
        if parent is None:
            neighbors = children(tree, v)
        else:
            neighbors = chain([parent], children(tree, v))
        for w in neighbors:
            if v == root:
                s[v, w] = 1 / distance(tree, v, w)
                S += s[v, w]
            elif w == parent:
                s[w, v] = 1 / (distance(tree, w, v))
                S += s[w, v]
            else:
                s[v, w] = 1 / (distance(tree, v, w) * (deg_r(tree, v, root) - 1))
                S += s[v, w]

        t1 = 0
        t2 = 0
        for w in children(tree, v):
            t1 += s[v, w] / S * c[w]
            t2 += s[v, w] / S * d[w]
        if v != root:
            c[v] = s[parent, v] / (S * (1 - t1))
        d[v] = t2 / (1 - t1)
    return i


def preorder_traversal_circular(tree, v, root, parent, x, c, d):
    if v == root:
        x[v] = d[v]
    else:
        x[v] = c[v] * x[parent] + d[v]

    for w in children(tree, v):
        preorder_traversal_circular(tree, w, root, v, x, c, d)


def get_points_circular(rooted_tree, root=0):
    """See Algorithm 2: CIRCLE-LAYOUT in:
    Bachmaier, Christian, Ulrik Brandes, and Barbara Schlieper.
    "Drawing phylogenetic trees." Algorithms and Computation (2005): 1110-1121.

    It is important to remove negative distances in the tree before running this function.
    """
    i = 0
    k = 0
    s = {}
    x = {}
    c = {}
    d = {}
    for v in rooted_tree:
        if deg_r(rooted_tree, v, root) == 1:
            k += 1

    postorder_traversal_circular(rooted_tree, root, root, i, k, None, c, d, s)
    preorder_traversal_circular(rooted_tree, root, root, None, x, c, d)

    return x
