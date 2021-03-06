from copy import deepcopy
from itertools import chain
from math import cos, sin, pi

import numpy as np
from numpy import ma


def run_neighbor_joining(d, progress=None):
    """
    Run the neighbor joining algorithm on the distance matrix d and return a tree.
    The tree is structured as {index1: [[index2, distance12], ...], ...}.

    :param d:
    :param progress:
    :return tree:
    """
    real_nodes = d.shape[0]
    D = ma.empty((2*real_nodes-2, 2*real_nodes-2))
    D[:real_nodes,:real_nodes] = d
    D.mask = ma.make_mask_none(D.shape)
    D.mask[real_nodes:,:] = True
    D.mask[:,real_nodes:] = True
    joined_nodes = []
    
    result = {}
    current_nodes = real_nodes
    n = current_nodes - len(joined_nodes)
    while n > 2:
        # Based on the current distance matrix calculate the matrix Q
        row_sums = np.sum(D[:current_nodes,:current_nodes], axis=1)
        Q = ((n - 2)*D[:current_nodes,:current_nodes] - row_sums).T - row_sums
        
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
        result[i].append([current_nodes, dist_i])
        if j not in result:
            result[j] = []
        result[j].append([current_nodes, dist_j])
        if current_nodes not in result:
            result[current_nodes] = []
        result[current_nodes].append([i, dist_i])
        result[current_nodes].append([j, dist_j])
        
        # Calculate the distance from each of the taxa outside of this pair to the new node
        dist = D[i,j]
        i_col = D[:current_nodes,i]
        j_col = D[:current_nodes,j]
        new = (i_col + j_col - dist)/2
        
        # Add the new node to the distance matrix
        D[current_nodes,:current_nodes] = new
        D.mask[current_nodes,:current_nodes] = False
        D[:current_nodes,current_nodes] = new
        D.mask[:current_nodes,current_nodes] = False
        D[current_nodes,current_nodes] = 0
        D.mask[current_nodes,current_nodes] = False
        
        # Remove the two nodes which were joined
        D.mask[i,:] = True
        D.mask[j,:] = True
        D.mask[:,i] = True
        D.mask[:,j] = True

        joined_nodes.append(i)
        joined_nodes.append(j)

        D.mask[current_nodes,joined_nodes] = True
        D.mask[joined_nodes,current_nodes] = True

        current_nodes += 1
        n = current_nodes - len(joined_nodes)
        if progress is not None:
            progress.advance()
    
    # Join the last two remaining nodes
    i, j = np.where(np.logical_not(np.in1d(np.arange(len(D)), joined_nodes)))[0]
    
    if i not in result:
        result[i] = []
    result[i].append([j, D[i,j]])
    if j not in result:
        result[j] = []
    result[j].append([i, D[i,j]])
    
    return result


def get_children(tree, index):
    """
    Return a list of indexes of child nodes of a parent node in a tree.

    :param tree:
    :param index:
    :return children:
    """
    if len(tree[index]) == 0:
        return ()
    else:
        return list(zip(*tree[index]))[0]

    
def remove_backlinks(tree, child, parent):
    try:
        ix = get_children(tree, child).index(parent)
        del tree[child][ix]
    except ValueError:
        pass
    for c in get_children(tree, child):
        remove_backlinks(tree, c, child)

        
def make_rooted(tree, root=0):
    """
    Turn an unrooted tree into a rooted tree. All connections are turned from
    bidirectional to unidirectional while keeping all nodes reachable from the chosen root.

    :param tree:
    :param root:
    :return rooted_tree:
    """
    t = deepcopy(tree)
    for child in get_children(t, root):
        remove_backlinks(t, child, root)
    return t


def get_distance(tree, parent, child):
    """Return distance between a parent and a child node in a tree."""
    ix = get_children(tree, parent).index(child)
    return tree[parent][ix][1]


def postorder_traverse_radial(tree, v, l):
    if len(tree[v]) == 0:
        l[v] = 1
    else:
        l[v] = 0
        for w in get_children(tree, v):
            postorder_traverse_radial(tree, w, l)
            l[v] += l[w]

            
def preorder_traverse_radial(tree, v, parent, root, x, l, omega, tau):
    if v != root:
        u = parent
        angle = tau[v] + omega[v]/2
        x[v] = x[u] + get_distance(tree, u, v) * np.array((cos(angle), sin(angle)))
    eta = tau[v]
    for w in get_children(tree, v):
        omega[w] = 2*pi * l[w]/l[root]
        tau[w] = eta
        eta += omega[w]
        preorder_traverse_radial(tree, w, v, root, x, l, omega, tau)

        
def get_points_radial(rooted_tree, root=0):
    """See Algorithm 1: RADIAL-LAYOUT in:
    Bachmaier, Christian, Ulrik Brandes, and Barbara Schlieper.
    "Drawing phylogenetic trees." Algorithms and Computation (2005): 1110-1121.

    :param rooted_tree:
    :param root:
    :return:
    """
    l = {}
    x = {}
    omega = {}
    tau = {}
    
    postorder_traverse_radial(rooted_tree, root, l)
    
    x[root] = np.array((0, 0))
    omega[root] = 2*pi
    tau[root] = 0
    preorder_traverse_radial(rooted_tree, root, None, root, x, l, omega, tau)
    
    return x


def get_degree_rooted(rooted_tree, v, root):
    """Get degree of a node in a rooted tree."""
    if v == root:
        return len(get_children(rooted_tree, v))
    else:
        return len(get_children(rooted_tree, v)) + 1


def postorder_traverse_circular(tree, v, root, i, k, parent, c, d, s):
    for w in get_children(tree, v):
        i = postorder_traverse_circular(tree, w, root, i, k, v, c, d, s)
    if get_degree_rooted(tree, v, root) == 1:
        c[v] = 0
        d[v] = np.array((cos(2 * pi * i / k), sin(2 * pi * i / k)))
        i += 1
    else:
        S = 0
        if parent is None:
            neighbors = get_children(tree, v)
        else:
            neighbors = chain([parent], get_children(tree, v))
        for w in neighbors:
            if v == root:
                s[v, w] = 1 / get_distance(tree, v, w)
                S += s[v, w]
            elif w == parent:
                s[w, v] = 1 / (get_distance(tree, w, v))
                S += s[w, v]
            else:
                s[v, w] = 1 / (get_distance(tree, v, w) * (get_degree_rooted(tree, v, root) - 1))
                S += s[v, w]

        t1 = 0
        t2 = 0
        for w in get_children(tree, v):
            t1 += s[v, w] / S * c[w]
            t2 += s[v, w] / S * d[w]
        if v != root:
            c[v] = s[parent, v] / (S * (1 - t1))
        d[v] = t2 / (1 - t1)
    return i


def preorder_traverse_circular(tree, v, root, parent, x, c, d):
    if v == root:
        x[v] = d[v]
    else:
        x[v] = c[v] * x[parent] + d[v]

    for w in get_children(tree, v):
        preorder_traverse_circular(tree, w, root, v, x, c, d)


def get_points_circular(rooted_tree, root=0):
    """See Algorithm 2: CIRCLE-LAYOUT in:
    Bachmaier, Christian, Ulrik Brandes, and Barbara Schlieper.
    "Drawing phylogenetic trees." Algorithms and Computation (2005): 1110-1121.

    It is important to remove negative distances in the tree before running this function.

    :param rooted_tree:
    :param root:
    :return:
    """
    x = {}
    c = {}
    d = {}
    k = sum(1 for v in rooted_tree if get_degree_rooted(rooted_tree, v, root) == 1)

    postorder_traverse_circular(rooted_tree, root, root, 0, k, None, c, d, {})
    preorder_traverse_circular(rooted_tree, root, root, None, x, c, d)

    return x


def set_distance_floor(tree, min_dist):
    """
    Change distances of a tree so no distance in the tree is below min_dist. If a
    distance of a connection is increased another distance is decreased by the same
    amount if this doesn't cause the latter distance to be below min_dist.

    :param tree:
    :param min_dist:
    :return:
    """
    for l in tree.values():
        if len(l) == 1 and l[0][1] < min_dist:
            l[0][1] = min_dist
        if len(l) == 2:
            if l[0][1] + l[1][1] < 2 * min_dist:
                l[0][1] = l[1][1] = min_dist
            else:
                for i in range(2):
                    if l[i][1] < min_dist:
                        l[(i + 1) % 2][1] += l[i][1] - min_dist
                        l[i][1] = min_dist
        if len(l) == 3:
            if sum(c[1] for c in l) < 3 * min_dist:
                l[0][1] = l[1][1] = l[2][1] = min_dist
            else:
                for i in range(3):
                    if l[i][1] < min_dist:
                        l[(i + 1) % 3][1] += l[i][1] - min_dist
                        l[i][1] = min_dist

