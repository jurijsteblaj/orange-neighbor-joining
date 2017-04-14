# Neighbor Joining for Orange

This project will implement [neighbor joining](https://en.wikipedia.org/wiki/Neighbor_joining) for [Orange](https://github.com/biolab/orange3). 

## neighbor_joining

`tree = neighbor_joining(distance_matrix)`
Construct a tree structure from a distance matrix. The result is a dictionary `{index1: [[index2, distance], ...], ...}`.

`rooted_tree = rooted(tree, root=0)`
Make all connections one-directional and reachable from root. The root node is `0` by default but can be provided as the second argument.
**Note**: At the moment the root node doesn't get a label.

`points = get_points(rooted_tree, root=0)`
Get coordinates of points. The result is a dictionary `{index: [x, y], ...}`. The root node is `0` by default but can be provided as the second argument. It must match the root that was given to `rooted`.

`plot(rooted_tree, points, labels=[], classes=None)`
Draws the tree. Keyword argument `classes` is a list of lists of indexes. Lists of indexes contain indexes of nodes of the same class. They will be colored with the same color. Keyword argument `labels` is a list of strings which are displayed on the plot next to nodes with corresponding indexes.

