# Neighbor Joining for Orange

This project will implement [neighbor joining](https://en.wikipedia.org/wiki/Neighbor_joining) for [Orange](https://github.com/biolab/orange3).

See Examples.ipynb for examples of use of functions in neighbor_joining.py.

## Installation

To install, run `pip install .` or `pip install -e .`.

## neighbor_joining.py

### `tree = run_neighbor_joining(distance_matrix)`

Construct a tree structure from a distance matrix. The result is a dictionary `{index1: [[index2, distance], ...], ...}`.

### `rooted_tree = make_rooted(tree, root=0)`

Make all connections one-directional and reachable from root. The root node is `0` by default but can be provided as the second argument.

### `points = get_points_radial(rooted_tree, root=0)`

Get coordinates of points. The result is a dictionary `{index: [x, y], ...}`. The root node is `0` by default but can be provided as the second argument. It must match the root that was provided to `rooted`.

