import unittest

import numpy as np

from neighborjoining.neighbor_joining import (
    run_neighbor_joining, make_rooted, get_points_radial, get_points_circular, get_children, set_distance_floor
)
from scipy.spatial.distance import squareform


class TestNeighborJoining(unittest.TestCase):
    def setUp(self):
        self.matrix = np.array([
            [0, 5, 9, 9, 8],
            [5, 0, 10, 10, 9],
            [9, 10, 0, 8, 7],
            [9, 10, 8, 0, 3],
            [8, 9, 7, 3, 0]])
        self.root = 0

        self.tree = {
            0: [[5, 2.0]],
            1: [[5, 3.0]],
            2: [[6, 4.0]],
            3: [[7, 2.0]],
            4: [[7, 1.0]],
            5: [[1, 3.0], [0, 2.0], [6, 3.0]],
            6: [[5, 3.0], [2, 4.0], [7, 2.0]],
            7: [[4, 1.0], [3, 2.0], [6, 2.0]]}

        self.rooted_tree = {
            0: [[5, 2.0]],
            1: [],
            2: [],
            3: [],
            4: [],
            5: [[1, 3.0], [6, 3.0]],
            6: [[2, 4.0], [7, 2.0]],
            7: [[4, 1.0], [3, 2.0]]}

        self.points_radial = {
            0: np.array([0, 0]),
            1: np.array([0.12132034, 2.12132034]),
            2: np.array([-6.94974747, 0.70710678]),
            3: np.array([-2.70710678, -5.53553391]),
            4: np.array([-4.82842712, -4.82842712]),
            5: np.array([-2.00000000e+00, 2.44929360e-16]),
            6: np.array([-4.12132034, -2.12132034]),
            7: np.array([-4.12132034, -4.12132034])}

        self.points_circular = {
            0: np.array([0.30901699, -0.95105652]),
            1: np.array([1., 0.]),
            2: np.array([0.30901699, 0.95105652]),
            3: np.array([-0.80901699, -0.58778525]),
            4: np.array([-0.80901699, 0.58778525]),
            5: np.array([0.40230053, -0.58611944]),
            6: np.array([0.08445164, -0.07742765]),
            7: np.array([-0.45162954, 0.08658599])
        }

    def assertDict(self, fun, a, b, **kwargs):
        self.assertEqual(len(a), len(b))
        for ka, kb in zip(a, b):
            fun(ka, kb, kwargs)
            for va, vb in zip(a[ka], b[kb]):
                for vva, vvb in zip(va, vb):
                    fun(vva, vvb, kwargs)

    def test_run_neighbor_joining(self):
        self.assertDict(self.assertAlmostEqual, run_neighbor_joining(self.matrix), self.tree)

    def test_make_rooted(self):
        self.assertAlmostEqual(make_rooted(self.tree, self.root), self.rooted_tree)

    def test_get_points_radial(self):
        points = get_points_radial(self.rooted_tree, self.root)
        np.testing.assert_array_almost_equal(np.array(list(points.keys())), np.array(list(self.points_radial.keys())))
        np.testing.assert_array_almost_equal(np.array(list(points.values())), np.array(list(self.points_radial.values())))

    def test_get_points_circular(self):
        points = get_points_circular(self.rooted_tree, self.root)
        np.testing.assert_array_almost_equal(np.array(list(points.keys())), np.array(list(self.points_circular.keys())))
        np.testing.assert_array_almost_equal(np.array(list(points.values())), np.array(list(self.points_circular.values())))

    def test_get_children(self):
        self.assertEqual(get_children(self.tree, 0), (5,))
        self.assertEqual(get_children(self.tree, 5), (1, 0, 6))
        self.assertEqual(get_children(self.rooted_tree, 1), ())
        self.assertEqual(get_children(self.rooted_tree, 5), (1, 6))

    def test_set_distance_floor(self):
        matrix = squareform(np.array(
            [0.290,
             0.289, 0.028,
             0.297, 0.058, 0.055,
             0.270, 0.134, 0.135, 0.134,
             0.293, 0.156, 0.155, 0.161, 0.139,
             0.299, 0.148, 0.147, 0.154, 0.156, 0.130,
             0.288, 0.187, 0.188, 0.181, 0.185, 0.205, 0.205,
             0.250, 0.196, 0.194, 0.198, 0.179, 0.214, 0.208, 0.210,
             0.274, 0.207, 0.209, 0.205, 0.193, 0.221, 0.219, 0.217, 0.092,
             0.250, 0.202, 0.197, 0.199, 0.181, 0.214, 0.213, 0.202, 0.081, 0.092
             ]))
        tree = run_neighbor_joining(matrix)
        rooted_tree = make_rooted(tree)
        min_dist = 1e-2
        # if this fails the the test is meaningless because min_dist is too low
        self.assertGreater(min_dist, min(y[1] for x in rooted_tree for y in rooted_tree[x]))
        set_distance_floor(rooted_tree, min_dist)
        self.assertLessEqual(min_dist, min(y[1] for x in rooted_tree for y in rooted_tree[x]))
