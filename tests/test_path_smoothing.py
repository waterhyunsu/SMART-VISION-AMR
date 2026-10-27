import unittest

from smart_vision_amr.path_geometry import line_is_clear, smooth_grid_path


class GridPathSmoothingTests(unittest.TestCase):
    def test_free_staircase_is_reduced_to_one_safe_segment(self):
        path = [(0, 0), (1, 0), (2, 0), (3, 1), (4, 1), (5, 2)]

        result = smooth_grid_path(path, lambda cell: False)

        self.assertEqual(result, [path[0], path[-1]])
        self.assertTrue(line_is_clear(result[0], result[-1], lambda cell: False))

    def test_smoothing_keeps_detour_around_blocked_cells(self):
        blocked = {(2, 1)}
        path = [(0, 1), (1, 0), (2, 0), (3, 0), (4, 1)]

        result = smooth_grid_path(path, lambda cell: cell in blocked)

        self.assertEqual(result[0], path[0])
        self.assertEqual(result[-1], path[-1])
        self.assertLess(len(result), len(path))
        for start, end in zip(result, result[1:]):
            self.assertTrue(line_is_clear(start, end, lambda cell: cell in blocked))

    def test_smoothing_does_not_cut_diagonal_obstacle_corners(self):
        blocked = {(1, 0)}
        path = [(0, 0), (0, 1), (1, 1)]

        result = smooth_grid_path(path, lambda cell: cell in blocked)

        self.assertEqual(result, path)
        self.assertFalse(line_is_clear((0, 0), (1, 1), lambda cell: cell in blocked))


if __name__ == "__main__":
    unittest.main()
