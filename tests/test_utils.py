import unittest

from rl_envs.blocksworld.utils import validate_state


class TestUtils(unittest.TestCase):
    def setUp(self) -> None:
        return super().setUp()

    def test_validate_state(self):
        self.assertTrue(validate_state([3, 1, 0]))
        self.assertTrue(validate_state([0, 0, 0]))
        self.assertTrue(validate_state([2, 3, 0]))

        self.assertFalse(validate_state([3, 2, 0]))
        self.assertFalse(validate_state([2, 3, 1]))
        self.assertFalse(validate_state([1, 3, 0]))

    def test_validate_larger_state(self):
        self.assertTrue(validate_state([3, 1, 0, 0, 0, 0]))
        self.assertTrue(validate_state([0, 0, 0, 0, 0, 0]))
        self.assertTrue(validate_state([2, 3, 0, 0, 0, 0]))

        self.assertFalse(validate_state([3, 2, 0, 0, 0, 0]))
        self.assertFalse(validate_state([2, 3, 1, 0, 0, 0]))
        self.assertFalse(validate_state([1, 3, 0, 0, 0, 0]))

        self.assertTrue(validate_state([3, 1, 0, 0, 4, 0, 0, 7]))
        self.assertTrue(validate_state([0, 0, 0, 6, 4, 0, 0, 7]))
        self.assertTrue(validate_state([2, 3, 8, 0, 4, 0, 0, 7]))

        self.assertFalse(validate_state([2, 4, 1, 0, 0, 0, 0, 1]))
        self.assertFalse(validate_state([1, 4, 0, 0, 0, 0, 0, 1]))


if __name__ == '__main__':
    unittest.main()
