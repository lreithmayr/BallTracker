import unittest
from Tracker import Tracker
from collections import deque

class TestTracker(unittest.TestCase):
    def test_velocity_computation(self):
        cap = None
        tracker = Tracker(cap)
        tracker.position = ((75, 30), (46, 200))
        self.assertEqual(tracker.get_velocity(), deque([None, (-58, 340)]))


if __name__ == '__main__':
    unittest.main()
