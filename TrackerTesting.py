import unittest
from Tracker import Tracker
from collections import deque

class TestTracker(unittest.TestCase):

    def setUp(self) -> None:
        self.cap = None
        self.tracker = Tracker(self.cap)

    def test_velocity_computation(self):
        self.tracker.position = ((75, 30), (46, 200))
        self.assertEqual(self.tracker.get_velocity(), deque([None, (-58, 340)]))

    def test_acceleration_computation(self):
        self.tracker.velocity = ([None, (-58, 340), (46, 223)])
        self.assertEqual(self.tracker.get_accel(), deque([None, (208, -234)]))




if __name__ == '__main__':
    unittest.main(verbosity=2)
