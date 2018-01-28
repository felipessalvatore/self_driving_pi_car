import unittest
import os
import sys
import inspect
import numpy as np

almost_current = os.path.abspath(inspect.getfile(inspect.currentframe()))
currentdir = os.path.dirname(almost_current)
parentdir = os.path.dirname(currentdir)
sys.path.insert(0, parentdir)


from util import run_test # noqa
from data_mani import create_data_set_as_np_array # noqa


class TestDataMani(unittest.TestCase):
    """
    Class that test the data manipulation functions
    """
    @classmethod
    def setUpClass(cls):
        cls.image_folder = currentdir
        cls.width = 160
        cls.height = 90
        cls.channels = 3
        cls.data_name = os.path.join(currentdir, "toy_data.npy")
        cls.label_name = os.path.join(currentdir, "toy_label.npy")

    @classmethod
    def tearDown(cls):
        if os.path.exists(cls.data_name):
            os.remove(cls.data_name)
        if os.path.exists(cls.label_name):
            os.remove(cls.label_name)

    def test_data_is_created_from_image_folder_and_pickle(self):
        create_data_set_as_np_array(self.image_folder,
                                    self.data_name,
                                    self.label_name,
                                    self.width,
                                    self.height,
                                    self.channels,
                                    verbose=False)
        data = np.load(self.data_name)
        labels = np.load(self.label_name)
        data_expected_shape = (25,
                               self.width * self.height * self.channels)
        self.assertEqual(data.shape, data_expected_shape)
        self.assertEqual(labels.shape, (25, 1))
        self.assertEqual(np.uint8, labels.dtype)
        self.assertEqual(np.uint8, data.dtype)


if __name__ == "__main__":
    run_test(TestDataMani,
             "\n=== Running data manipulation tests ===\n")
