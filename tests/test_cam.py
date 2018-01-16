import unittest
import cv2
import os


class CamTest(unittest.TestCase):

    @classmethod
    def tearDown(cls):
        if os.path.exists(cls.img_path):
            os.remove(cls.img_path)

    @classmethod
    def setUp(cls):
        cls.img_path = os.path.join(os.getcwd(),
                                    "img.png")
        cls.cam = cv2.VideoCapture(0)
        height_param = 4
        width_param = 3
        width_size = 160
        height_size = 90
        cls.cam.set(width_param, width_size)
        cls.cam.set(height_param, height_size)

    def test_take_and_save_image_as_grayscale(self):
        _, img = self.cam.read()
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        cv2.imwrite(self.img_path, img)
        test_shape = (90, 160)
        self.assertEqual(test_shape, img.shape)
        self.assertTrue(os.path.exists(self.img_path))
