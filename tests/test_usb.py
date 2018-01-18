import unittest
from nxt import locator
from nxt import usbsock


class USBTest(unittest.TestCase):

    @classmethod
    def setUp(cls):
        cls.connection_method = locator.Method(bluetooth=False)
        cls.generator = locator.find_bricks(method=cls.connection_method)

    def test_usb_raspberry_can_recognize_nxt(self):
        self.assertRaises(StopIteration, next(self.generator))

    def test_usb_check_usbsock_nxt(self):
        get_result = next(self.generator)
        self.assertIs(get_result, usbsock.USBSock)