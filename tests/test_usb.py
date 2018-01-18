import unittest
from nxt import locator
from nxt import usbsock


class USBTest(unittest.TestCase):

    @classmethod
    def setUp(cls):
        cls.connection_method = locator.Method(bluetooth=False)
        cls.generator = locator.find_bricks(method=cls.connection_method)

    def test_usb_1_raspberry_is_not_connected(self):
        self.assertRaises(StopIteration, self.generator.next)

    def test_usb_2_check_usbsock_nxt(self):
        get_result = next(self.generator, None)
        self.assertIsInstance(get_result, usbsock.USBSock)
