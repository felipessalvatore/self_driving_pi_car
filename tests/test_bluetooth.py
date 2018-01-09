import unittest
import bluetooth


class BluetoothTest(unittest.TestCase):
 
    @classmethod
    def setUp(cls):
       self.ID = '00:16:53:17:EF:0A' # MAC address NXT11

    def test_rasberry_can_find_nxt_via_bluetooth(self):
        blue_list = bluetooth.discover_devices()
        self.assertIn(self.ID, blue_list)

    def test_bluetooth_connection(self):
        blue_list = bluetooth.discover_devices()
        self.assertIn(self.ID, blue_list)