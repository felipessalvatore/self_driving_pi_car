import unittest
import bluetooth as blue


class BluetoothTest(unittest.TestCase):
 
    @classmethod
    def setUp(cls):
       cls.ID = '00:16:53:17:EF:0A' # MAC address NXT11

    def test_rasberry_can_find_nxt_via_bluetooth(self):
        blue_list = blue.discover_devices()
        self.assertIn(self.ID, blue_list)

    def test_bluetooth_connection(self):
        socket = blue.BluetoothSocket(blue.RFCOMM)
        socket.connect((self.ID, 1))
        peer_name = socket.getpeername() 
        self.asserEqual(self.ID, peer_name[0])