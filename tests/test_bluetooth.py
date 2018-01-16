import unittest
import bluetooth as blue


class BluetoothTest(unittest.TestCase):

    @classmethod
    def setUp(cls):
        cls.ID = '00:16:53:17:EF:0A'  # MAC address NXT11

    def test_bluetooth_2_raspberry_can_find_nxt(self):
        blue_list = blue.discover_devices()
        self.assertIn(self.ID, blue_list)

    def test_bluetooth_1_connection(self):
        socket = blue.BluetoothSocket(blue.RFCOMM)
        socket.connect((self.ID, 1))
        peer_name = socket.getpeername()
        self.assertEqual(self.ID, peer_name[0])
