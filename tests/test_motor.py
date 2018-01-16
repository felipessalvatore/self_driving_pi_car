import unittest
from nxt.bluesock import BlueSock
import nxt
import time


class NxtTest(unittest.TestCase):

    @classmethod
    def tearDown(cls):
        cls.leftMotor.idle()
        cls.rightMotor.idle()
        cls.leftMotor.brake()
        cls.rightMotor.brake()
        cls.sock.close()

    @classmethod
    def setUp(cls):
        ID = '00:16:53:17:EF:0A'  # MAC address NXT11
        cls.sock = BlueSock(ID)
        brick = cls.sock.connect()
        cls.leftMotor = nxt.Motor(brick, nxt.PORT_C)
        cls.rightMotor = nxt.Motor(brick, nxt.PORT_B)
        cls.leftMotor.reset_position(False)
        cls.rightMotor.reset_position(False)

    def test_motor_left_right(self):
        turnDegrees = 360
        self.rightMotor.weak_turn(20, turnDegrees)
        self.leftMotor.weak_turn(- 20, turnDegrees)
        time.sleep(3)  # wait for the motors to stop
        rtc = abs(self.rightMotor.get_tacho().tacho_count)
        ltc = abs(self.leftMotor.get_tacho().tacho_count)
        self.assertAlmostEqual(rtc, turnDegrees, delta=3)
        self.assertAlmostEqual(ltc, turnDegrees, delta=3)
