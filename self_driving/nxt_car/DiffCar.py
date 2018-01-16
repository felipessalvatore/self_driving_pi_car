import nxt
import nxt_bluetooth


class DiffCar(object):
    """
    to do
    """
    def __init__(self,
                 turn_ratio=0,
                 power_up=40,
                 power_down=-40,
                 power_left=20,
                 tacho_left=30,
                 power_right=20,
                 tacho_right=30):
        self.sock, self.brick = nxt_bluetooth.connectCar()
        self.leftMotor = nxt.Motor(self.brick, nxt.PORT_C)
        self.rightMotor = nxt.Motor(self.brick, nxt.PORT_B)
        self.both = nxt.SynchronizedMotors(self.leftMotor,
                                           self.rightMotor,
                                           turn_ratio)
        self.power_up = power_up
        self.power_down = power_down
        self.power_left = power_left
        self.tacho_left = tacho_left
        self.power_right = power_right
        self.tacho_right = tacho_right

    def move_up(self):
        """
        to do
        """
        self.both.run(self.power_up)

    def move_left(self):
        """
        to do
        """
        self.rightMotor.weak_turn(self.power_left, self.tacho_left)
        self.leftMotor.weak_turn(- self.power_left, self.tacho_left)

    def move_right(self):
        """
        to do
        """
        self.rightMotor.weak_turn(- self.power_right, self.tacho_right)
        self.leftMotor.weak_turn(self.power_right, self.tacho_right)

    def move_down(self):
        """
        to do
        """
        self.both.run(self.power_down)

    def idle(self):
        """
        to do
        """
        self.leftMotor.idle()
        self.rightMotor.idle()
