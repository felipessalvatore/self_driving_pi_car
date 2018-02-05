import nxt
import nxt_bluetooth


class DiffCar(object):
    """
    Class to communicate with the nxt differential car.

    :param turn_ratio: motor's ratio
    :type turn_ratio: int
    :param power_up: power to be used in the left and right motors
                     to move up
    :type power_up: int
    :param power_down: power to be used in the left and right motors
                       to move down
    :type power_down: int
    :param power_left: power to be used in the left motor
    :type power_left: int
    :param tacho_left: tacho count of the left motor
    :type tacho_left: int
    :param power_right: power to be used in the right motor
    :type power_right: int
    :param tacho_right: tacho count of the right motor
    :type tacho_right: int
    :param bluetooth: param to control if the bluetooth will be used.
    :type bluetooth: bool
    """
    def __init__(self,
                 turn_ratio=0,
                 power_up=40,
                 power_down=-40,
                 power_left=20,
                 tacho_left=30,
                 power_right=20,
                 tacho_right=30,
                 bluetooth=False):
        if bluetooth:
            self.sock, self.brick = nxt_bluetooth.connectCar()  # noqa PKSM '00:16:53:17:B4:04'
        else:
            self.brick = nxt.locator.find_one_brick()
        self.leftMotor = nxt.Motor(self.brick, nxt.PORT_B)
        self.rightMotor = nxt.Motor(self.brick, nxt.PORT_A)
        self.both = nxt.SynchronizedMotors(self.leftMotor,
                                           self.rightMotor,
                                           turn_ratio)
        self.power_up = power_up
        self.power_down = power_down
        self.power_left = power_left
        self.tacho_left = tacho_left
        self.power_right = power_right
        self.tacho_right = tacho_right
        self.btCon = bluetooth

    def move_up(self):
        """
        Execute one action of moving up
        """
        self.both.run(self.power_up)

    def move_left(self):
        """
        Execute one action of moving left
        """
        self.rightMotor.weak_turn(self.power_left, self.tacho_left)
        self.leftMotor.weak_turn(- self.power_left, self.tacho_left)

    def move_right(self):
        """
        Execute one action of moving rigth
        """
        self.rightMotor.weak_turn(- self.power_right, self.tacho_right)
        self.leftMotor.weak_turn(self.power_right, self.tacho_right)

    def move_down(self):
        """
        Execute one action of moving down
        """
        self.both.run(self.power_down)

    def idle(self):
        """
        Rest motors
        """
        self.leftMotor.idle()
        self.rightMotor.idle()

    def disconnect(self, socket):
        """
        Disconnect from bluetooth
        """
        nxt_bluetooth.disconnectCar(socket)
