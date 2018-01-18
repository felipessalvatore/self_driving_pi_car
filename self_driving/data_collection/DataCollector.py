import sys
import os
import inspect
import keyboard as key
from util import get_date
import abc
import time

almost_current = os.path.abspath(inspect.getfile(inspect.currentframe()))
currentdir = os.path.dirname(almost_current)
parentdir = os.path.dirname(currentdir)
sys.path.insert(0, parentdir)

from nxt_car.DiffCar import DiffCar # noqa
from vision.Camera import Camera # noqa


class DataCollector():
    __metaclass__ = abc.ABCMeta

    """
    to do
    """
    def __init__(self, robot, cam, name=None):
        date = get_date()
        if name is not None:
            if not os.path.exists(name):
                os.mkdir(name)
            self.dir_name = os.path.join(name, date)
            self.pickle_name = os.path.join(name,
                                            date + '_pickle')
        else:
            self.dir_name = date
            self.pickle_name = os.path.join(date + '_pickle')
        os.mkdir(self.dir_name)
        self.data_dict = {}
        self.count = 0
        self.robot = robot
        self.camera = cam

    def save_image_write_dict(self, image, command):
        """
        to do
        """
        name = str(self.count) + ".png"
        name = os.path.join(self.dir_name, name)
        self.camera.save_image(name, image)
        self.data_dict[str(self.count)] = command
        self.count += 1

    @abc.abstractmethod
    def generate(self):
        """
        to do
        """
        return


class BasicDiffCollector(DataCollector):
    """
    to do
    """

    def __init__(self, robot, cam, name):
        super(BasicDiffCollector,self).__init__(robot, cam, name)

    def generate(self):
        while True:
            img = self.camera.take_picture_rgb()

            print "Working"

            if key.is_pressed('q'):
                print('Exiting...')
                break

            elif key.is_pressed('up'):
                self.robot.move_up()
                self.save_image_write_dict(img, 'up')
                time.sleep(0.05)

            elif key.is_pressed('down'):
                self.robot.move_down()
                self.save_image_write_dict(img, 'down')
                time.sleep(0.05)

            elif key.is_pressed('left'):
                self.robot.move_left()
                self.save_image_write_dict(img, 'left')

            elif key.is_pressed('right'):
                self.robot.move_right()
                self.save_image_write_dict(img, 'right')
            else:
                self.robot.idle()


if __name__ == '__main__':
    name = "basic_test"
    robot = DiffCar(bluetooth=False)
    cam = Camera()
    dc = BasicDiffCollector(robot, cam, name)
    dc.generate()
    if robot.btCon:
        robot.disconnect(robot.sock)
    print(dc.data_dict)
    time.sleep(0.3) # waits for keyboard thread to shutdown
