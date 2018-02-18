import sys
import os
import inspect
import keyboard as key
from util import get_date
import abc
import time
import pickle
import argparse

almost_current = os.path.abspath(inspect.getfile(inspect.currentframe()))
currentdir = os.path.dirname(almost_current)
parentdir = os.path.dirname(currentdir)
sys.path.insert(0, parentdir)

from nxt_car.DiffCar import DiffCar  # noqa
from vision.Camera import Camera  # noqa


class DataCollector():
    __metaclass__ = abc.ABCMeta

    """
    Abstract class to collect images and commands,
    both to the classification setting such as the
    regression setting.

    :param robot: robot class to control the nxt car
    :type robot: nxt_car.DiffCar
    :param cam: camera class to take pictures
    :type cam: vision.Camera
    :param name: name the folder in which all the pictures
                 will be saved
    :type name: None or str
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
        Given one image and one command this method,
        stores the image with name "self.count".png
        and stores the given command in the dict
        {name: command}.

        :param image: image taked by the camera
        :type image: np.array
        :param command: the real command passed to the robot,
                        it can be a class ("up", "down", etc.)
                        or a vector ([acceleration, steering wheel angle])
        :type command: int or np.array
        """
        name = str(self.count) + ".png"
        name = os.path.join(self.dir_name, name)
        self.camera.save_image(name, image)
        self.data_dict[str(self.count)] = command
        self.count += 1

    @abc.abstractmethod
    def generate(self):
        """
        Method to generate the dataset.
        """
        return


class BasicDiffCollector(DataCollector):
    """
    Collector class for the differential model.
    In this case each image will be classified
    as "up", "down", "left" and "right".

    :param robot: robot class to controll the nxt car
    :type robot: nxt_car.DiffCar
    :param cam: camera class to take picture
    :type cam: vision.Camera
    :param name: name the folder in which all the pictures
                 will be saved
    :type name: str
    """

    def __init__(self, robot, cam, name):
        super(BasicDiffCollector, self).__init__(robot, cam, name)

    def generate(self):
        """
        Method to generate the dataset.
        The car is controlled with the keyboard
        using the arrow keys, to exit just type "q".
        """
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
                time.sleep(0.05)

            elif key.is_pressed('left'):
                self.robot.move_left()
                self.save_image_write_dict(img, 'left')

            elif key.is_pressed('right'):
                self.robot.move_right()
                self.save_image_write_dict(img, 'right')
            else:
                self.robot.idle()
        with open(self.pickle_name, "wb") as f:
            pickle.dump(self.data_dict, f)


def main():
    """
    Script to collect data of the Diffcar robot.
    """
    parser = argparse.ArgumentParser()

    parser.add_argument("-n",
                        "--name",
                        type=str,
                        default="pista",
                        help="folder name (default='pista')")
    user_args = parser.parse_args()
    robot = DiffCar(bluetooth=False)
    cam = Camera()
    dc = BasicDiffCollector(robot, cam, user_args.name)
    dc.generate()
    if robot.btCon:
        robot.disconnect(robot.sock)
    print(dc.data_dict)
    time.sleep(0.3)  # waits


if __name__ == '__main__':
    main()
