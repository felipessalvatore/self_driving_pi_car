import tensorflow as tf
import os
import numpy as np
import sys
import inspect
import keyboard as key
import time

almost_current = os.path.abspath(inspect.getfile(inspect.currentframe()))
currentdir = os.path.dirname(almost_current)
parentdir = os.path.dirname(currentdir)
sys.path.insert(0, parentdir)

print(parentdir)

from vision.image_manipulation import binarize_image, green_channel, grayscale_image # noqa
print(binarize_image)
from DataHolder import DataHolder # noqa
from Config import Config # noqa
from Trainer import Trainer # noqa
from DFN import DFN # noqa
from nxt_car.DiffCar import DiffCar # noqa
from vision.Camera import Camera # noqa


command2int = {"up": 0, "down": 1, "left": 2, "right": 3}
int2command = {i[1]: i[0] for i in command2int.items()}


class Controller():
    """
    to do
    """
    def __init__(self, robot, cam, mode="pure"):
        assert mode == "pure" or mode == "green" or mode == "bin" or mode == "gray"
        self.robot = robot
        self.cam = cam
        self.mode = mode
        record = ["a",  # HACK !!!!!!!!!!!!!!!! remove in the future
                  "b",
                  "c"]
        if mode == "pure":
            channels = 3
        else:
            channels = 1
        config = Config(channels=channels)
        data = DataHolder(config, records=record)
        graph = tf.Graph()
        network = DFN(graph, config)
        self.trainer = Trainer(graph, config, network, data)

    def get_command(self, img):
        command_int = int(self.trainer.predict(img)[0])
        command_int = int2command[command_int]
        return command_int

    def transform_image(self, img):
        if self.mode == "pure":
            img = img.astype(np.float32) / 255
            img = img.reshape((1, img.shape[0] * img.shape[1] * img.shape[2]))
            return img
        elif self.mode == "green":
            img = green_channel(img)
        elif self.mode == "bin":
            img = binarize_image(img)
        elif self.mode == "gray":
            img = grayscale_image(img)
        img = img.astype(np.float32) / 255
        img = img.reshape((1, img.shape[0] * img.shape[1]))
        return img

    def drive(self):

        while True:
            img = self.cam.take_picture_rgb()

            if key.is_pressed('q'):
                print('Exiting...')
                break
            else:
                img = self.transform_image(img)
                command = self.get_command(img)
                print(command)
                if command == 'up':
                    self.robot.move_up()
                    time.sleep(0.05)
                    self.robot.idle()
                elif command == 'down':
                    self.robot.move_down()
                    time.sleep(0.05)
                    self.robot.idle()
                elif command == 'left':
                    self.robot.move_left()
                    self.robot.idle()
                elif command == 'right':
                    self.robot.move_right()
                    self.robot.idle()


if __name__ == '__main__':
    robot = DiffCar(bluetooth=False)
    cam = Camera()
    car = Controller(robot, cam, mode="pure")
    car.drive()
    if robot.btCon:
        robot.disconnect(robot.sock)
    time.sleep(0.3)  # waits for keyboard thread to shutdown
