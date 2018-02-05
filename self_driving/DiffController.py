import argparse
import time
import tensorflow as tf
import numpy as np
import keyboard as key
from ml_training.DataHolder import DataHolder
from ml_training.Config import Config
from ml_training.Trainer import Trainer
from ml_training.DFN import DFN
from nxt_car.DiffCar import DiffCar
from vision.Camera import Camera
from ml_training.util import int2command


class DiffController():
    """
    Class that controls the Diffcar robot using one DFN.

    :param mode: param to control type of image
    :type mode: str
    :param bluetooth: param to control if the bluetooth will be used.
    :type bluetooth: bool
    """
    def __init__(self, mode="pure", bluetooth=False):
        assert mode == "pure" or mode == "green" or mode == "bin" or mode == "gray" # noqa
        self.robot = DiffCar(bluetooth=bluetooth)
        self.cam = Camera(mode=mode)
        self.mode = mode
        if mode == "pure":
            channels = 3
        else:
            channels = 1
        config = Config(channels=channels)
        data = DataHolder(config)
        graph = tf.Graph()
        network = DFN(graph, config)
        self.trainer = Trainer(graph, config, network, data)

    def get_command(self, img, label_dict=int2command):
        """
        Get command from model's prediction

        :param img: image
        :type img: np.array
        :param label_dict: dict translating label to command
        :type label_dict: dict
        """
        command_int = int(self.trainer.predict(img)[0])
        command_int = label_dict[command_int]
        return command_int

    def image2float(self, img):
        """
        Change type and shape of the image's array
        according to self.mode.

        :param img: image
        :type img: np.array
        :rtype: np.array
        """
        if self.mode == "pure":
            img = img.astype(np.float32) / 255
            img = img.reshape((1, img.shape[0] * img.shape[1] * img.shape[2]))
            return img
        else:
            img = img.astype(np.float32) / 255
            img = img.reshape((1, img.shape[0] * img.shape[1]))
        return img

    def drive(self):
        """
        Drive car until the key "q" is pressed.
        """
        last_command = None
        while True:
            img = self.cam.take_picture()

            if key.is_pressed('q'):
                print('Exiting...')
                break
            else:
                img = self.image2float(img)
                command = self.get_command(img)
                print(command)
                if command == 'up':
                    self.robot.move_up()
                    time.sleep(0.05)
                elif command == 'down':
                    self.robot.move_down()
                    time.sleep(0.05)
                elif command == 'left':
                    self.robot.move_left()
                elif command == 'right':
                    self.robot.move_right()
                if last_command is not None:
                    if last_command != command:
                        self.robot.idle()
                        last_command = command


def main():
    """
    Script to let the car drive itself.
    """
    parser = argparse.ArgumentParser()

    parser.add_argument("-m",
                        "--mode",
                        type=str,
                        default="pure",
                        help="image mode (default='pure')")
    parser.add_argument("-b",
                        "--bluetooth",
                        type=bool,
                        default=False,
                        help="bluetooth control (default=False)")
    user_args = parser.parse_args()
    car = DiffController(user_args.mode, user_args.bluetooth)
    car.drive()
    if car.robot.btCon:
        car.robot.disconnect(car.robot.sock)
    time.sleep(0.3)  # waits for keyboard thread to shutdown


if __name__ == '__main__':
    main()
