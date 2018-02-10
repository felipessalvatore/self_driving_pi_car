import os
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
from vision.util import write_img


class DiffController():
    """
    Class that controls the Diffcar robot using one DFN.

    :param mode: param to control type of image
    :type mode: str
    :param bluetooth: param to control if the bluetooth will be used.
    :type bluetooth: bool
    """
    def __init__(self,
                 height,
                 width,
                 mode="pure",
                 bluetooth=False,
                 architecture=[4],
                 resize=100):
        assert mode == "pure" or mode == "green" or mode == "bin" or mode == "gray" # noqa
        self.robot = DiffCar(bluetooth=bluetooth)
        self.cam = Camera(mode=mode,
                          debug=True,
                          resize=resize / 100.0)
        self.mode = mode
        if mode == "pure":
            channels = 3
        else:
            channels = 1
        config = Config(channels=channels,
                        height=height,
                        width=width,
                        architecture=architecture)
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

    def get_command_and_prob(self, img, label_dict=int2command):
        """
        Get command from model's prediction

        :param img: image
        :type img: np.array
        :param label_dict: dict translating label to command
        :type label_dict: dict
        """
        prob = self.trainer.predict_prob(img)[0]
        result = np.argmax(prob, axis=0)
        result = result.astype(np.int32)
        command_int = int(result)
        command_int = label_dict[command_int]
        return command_int, prob

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
            img, _ = self.cam.take_picture()

            if key.is_pressed('q'):
                print('Exiting...')
                self.robot.idle()
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

    def drive_debug(self):
        """
        Drive car until the key "q" is pressed.
        images are stored in the folder "debug_run"
        to check the model performance.
        """
        if not os.path.exists("debug_run"):
            os.makedirs("debug_run")
        last_command = None
        count = 0
        while True:
            init = time.time()
            img, original_img = self.cam.take_picture()
            init = time.time() - init
            print("take_picture_time = {0:.3f}".format(init))

            if key.is_pressed('q'):
                print('Exiting...')
                self.robot.idle()
                break
            else:
                img_flatt = self.image2float(img)
                init = time.time()
                command, prob = self.get_command_and_prob(img_flatt)
                init = time.time() - init
                print("foward_time = {0:.3f}".format(init))
                commands = ['up', 'down', 'left', 'right']
                commands_prob = []
                for i, com in enumerate(commands):
                    commands_prob.append(com + ":{0:.2f}".format(prob[i]))
                print(commands_prob)
                print(command)
                name = os.path.join("debug_run", str(count) + ".png")
                write_img(original_img, commands_prob, name)
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
            count += 1


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
    parser.add_argument("-d",
                        "--debug",
                        action="store_true",
                        default=False,
                        help="debug (default=False)")  # noqa
    parser.add_argument("-he",
                        "--height",
                        type=int,
                        default=90,
                        help="image height (default=90)")
    parser.add_argument("-w",
                        "--width",
                        type=int,
                        default=160,
                        help="image width (default=160)")
    parser.add_argument('-a',
                        '--architecture',
                        type=int,
                        nargs='+',
                        help='sizes for hidden layers and output layer, should end with "4" !, (default=[4])',  # noqa
                        default=[4])
    parser.add_argument('-r',
                        '--resize',
                        type=int,
                        default=100,
                        help='resize percentage, (default=100)')
    user_args = parser.parse_args()
    car = DiffController(user_args.height,
                         user_args.width,
                         user_args.mode,
                         user_args.bluetooth,
                         user_args.architecture,
                         user_args.resize)
    if user_args.debug:
        car.drive_debug()
    else:
        car.drive()
    if car.robot.btCon:
        car.robot.disconnect(car.robot.sock)
    time.sleep(0.3)  # waits for keyboard thread to shutdown


if __name__ == '__main__':
    main()
