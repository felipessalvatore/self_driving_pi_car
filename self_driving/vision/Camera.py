import cv2


class Camera(object):
    """
    to do
    """
    def __init__(self,
                 width_size=160,
                 height_size=90,
                 input_cam_device=0,
                 height_param=4,
                 width_param=3):
        self.cam = cv2.VideoCapture(input_cam_device)
        self.cam.set(width_param, width_size)
        self.cam.set(height_param, height_size)

    def save_image(self, path, img):
        """
        to do
        """
        cv2.imwrite(path, img)

    def take_picture_rgb(self):
        """
        to do
        """
        _, img = self.cam.read()
        return img

    def take_picture_gray(self):
        """
        to do
        """
        _, img = self.cam.read()
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        return img

    def take_picture_green(self):
        """
        to do
        """
        _, img = self.cam.read()
        return img[1]
