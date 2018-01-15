import cv2

def getCamera(width_size=160,
              height_size=90,
              input_cam_device=0):
    '''
    Get a camera object for recording,
    we can set the width and height of
    the image that will be recorded.

    :param width_size: width of the image 
    :type width_size: int
    :param height_size: height of the image
    :type height_size: int
    :param input_cam_device: camera port
    :type input_cam_device: int
    :rtype: cv2.VideoCapture
    '''
    cam = cv2.VideoCapture(input_cam_device)
    height_param = 4
    width_param = 3
    cam.set(width_param, width_size)
    cam.set(height_param, height_size)
    return cam


# start = time.time()
# cam = cv2.VideoCapture(0)


# print(cam.get(cv2.CAP_PROP_FRAME_WIDTH))


# cam.set(3, 90)
# cam.set(4, 160)

# print(cam.get(cv2.CAP_PROP_FRAME_WIDTH))


# count = 0

# while count<10:
#     bool, im = cam.read()
#     im = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
#     cv2.imwrite("teste_uol/" + str(count) + "UOL.png", im)
#     print(im.shape)
#     count+=1

# print(time.time() - start)

# cv2.imshow('LIAMF-COOL',im)
# cv2.waitKey(0)
# cv2.destroyAllWindows()
