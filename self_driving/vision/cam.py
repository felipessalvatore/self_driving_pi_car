import cv2
import time

start = time.time()
cam = cv2.VideoCapture(0)


print(cam.get(cv2.CAP_PROP_FRAME_WIDTH))


cam.set(3, 90)
cam.set(4, 160)

print(cam.get(cv2.CAP_PROP_FRAME_WIDTH))


count = 0

while count<10:
    bool, im = cam.read()
    im = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
    cv2.imwrite("teste_uol/" + str(count) + "UOL.png", im)
    print(im.shape)
    count+=1

print(time.time() - start)

# cv2.imshow('LIAMF-COOL',im)
# cv2.waitKey(0)
# cv2.destroyAllWindows()
