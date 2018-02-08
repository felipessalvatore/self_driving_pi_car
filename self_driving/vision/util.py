import cv2


def write_img(img,
              prob,
              path,
              font=cv2.FONT_HERSHEY_PLAIN,
              fontScale=1,
              fontColor=(255, 25, 55),
              lineType=2,
              position1=(15, 25),
              position2=(15, 38),
              position3=(15, 52),
              position4=(15, 66)):
    """
    Write probabilistic distribution on image (4 classes).

    :param img: image
    :type img: np.array
    :param prob: probability of classess
    :type prob: np.array
    :param font: type of font
    :type font: cv2.FONT_HERSHEY_PLAIN
    :param fontScale: scale of font
    :type fontScale: float
    :param fontColor: font's Color
    :type fontColor: tuple
    :param lineType: type of the line
    :type lineType: int
    :param position1: position of the first prob
    :type position1: tuple
    :param position2: position of the second prob
    :type position2: tuple
    :param position3: position of the third prob
    :type position3: tuple
    :param position4: position of the fourth prob
    :type position4: tuple
    """

    cv2.putText(img, prob[0],
                position1,
                font,
                fontScale,
                fontColor,
                lineType)
    cv2.putText(img, prob[1],
                position2,
                font,
                fontScale,
                fontColor,
                lineType)
    cv2.putText(img, prob[2],
                position3,
                font,
                fontScale,
                fontColor,
                lineType)
    cv2.putText(img, prob[3],
                position4,
                font,
                fontScale,
                fontColor,
                lineType)
    cv2.imwrite(path, img)
