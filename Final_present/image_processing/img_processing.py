import numpy as np
import cv2


def image_prepossessing(img_GRAY):
    # Building Mask
    # 二值化
    ret, thresh = cv2.threshold(img_GRAY, 10, 255, cv2.THRESH_BINARY)

    # Morphological transformation (去雜訊)
    kernel = np.ones((5, 5), np.uint8)
    closing = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)

    # Resize the image
    resizing_szie = (int(round(closing.shape[0] * 0.9, -1)), int(round(closing.shape[0] * 0.9, -1)))
    resizing_img = cv2.resize(closing, resizing_szie)
    background = np.zeros([img_GRAY.shape[0], img_GRAY.shape[1]])

    # Pasting image
    # 圖片貼上的起始位置
    x_loc = int((closing.shape[0] - resizing_img.shape[0]) / 2)
    y_loc = int((closing.shape[1] - resizing_img.shape[1]) / 2)

    # 圖片貼上的最終位置
    x_end = x_loc + resizing_img.shape[1]
    y_end = y_loc + resizing_img.shape[0]

    # Pasting
    background[y_loc:y_end, x_loc:x_end] = resizing_img
    mask = background.astype(np.uint8)
    # 用 uint 8 才能使用 bitwise 參數

    # Image processing
    # Gassium filter
    blur = cv2.GaussianBlur(img_GRAY, (71, 71), 3)

    # 直方圖均衡化(CLAHE)
    clahe = cv2.createCLAHE(clipLimit=8, tileGridSize=(30, 30))
    blur = clahe.apply(blur)

    # Combine
    Final = cv2.bitwise_or(blur, blur, mask=mask)
    return Final
