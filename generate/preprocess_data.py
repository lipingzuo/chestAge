import os
import numpy as np
import cv2
from common_tools.dicom.series_process_ct import normalize_image


def windowing(img, window_lever, window_width):
    min_value = int(window_lever - window_width * 0.5)
    new_img = (img - min_value) / window_width
    new_img[new_img < 0] = 0
    new_img[new_img > 1] = 1
    new_img = (new_img * 255).astype('uint8')
    return new_img


def resize(data, long_side=320):

    t, c, h, w = data.shape
    new_data = np.zeros([t, c, long_side, long_side], dtype=data.dtype)

    h_radio = long_side / h
    w_radio = long_side / w

    for i in range(t):
        img = np.transpose(data[i], (1, 2, 0))
        img = normalize_image(img, -800, 1600)  # 肺窗
        # img = normalize_image(img, 40, 350) # 纵隔窗
        if h_radio < w_radio:
            r = h_radio
        else:
            r = w_radio

        new_h = int(h * r)
        new_w = int(w * r)

        top_dh = int(round((long_side - new_h) * 0.5))
        bottom_dh = long_side - new_h - top_dh
        left_dw = int(round((long_side - new_w) * 0.5))
        right_dw = long_side - new_w - left_dw

        img = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_NEAREST)
        img = cv2.copyMakeBorder(img, top=top_dh, bottom=bottom_dh, left=left_dw, right=right_dw, borderType=cv2.BORDER_CONSTANT, value=(0, 0, 0))
        new_data[i] = np.transpose(img, (2, 0, 1))

    return new_data


def generate_png(npy_file):
    img_path = os.path.dirname(npy_file)
    data = np.load(npy_file)
    new_data = resize(data, long_side=224)
        
    for i in range(new_data.shape[0]):
        img = new_data[i]
        img = img.transpose(1, 2, 0)
        cv2.imwrite(f'{img_path}/{i}.png', img)
    return img_path
