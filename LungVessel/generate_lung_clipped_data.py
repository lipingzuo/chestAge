import os
import cv2
import numpy as np
import glob
from common_tools.dicom.series_process_ct import normalize_image
from common_tools.dicom.nifti import save_nii_with_param, read_nii_with_param

from .lung_mask_v2.two_lung_clip import TwoLungClip


def find_body_part_region(image_array):
    im = normalize_image(image_array, window_center=-550, window_width=100, as_uint8=True)
    body_part_mask = np.ones(shape=im.shape)
    for im_i in range(im.shape[0]):
        image = cv2.medianBlur(im[im_i], 11)
        _, image = cv2.threshold(image, 127, 255, cv2.THRESH_BINARY)
        counters_result = cv2.findContours(image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        try: \
                _, contours, _ = counters_result
        except:
            contours, _ = counters_result
        max_area_count_id = -1
        max_area = -1
        for c in range(len(contours)):
            area = cv2.contourArea(contours[c])
            if area > max_area:
                max_area = area
                max_area_count_id = c
        if max_area_count_id != -1:
            body_part_mask[im_i] = cv2.drawContours(np.zeros(image.shape), contours, max_area_count_id, 1, -1)
    return body_part_mask


def clip_lung(image_file, image_save_to, mask_file=None, mask_save_to=None):
    two_lung_clip = TwoLungClip()
    resolution = np.array([1, 1, 1])
    DO_CV_IMAGE_HULL = True

    two_lung_clip_result = two_lung_clip.load_nii_data_and_clip_lung(image_file, do_hull=DO_CV_IMAGE_HULL)

    if DO_CV_IMAGE_HULL:
        img_ori, m1, m2, spacing, reverse_params, im_uint8, m1_hull, m2_hull = two_lung_clip_result
    else:
        img_ori, m1, m2, spacing, reverse_params, im_uint8 = two_lung_clip_result

    img_ori[find_body_part_region(img_ori) == 0] = 0  # 去除背景，防止支气管误分割

    series_info = reverse_params['series_info']
    Mask = m1 + m2

    mask_int = np.ones_like(Mask, dtype=np.uint16)
    mask_int = mask_int * 2 * m1 + mask_int * m2  # 左肺1 右肺2

    temp_series_info = reverse_params['series_info']

    newshape = np.round(np.array(Mask.shape) * spacing / resolution)
    xx, yy, zz = np.where(Mask)

    box = np.array([[np.min(xx), np.max(xx)], [np.min(yy), np.max(yy)], [np.min(zz), np.max(zz)]])
    box = box * np.expand_dims(spacing, 1) / np.expand_dims(resolution, 1)
    margin = 5
    extendbox = np.vstack(
        [np.max([[0, 0, 0], box[:, 0] - margin], 0), np.min([newshape, box[:, 1] + 2 * margin], axis=0).T]).T

    ori_extendbox = extendbox * np.expand_dims(resolution, 1) / np.expand_dims(spacing, 1)
    ori_extendbox = np.floor(ori_extendbox).astype('int')

    # 处理原始图像
    print("image: ", img_ori.shape)
    img_clip = img_ori[ori_extendbox[0, 0]:ori_extendbox[0, 1],
               ori_extendbox[1, 0]:ori_extendbox[1, 1],
               ori_extendbox[2, 0]:ori_extendbox[2, 1]]
    save_nii_with_param(img_clip, series_info['origin'], series_info['spacing'], series_info['direction'],
                        image_save_to)
    if mask_file:
        mask_image, _, _, _ = read_nii_with_param(mask_file)
        print("mask: ", mask_image.shape)
        mask_clip = mask_image[ori_extendbox[0, 0]:ori_extendbox[0, 1],
                    ori_extendbox[1, 0]:ori_extendbox[1, 1],
                    ori_extendbox[2, 0]:ori_extendbox[2, 1]]
        save_nii_with_param(mask_clip, series_info['origin'], series_info['spacing'], series_info['direction'],
                            mask_save_to)

    return True


def get_lung(nii_file):
    os.environ['CUDA_VISIBLE_DEVICES'] = "0"
    clip_lung(nii_file, nii_file)
    return nii_file
