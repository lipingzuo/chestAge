# coding=utf-8
import os
import SimpleITK as sitk
import numpy as np
import sys
import cv2
import math
import tqdm

MIN_ICH_LINKED_SLICE_COUNT = 10
MIN_ICH_CONTOUR_SIZE = 50


class LineSegment(object):
    def __init__(self, point1, point2, distance, z_index=0):
        self.point1 = point1
        self.point2 = point2
        self.distance = distance
        self.z_index = z_index

    def __str__(self):
        return 'from point ({},{}) to point ({},{}), z_index={}, distance: {:.6f} mm'.format(self.point1[0],
                                                                                             self.point1[1],
                                                                                             self.point2[0],
                                                                                             self.point2[1],
                                                                                             self.z_index,
                                                                                             self.distance)


class ICH_Block(object):

    def __init__(self, image_size):
        self.start_slice = -1
        self.end_slice = -1
        self.contours = []
        self.image_size = image_size
        self.long_dia = None
        self.short_dia = None
        self.hemorrhage_volume = None

    def join(self, ich_block, replacement=True):
        new_start_slice = min(self.start_slice, ich_block.start_slice)
        new_end_slice = max(self.end_slice, ich_block.end_slice)
        new_contours = []
        for it_slice in range(new_start_slice, new_end_slice + 1):
            new_contours.append(
                self.search_contour_by_slice_id(it_slice) + ich_block.search_contour_by_slice_id(it_slice))
        if replacement:
            self.start_slice = new_start_slice
            self.end_slice = new_end_slice
            self.contours = new_contours
            return self
        else:
            ich_block_return = ICH_Block(self.image_size)
            ich_block_return.start_slice = new_start_slice
            ich_block_return.end_slice = new_end_slice
            ich_block_return.contours = new_contours
            return ich_block_return

    def search_contour_by_slice_id(self, slice_id):
        if self.start_slice <= slice_id <= self.end_slice:
            return self.contours[slice_id - self.start_slice]
        else:
            return []

    def append(self, slice_id, contour):
        if self.start_slice == -1:
            self.start_slice = slice_id
            self.contours = [[contour]]
            self.end_slice = self.start_slice
        else:
            if slice_id <= self.end_slice:
                self.contours[slice_id - self.start_slice].append(contour)
            elif slice_id == self.end_slice + 1:
                self.contours.append([contour])
                self.end_slice = slice_id
            else:
                raise RuntimeError('slice_id not allowed')

    def check_overlap(self, slice_id, contour):
        if slice_id > self.end_slice or slice_id < self.start_slice:
            return False
        mask_in = cv2.drawContours(np.zeros(self.image_size), [contour], 0, 1, -1)
        for it_contour in self.contours[slice_id - self.start_slice]:
            mask_it = cv2.drawContours(np.zeros(self.image_size), [it_contour], 0, 1, -1)
            if np.sum(mask_it * mask_in) > 0:
                return True
        return False

    def slice_count(self):
        """
        血灶连续层数
        :return:
        """
        return self.end_slice - self.start_slice + 1

    def max_area_slice(self, single_contour=True):
        """
        找到最大出血面积的层（单体轮廓面积最大）
        :return: 层ID , 出血面积
        """
        max_area_slice_id = -1
        max_area_value = -1
        max_single_area_value = -1
        max_single_area_slice_id = -1
        max_single_area_contour_id = 0
        for it, it_slice_contour in enumerate(self.contours):
            area_slice = 0
            for it_c, it_contour in enumerate(it_slice_contour):
                area_sub_contour = cv2.contourArea(it_contour)
                area_slice += area_sub_contour
                if area_sub_contour > max_single_area_value:
                    max_single_area_value = area_sub_contour
                    max_single_area_slice_id = self.start_slice + it
                    max_single_area_contour_id = it_c
            if area_slice > max_area_value:
                max_area_value = area_slice
                max_area_slice_id = self.start_slice + it
        if single_contour:
            return max_single_area_slice_id, max_single_area_value, max_single_area_contour_id
        return max_area_slice_id, max_area_value, max_single_area_contour_id

    def cal_hemorrhage_volume(self):
        """
        :param spacing: spacing with x,y,z direction
        :return:
        """
        total_area_pixel = 0
        for it, it_slice_contour in enumerate(self.contours):
            for it_c, it_contour in enumerate(it_slice_contour):
                total_area_pixel += cv2.contourArea(it_contour)
        return total_area_pixel

    def get_block_mask(self, image_layer_num):
        mask_images = np.zeros((image_layer_num, self.image_size[0], self.image_size[1]))
        for it, it_slice_contour in enumerate(self.contours):
            mask_image = np.zeros(self.image_size)
            for it_c, it_contour in enumerate(it_slice_contour):
                mask_image = cv2.drawContours(mask_image, it_slice_contour, it_c, 1, -1)
            mask_images[self.start_slice + it] = mask_image
        return mask_images

    def get_block_hull_mask(self, image_layer_num,mask_images):
        hull_images = np.zeros((image_layer_num, self.image_size[0], self.image_size[1]))

        for it, it_slice_contour in enumerate(self.contours):
            if len(it_slice_contour) == 1:
                hull = cv2.convexHull(it_slice_contour[0])
            else:
                contour_concat = np.vstack(it_slice_contour)
                hull = cv2.convexHull(contour_concat)
                hull = cv2.convexHull(hull)  # to make sure it's the hull we need
            image_hull_maks = cv2.drawContours(hull_images[self.start_slice + it], [hull], 0, 1, -1)
            if np.sum(image_hull_maks) > 2 * np.sum(mask_images[self.start_slice + it]):
                hull_images[self.start_slice + it] =mask_images[self.start_slice + it]
            else:
                hull_images[self.start_slice + it] = image_hull_maks

        return hull_images


def find_ich_blocks_from_mask(masks):
    ich_blocks = {}
    ich_blocks_index = 0
    # current
    for slice_id in range(masks.shape[0]):
        counters_result = cv2.findContours((masks[slice_id] * 255).astype(np.uint8), cv2.RETR_TREE,
                                           cv2.CHAIN_APPROX_SIMPLE)
        try:
            _, contours_current, _ = counters_result
        except:
            contours_current, _ = counters_result
        # fliter ich size:
        filted_contours_current = []
        for _cc in contours_current:
            if cv2.contourArea(_cc) > MIN_ICH_CONTOUR_SIZE:
                filted_contours_current.append(_cc)
        contours_current = filted_contours_current
        if len(ich_blocks) == 0:
            if len(contours_current) > 0:
                for ich_slice_it, it_contour in enumerate(contours_current):
                    ich_item = ICH_Block(image_size=masks.shape[1:])
                    ich_item.append(slice_id, it_contour)
                    ich_blocks[ich_blocks_index] = ich_item
                    ich_blocks_index += 1
            continue
        for ich_slice_it, it_contour in enumerate(contours_current):
            overlaped_indexs = []  # record ich block id with overlap
            for ich_block_key in ich_blocks:
                if ich_blocks[ich_block_key].check_overlap(slice_id - 1, it_contour):
                    overlaped_indexs.append(ich_block_key)
            if len(overlaped_indexs) == 0:
                ich_item = ICH_Block(image_size=masks.shape[1:])
                ich_item.append(slice_id, it_contour)
                ich_blocks[ich_blocks_index] = ich_item
                ich_blocks_index += 1
            elif len(overlaped_indexs) == 1:
                ich_blocks[overlaped_indexs[0]].append(slice_id, it_contour)
            else:
                # step 1. add to the first matched block.
                ich_blocks[overlaped_indexs[0]].append(slice_id, it_contour)
                # step 2. merge all matched block, and then delete it.
                for ich_block_key in overlaped_indexs[1:]:
                    ich_blocks[overlaped_indexs[0]].join(ich_blocks[ich_block_key])
                    ich_blocks.pop(ich_block_key)
    return ich_blocks


def filter_ich_block(ich_blocks):
    need_poped_keys = []
    for key in ich_blocks:
        if ich_blocks[key].slice_count() < MIN_ICH_LINKED_SLICE_COUNT:
            need_poped_keys.append(key)
    for key in need_poped_keys:
        ich_blocks.pop(key)
    return ich_blocks


def find_max_area_block(ich_blocks):
    max_area = -1
    max_area_id = -1
    for key in ich_blocks:

        area = ich_blocks[key].cal_hemorrhage_volume()
        if area > max_area:
            max_area = area
            max_area_id = key
    if max_area_id == -1:
        return None
    else:
        return ich_blocks[max_area_id]

