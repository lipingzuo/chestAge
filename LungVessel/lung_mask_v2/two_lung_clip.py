import sys
from . import lung_mask
import os
import SimpleITK as sitk
import time
from common_tools import logger
from .data_processing import prepare_one_series_ITK
from .volume_measurement import find_ich_blocks_from_mask, filter_ich_block, find_max_area_block


def path(string):
    if os.path.exists(string):
        return string
    else:
        sys.exit(f'File not found: {string}')


class TwoLungClip(object):
    def __init__(self, gpu_id=None):
        if gpu_id is not None:
            os.environ['CUDA_VISIBLE_DEVICES'] = str(gpu_id)
        try:
            time_s = time.time()
            self.model = lung_mask.get_model('unet', 'R231', os.path.join(os.path.dirname(__file__),'unet_r231-d5d2fc3d.pth'), 3)

            logger.info('[LungClip] load model success! cost time:{:.3f}s'.format(time.time() - time_s))
            time_s = time.time()
            self.model.cuda()
            logger.info('[LungClip] convert model to cuda cost:{:.5f}s'.format(time.time() - time_s))

        except:
            raise RuntimeError('[LungClip] load model Error!')

    def load_ct_data_and_clip_lung(self, target_dcm_path, do_hull=True):
        time_s = time.time()
        logger.info('[LungClip] prepare image data...')
        case_pixels, image_array, spacing, reverse_params = prepare_one_series_ITK(target_dcm_path)

        logger.info('[LungClip] images shape:({},{},{})'.format(*case_pixels.shape))
        logger.info('[LungClip] spacing:({:.2f},{:.2f},{:.2f})'.format(*spacing))
        logger.info('[LungClip] load data cost time: {:.3f}s'.format(time.time() - time_s))
        time_s = time.time()
        batch_size = 10
        lung_seg_result = lung_mask.apply(case_pixels, self.model, force_cpu=False, batch_size=batch_size,
                            volume_postprocessing=False, noHU=False)
        logger.info('[LungClip] infer time: {:.3f}s'.format(time.time() - time_s))
        time_s = time.time()
        bw1 = lung_seg_result==1
        bw2 = lung_seg_result==2

        bw1_blocks = find_ich_blocks_from_mask(bw1)
        bw1_blocks = filter_ich_block(bw1_blocks)
        bw2_blocks = find_ich_blocks_from_mask(bw2)
        bw2_blocks = filter_ich_block(bw2_blocks)
        logger.info('[LungClip] lung blocks count: left({}) right({})'.format(len(bw1_blocks), len(bw2_blocks)))

        if len(bw1_blocks) == 0 or len(bw1_blocks) == 0:
            raise Exception('No Lung Can Be Found!')

        bw1_block = find_max_area_block(bw1_blocks)
        bw1_block_mask = bw1_block.get_block_mask(lung_seg_result.shape[0])
        bw2_block = find_max_area_block(bw2_blocks)
        bw2_block_mask = bw2_block.get_block_mask(lung_seg_result.shape[0])
        logger.info('[LungClip] post-process cost: {:.3f}s'.format(time.time() - time_s))

        if do_hull:
            time_s = time.time()
            bw1_hull_image = bw1_block.get_block_hull_mask(lung_seg_result.shape[0], bw1_block_mask)
            bw2_hull_image = bw2_block.get_block_hull_mask(lung_seg_result.shape[0], bw2_block_mask)
            logger.info('[LungClip] image hull cost: {:.3f}s'.format(time.time() - time_s))
            return case_pixels, bw1_block_mask > 0, bw2_block_mask > 0, spacing, reverse_params, image_array, bw1_hull_image > 0, bw2_hull_image > 0
        else:
            return case_pixels, bw1_block_mask > 0, bw2_block_mask > 0, spacing, reverse_params, image_array


    def load_numpy_data_clip_lung(self,case_pixels, do_hull=True):
        from .data_processing import lumTrans
        image_array = lumTrans(case_pixels)

        logger.info('[LungClip] images shape:({},{},{})'.format(*case_pixels.shape))
        time_s = time.time()
        batch_size = 10
        lung_seg_result = lung_mask.apply(case_pixels, self.model, force_cpu=False, batch_size=batch_size,
                                          volume_postprocessing=False, noHU=False)
        logger.info('[LungClip] infer time: {:.3f}s'.format(time.time() - time_s))
        time_s = time.time()
        bw1 = lung_seg_result == 1
        bw2 = lung_seg_result == 2

        bw1_blocks = find_ich_blocks_from_mask(bw1)
        bw1_blocks = filter_ich_block(bw1_blocks)
        bw2_blocks = find_ich_blocks_from_mask(bw2)
        bw2_blocks = filter_ich_block(bw2_blocks)
        logger.info('[LungClip] lung blocks count: left({}) right({})'.format(len(bw1_blocks), len(bw2_blocks)))

        if len(bw1_blocks) == 0 or len(bw1_blocks) == 0:
            raise Exception('No Lung Can Be Found!')

        bw1_block = find_max_area_block(bw1_blocks)
        bw1_block_mask = bw1_block.get_block_mask(lung_seg_result.shape[0])
        bw2_block = find_max_area_block(bw2_blocks)
        bw2_block_mask = bw2_block.get_block_mask(lung_seg_result.shape[0])
        logger.info('[LungClip] post-process cost: {:.3f}s'.format(time.time() - time_s))
        if do_hull:
            time_s = time.time()
            bw1_hull_image = bw1_block.get_block_hull_mask(lung_seg_result.shape[0], bw1_block_mask)
            bw2_hull_image = bw2_block.get_block_hull_mask(lung_seg_result.shape[0], bw2_block_mask)
            logger.info('[LungClip] image hull cost: {:.3f}s'.format(time.time() - time_s))
            return case_pixels, bw1_block_mask > 0, bw2_block_mask > 0, None,None, image_array, bw1_hull_image > 0, bw2_hull_image > 0
        else:
            return case_pixels, bw1_block_mask > 0, bw2_block_mask > 0,None,None, image_array

    def load_nii_data_and_clip_lung(self, nii_path, do_hull=True):
        from common_tools.dicom.nifti import read_nii_with_param
        from .data_processing import lumTrans
        time_s = time.time()
        logger.info('[LungClip] prepare image data...')
        case_pixels, origin, spacing, direction = read_nii_with_param(nii_path)
        series_info={"spacing":spacing,'origin':origin,'direction':direction}
        image_array = lumTrans(case_pixels)
        logger.info('[LungClip] images shape:({},{},{})'.format(*case_pixels.shape))
        logger.info('[LungClip] spacing:({:.2f},{:.2f},{:.2f})'.format(*spacing))
        logger.info('[LungClip] load data cost time: {:.3f}s'.format(time.time() - time_s))
        time_s = time.time()
        batch_size = 10
        lung_seg_result = lung_mask.apply(case_pixels, self.model, force_cpu=False, batch_size=batch_size,
                            volume_postprocessing=False, noHU=False)
        logger.info('[LungClip] infer time: {:.3f}s'.format(time.time() - time_s))
        time_s = time.time()
        bw1 = lung_seg_result==1
        bw2 = lung_seg_result==2

        bw1_blocks = find_ich_blocks_from_mask(bw1)
        bw1_blocks = filter_ich_block(bw1_blocks)
        bw2_blocks = find_ich_blocks_from_mask(bw2)
        bw2_blocks = filter_ich_block(bw2_blocks)
        logger.info('[LungClip] lung blocks count: left({}) right({})'.format(len(bw1_blocks), len(bw2_blocks)))

        if len(bw1_blocks) == 0 or len(bw1_blocks) == 0:
            raise Exception('No Lung Can Be Found!')

        bw1_block = find_max_area_block(bw1_blocks)
        bw1_block_mask = bw1_block.get_block_mask(lung_seg_result.shape[0])
        bw2_block = find_max_area_block(bw2_blocks)
        bw2_block_mask = bw2_block.get_block_mask(lung_seg_result.shape[0])
        logger.info('[LungClip] post-process cost: {:.3f}s'.format(time.time() - time_s))

        if do_hull:
            time_s = time.time()
            bw1_hull_image = bw1_block.get_block_hull_mask(lung_seg_result.shape[0], bw1_block_mask)
            bw2_hull_image = bw2_block.get_block_hull_mask(lung_seg_result.shape[0], bw2_block_mask)
            logger.info('[LungClip] image hull cost: {:.3f}s'.format(time.time() - time_s))
            return case_pixels, bw1_block_mask > 0, bw2_block_mask > 0, spacing, {'series_info':series_info}, image_array, bw1_hull_image > 0, bw2_hull_image > 0
        else:
            return case_pixels, bw1_block_mask > 0, bw2_block_mask > 0, spacing, {'series_info':series_info}, image_array

