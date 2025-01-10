# coding=utf-8

import os
import warnings

import SimpleITK as sitk
import numpy as np
import pydicom

from common_tools import logger
from common_tools import series_info_ct

warnings.filterwarnings("ignore")

MIN_SLICE = 50


def mkdir_with_check(dir_path):
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)


def load_scan(path):
    slices = []
    for s in os.listdir(path):
        if os.path.isdir(s):
            continue
        try:
            slices.append(pydicom.read_file(path + '/' + s))
        except Exception as e:
            print('print load dicom fail![{}]'.format(path + '/' + s))
            raise e
    slices.sort(key=lambda x: float(x.InstanceNumber), reverse=True)
    need_reverse_z = False
    try:
        slice_thickness = np.abs(slices[0].ImagePositionPatient[2] - slices[1].ImagePositionPatient[2])
    except:
        slice_thickness = np.abs(slices[0].SliceLocation - slices[1].SliceLocation)

    for s in slices:
        s.SliceThickness = slice_thickness

    return slices, need_reverse_z


def get_pixels_hu(slices):
    image = np.stack([s.pixel_array for s in slices])
    # Convert to int16 (from sometimes int16),
    # should be possible as values should always be low enough (<32k)
    image = image.astype(np.int16)

    # Convert to Hounsfield units (HU)
    for slice_number in range(len(slices)):
        intercept = slices[slice_number].RescaleIntercept
        slope = slices[slice_number].RescaleSlope

        if slope != 1:
            image[slice_number] = slope * image[slice_number].astype(np.float64)
            image[slice_number] = image[slice_number].astype(np.int16)

        image[slice_number] += np.int16(intercept)

    b = (slices[0].PixelSpacing)
    a = [slices[0].SliceThickness]
    c = b[:]
    return np.array(image, dtype=np.int16), np.array(a + c, dtype=np.float32)


def lumTrans(img):
    img[np.isnan(img)] = -2000
    lungwin = np.array([-1350., 150.])
    newimg = (img - lungwin[0]) / (lungwin[1] - lungwin[0])
    newimg[newimg < 0] = 0
    newimg[newimg > 1] = 1
    newimg = (newimg * 255).astype('uint8')
    return newimg


def extract_single_scan(single_series_dcm_path):
    slices = [[pydicom.read_file(_), _] for _ in glob.glob(single_series_dcm_path + '/*') if
              not os.path.basename(_).startswith('.')]
    all_dcm_files = [_[1] for _ in slices]
    slices.sort(key=lambda x: float(x[0].InstanceNumber), reverse=False)
    new_slice = []
    for it in range(len(slices)):
        if it != 0:
            if abs(slices[it][0].SliceLocation - slices[it - 1][0].SliceLocation) > 20:
                break
        new_slice.append(slices[it])
    # 删除无用序列
    reserved_dcm_files = [_[1] for _ in new_slice]
    need_delete_dcm_files = [_ for _ in all_dcm_files if _ not in reserved_dcm_files]
    if len(need_delete_dcm_files) > 0:
        logger.info('extract single scan from {} to {}'.format(len(all_dcm_files), len(reserved_dcm_files)))
        new_single_series_dcm_path = os.path.join(single_series_dcm_path, os.path.basename(single_series_dcm_path))
        os.makedirs(new_single_series_dcm_path, exist_ok=True)
        for sub_file in reserved_dcm_files:
            file_name = os.path.basename(sub_file)
            os.symlink(sub_file, os.path.join(new_single_series_dcm_path, file_name))
        return new_single_series_dcm_path, len(all_dcm_files), True
    return None, len(all_dcm_files), False


def prepare_one_series(dcm_path):
    if global_config['do_single_sane_extract']:
        new_single_series_dcm_path, ori_img_count, do_create_new_folder = extract_single_scan(dcm_path)
    else:
        new_single_series_dcm_path = None
        ori_img_count = len(glob.glob(dcm_path + '/*'))
    dcm_path = dcm_path if new_single_series_dcm_path is None else new_single_series_dcm_path
    case, need_reverse_z = load_scan(dcm_path)
    case_pixels, spacing = get_pixels_hu(case)
    reverse_params = {'need_reverse_z': need_reverse_z, 'ori_image_count': ori_img_count,
                      'new_single_series_dcm_path': new_single_series_dcm_path}
    return case_pixels, lumTrans(case_pixels), spacing, reverse_params


def prepare_one_series_ITK_bk(dcm_path):
    reader = sitk.ImageSeriesReader()
    seriesIDs = reader.GetGDCMSeriesIDs(dcm_path)
    art_seriesid = None
    if len(seriesIDs) == 0:
        return None
    if len(seriesIDs) == 1:
        art_seriesid = seriesIDs[0]
        dicom_names = reader.GetGDCMSeriesFileNames(dcm_path, seriesIDs[0])
    elif len(seriesIDs) > 1:
        # find longest series
        dicom_names = None
        for series_id in seriesIDs:
            temp_dicom_names = reader.GetGDCMSeriesFileNames(dcm_path, series_id)
            if dicom_names is None or len(temp_dicom_names) > len(dicom_names):
                art_seriesid = series_id
                dicom_names = temp_dicom_names
    first_dicom = pydicom.read_file(dicom_names[0])
    least_dicom = pydicom.read_file(dicom_names[-1])
    need_reverse_z = False
    if least_dicom.InstanceNumber > first_dicom.InstanceNumber:
        need_reverse_z = True
    first_instance_dicom = first_dicom if need_reverse_z else least_dicom
    try:
        BodyPartExamined = first_instance_dicom.BodyPartExamined
    except:
        BodyPartExamined = ""
    try:
        ProtocolName = first_instance_dicom.ProtocolName
    except:
        ProtocolName = ""
    try:
        SeriesDescription = first_instance_dicom.SeriesDescription
    except:
        SeriesDescription = ''
    patient_id = first_instance_dicom.PatientID

    reader.SetFileNames(dicom_names)
    image = reader.Execute()
    image_array = sitk.GetArrayFromImage(image)  # z, y, x
    spacing = image.GetSpacing()
    origin = image.GetOrigin()
    direction = image.GetDirection()

    if len(spacing) != 3:
        raise RuntimeError('spacing require 3 values but got {} values'.format(len(spacing)))

    series_info = {
        'patient_id': patient_id,
        'series_id': art_seriesid,
        'need_reverse_z': need_reverse_z,
        'ori_image_count': image_array.shape[0],
        'spacing': spacing,
        'origin': origin,
        'direction': direction,
        'BodyPartExamined': BodyPartExamined,
        'ProtocolName': ProtocolName,
        'SeriesDescription': SeriesDescription,
    }
    spacing = [spacing[-1], spacing[1], spacing[0]]
    reverse_params = {'need_reverse_z': need_reverse_z, 'ori_image_count': image_array.shape[0],
                      'new_single_series_dcm_path': None,
                      'series_info': series_info}
    return image_array, lumTrans(image_array), np.array(spacing, dtype=np.float32), reverse_params


def prepare_one_series_ITK(dcm_path):
    series_info_dict, series_image_arrays = series_info_ct.get_series_info_from_path(dcm_path, return_images=True,
                                                                                     MIN_SLICE=MIN_SLICE)
    if len(series_info_dict) == 0:
        raise RuntimeError('No series can be found from {} with MIN_SLICE={}'.format(dcm_path, MIN_SLICE))
    matched_series_id = list(series_info_dict.keys())[0]
    if len(series_info_dict) > 1:
        # find the series with most images、
        max_image_count = 0
        for series_id in series_info_dict:
            im_count = series_info_dict[series_id]['image_count']
            if im_count > max_image_count:
                matched_series_id = series_id
                max_image_count = im_count

    series_info = series_info_dict[matched_series_id]
    image_array = series_image_arrays[matched_series_id]
    spacing = series_info['spacing']
    if len(spacing) != 3:
        raise RuntimeError('spacing require 3 values but got {} values'.format(len(spacing)))
    spacing = [spacing[-1], spacing[1], spacing[0]]
    reverse_params = {'need_reverse_z': not series_info['z_reverse'],
                      'image_count': image_array.shape[0],
                      'new_single_series_dcm_path': None,
                      'series_info': series_info}
    return image_array, lumTrans(image_array), np.array(spacing, dtype=np.float32), reverse_params


def prepare_ct_data(dcm_input_path):
    reader = sitk.ImageSeriesReader()
    seriesIDs = reader.GetGDCMSeriesIDs(dcm_input_path)
    seriesid_case_pixels_map = {}
    seriesid_images_map = {}
    auxiliary_parameters = {}
    for series_id in seriesIDs:
        dicom_names = reader.GetGDCMSeriesFileNames(dcm_input_path, series_id)
        if len(dicom_names) < MIN_SLICE:
            continue
        first_dicom_file = pydicom.read_file(dicom_names[0])

        if 'chest' not in str(first_dicom_file.ProtocolName).lower():
            # just process chest CT
            continue

        # make temp dir
        temp_dcm_path = os.path.join(dcm_input_path, series_id)
        mkdir_with_check(temp_dcm_path)
        for dcm_name in dicom_names:
            os.system('ln -s {} {}'.format(dcm_name, os.path.join(temp_dcm_path, os.path.basename(dcm_name))))
        case_pixels, image_array, spacing = prepare_one_series(temp_dcm_path)
        seriesid_case_pixels_map[series_id] = case_pixels
        seriesid_images_map[series_id] = image_array
        auxiliary_parameters[series_id] = spacing
    return seriesid_images_map, auxiliary_parameters

