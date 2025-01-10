import SimpleITK as sitk
import numpy as np
import os
import cv2
import pydicom


def dcm2nii(dicom_path, save_dir):
    basename = os.path.basename(dicom_path)

    # 如果给定的是文件，则提取其所在的目录
    if os.path.isfile(dicom_path):
        dicom_dir_path = os.path.dirname(dicom_path)
    else:
        dicom_dir_path = dicom_path

    reader = sitk.ImageSeriesReader()
    seriesIDs = reader.GetGDCMSeriesIDs(dicom_dir_path)

    if not seriesIDs:
        raise ValueError(f"No DICOM series found in directory: {dicom_dir_path}")

    N = len(seriesIDs)
    if N > 1:
        lens = np.zeros([N])
        for i in range(N):
            dicom_names = reader.GetGDCMSeriesFileNames(dicom_dir_path, seriesIDs[i])
            lens[i] = len(dicom_names)
        N_MAX = np.argmax(lens)
    else:
        N_MAX = 0

    dicom_names = reader.GetGDCMSeriesFileNames(dicom_dir_path, seriesIDs[N_MAX])
    reader.SetFileNames(dicom_names)
    image = reader.Execute()

    niigz_file = f'{save_dir}/{basename}.nii.gz'
    sitk.WriteImage(image, niigz_file)
    return niigz_file


def get_dcm_info(dicom_file):
    dcm = pydicom.read_file(dicom_file, force=True)
    # 获取患者姓名、性别、年龄
    # name = dcm.PatientName
    sex = dcm.PatientSex
    age = dcm.PatientAge

    # 获取像素数据并显示图像
    dcm.file_meta.TransferSyntaxUID = pydicom.uid.ImplicitVRLittleEndian
    img = dcm.pixel_array
    return img, sex, age


def get_nii(tempdir):
    """
    将 DICOM 文件转换为 nii.gz 文件，并获取相关信息
    :param tempdir: 临时目录
    :return: nii.gz 文件路径, 图像, 性别, 真实年龄
    """
    dicom_dir_path = os.path.join(tempdir, 'dicom')
    niigz_file = dcm2nii(dicom_dir_path, tempdir)

    # 获取第一个 DICOM 文件的信息
    dicom_files = os.listdir(dicom_dir_path)
    if not dicom_files:
        raise ValueError(f"No DICOM files found in directory: {dicom_dir_path}")

    dicom_file = os.path.join(dicom_dir_path, dicom_files[0])
    img, sex, true_age = get_dcm_info(dicom_file)

    return niigz_file, img, sex, true_age
