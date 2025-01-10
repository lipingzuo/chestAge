import os
import numpy as np


def resampleSpacing(sitkImage, newspace=(1, 1, 1)):
    import SimpleITK as sitk

    euler3d = sitk.Euler3DTransform()
    xsize, ysize, zsize = sitkImage.GetSize()
    xspacing, yspacing, zspacing = sitkImage.GetSpacing()
    origin = sitkImage.GetOrigin()
    direction = sitkImage.GetDirection()
    # 新的X轴的Size = 旧X轴的Size *（原X轴的Spacing / 新设定的Spacing）
    new_size = (int(xsize * xspacing / newspace[0]), int(ysize * yspacing / newspace[1]), int(zsize * zspacing / newspace[2]))
    # 如果是对标签进行重采样，模式使用最近邻插值，避免增加不必要的像素值
    sitkImage = sitk.Resample(sitkImage, new_size, euler3d, sitk.sitkNearestNeighbor, origin, newspace, direction)

    return sitkImage


def get_data(nii_file, sequence):
    import SimpleITK as sitk
    import numpy as np
    import random
    nii_info = sitk.ReadImage(nii_file)
    # resample
    new_nii_info = resampleSpacing(nii_info)

    nii_img = sitk.GetArrayFromImage(new_nii_info)  # (z, y, x)
    y = nii_img.shape[1]
    opt_move = random.randint(15, 23)  # random move
    y_list = np.linspace(opt_move, y - opt_move + 3, sequence)
    data_img = np.zeros([sequence, 3, nii_img.shape[0], nii_img.shape[2]], dtype=nii_img.dtype)
    for k, i in enumerate(y_list):
        i = round(i)
        img = nii_img[::-1, i-1: i+2, :]
        img = np.transpose(img, (1, 0, 2))
        data_img[k] = img
    return data_img


def generate_sequence_data(nii_file, sequence=25):
    save_npy_file = nii_file.replace(".nii.gz", ".npy")
    # get sequence data from nii_file
    npy_data = get_data(nii_file, sequence)
    np.save(save_npy_file, npy_data)
    return save_npy_file
