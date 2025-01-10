import os
import zipfile
import tempfile
import shutil
import glob
import gradio as gr
from dicom2niigz.module import dcm2nii, get_dcm_info
from LungVessel.generate_lung_clipped_data import get_lung
from generate.generate_data import generate_sequence_data
from generate.preprocess_data import generate_png
from agePredict.inference import predict


def unzip_file(zip_file):
    tempdir = tempfile.mkdtemp()
    print("Temporary directory created at", tempdir)

    # 解压ZIP文件到临时目录
    with zipfile.ZipFile(zip_file, 'r') as zf:
        zf.extractall(tempdir)

    return tempdir


def destroy_dir(tempdir):
    # 删除临时目录及其内容
    shutil.rmtree(tempdir)


def get_nii(tempdir):
    folders = os.listdir(tempdir)
    dicom_path = os.path.join(tempdir, folders[0])
    niigz_file = dcm2nii(dicom_path, tempdir)

    dcm_files = glob.glob(f'{dicom_path}/*.dcm')
    dcm_file = dcm_files[int(len(dcm_files) / 2)]
    img, sex, true_age = get_dcm_info(dcm_file)
    true_age = int(true_age.split('Y')[0])
    destroy_dir(dicom_path)
    return niigz_file, img, sex, true_age


def show_png(img_path):
    import cv2
    img = cv2.imread(os.path.join(img_path, "12.png"))
    img_gray = img[:, :, 1]
    return img_gray


def run(zip_file):
    print("zip_file:", zip_file)
    tempdir = unzip_file(zip_file)
    chest_niigz_file, img, sex, true_age = get_nii(tempdir)  # chest dicom convert to nii.gz
    lung_niigz_file = get_lung(chest_niigz_file)  # generate lung from chest
    npy_file = generate_sequence_data(lung_niigz_file)  # resample and get sequence data (npz)
    img_path = generate_png(npy_file)  # generate image
    img_png = show_png(img_path)
    pred_age = predict(img_path)
    # destroy_dir(tempdir)
    return lung_niigz_file, img_png, sex, true_age, pred_age


def web_demo():
    demo = gr.Interface(fn=run,
                        inputs=gr.components.File(label="Upload ZIP file"),  # 使用 File 组件
                        outputs=[gr.components.File(label="Download nii.gz file"),
                                 gr.Image(label="picture", width=480, height=320),
                                 gr.Textbox(label="Sex", lines=1),
                                 gr.Textbox(label="Chronological Age", lines=1),
                                 gr.Textbox(label="Estimated Pulmonary Biological Age (ePBA)", lines=1),
                                 ]
                        )
    demo.launch(share=True)


if __name__ == '__main__':
    web_demo()
