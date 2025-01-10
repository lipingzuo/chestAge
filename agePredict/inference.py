import torch
from .lrcn_model import ConvLstm


device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')


def init_model():
    model_file = "./weights/best_vgg11.pth"
    # ====== set model ======
    model = ConvLstm(latent_dim=512, hidden_size=256, lstm_layers=2, bidirectional=True, n_class=1)
    model.load_state_dict(torch.load(model_file, map_location='cpu')['model'])
    model = model.to(device)
    model.eval()
    return model


def read_png(img_path):
    import os
    import cv2
    import numpy as np
    import glob
    img_files_list = glob.glob(f"{img_path}/*.png")
    img_files = []
    for img in img_files_list:
        img = os.path.basename(img)
        img_files.append(img)
    img_files = sorted(img_files, key=lambda x: int(x.split('.png')[0]))
    img_data = []
    for img_file in img_files:
        img_file = os.path.join(img_path, img_file)
        img = cv2.imread(img_file)
        img = img / 255
        img = np.transpose(img, (2, 0, 1))
        img_data.append(img)
    img_data = np.stack(img_data)  # t, c, h, w

    img_data = torch.from_numpy(img_data)
    img_data = torch.unsqueeze(img_data, dim=0)
    img_data = img_data.to(device, dtype=torch.float32)

    return img_data


def predict(img_dir):
    model = init_model()
    img_data = read_png(img_dir)
    with torch.no_grad():
        pred = model(img_data)
    pred = torch.squeeze(pred, dim=0)
    pred_age = round(float(pred.detach().cpu().numpy()) * 100, 2)

    return pred_age
