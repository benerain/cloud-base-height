# train_cbh_predictor.py

import numpy as np
from joblib import dump
from mord import LogisticAT
from method.utils.utils import load_data_tiles, load_means_stds, load_ae_model
from method.models.models import ConvAutoEncoder
import torch

def main():
    # 设置参数
    data_dir = 'data/tiles/'  # 瓦片数据目录
    means_stds_file = 'data/means_stds_save.txt'
    ae_model_path = 'models_save/ae_ocean_savecheckpoint80.pt'
    or_model_save_path = 'models_save/cbh_prediction_mord_logistic_at.joblib'
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 加载数据集
    param_cols = ['cloud_top_height', 'cloud_optical_thickness', 'cloud_water_path']
    subset = [1, 3, 4]
    means, stds, mins, _ = load_means_stds(means_stds_file, log_transform=False, subset=subset)

    from method.utils.pytorch_class import ModisGlobalTilesDataset
    from torch.utils.data import DataLoader

    dataset = ModisGlobalTilesDataset(
        ddir=data_dir,
        ext='nc',
        subset=subset,
        subset_cols=param_cols,
        transform=None,
        subscale=False,
        grid_size=128,
        normalize=True,
        mean=means,
        std=stds,
        min=mins,
        log_transform=False,
        get_calipso_cbh=True  # 确保包含CBH标签
    )

    dataloader = DataLoader(dataset, batch_size=64, shuffle=False, num_workers=4)

    # 加载自编码器模型
    ae_model = ConvAutoEncoder(n_channels=3, input_grid_size=128, latent_dim=256)
    ae_model.load_state_dict(torch.load(ae_model_path, map_location=device))
    ae_model = ae_model.to(device)
    ae_model.eval()

    # 提取特征和标签
    encodings = []
    labels = []
    with torch.no_grad():
        for batch in dataloader:
            inputs = batch['data'].to(device, dtype=torch.float)
            encoding = ae_model.encode(inputs)
            encodings.append(encoding.cpu().numpy())
            labels.extend(batch['calipso_cbh_center'].numpy())

    encodings = np.vstack(encodings)
    labels = np.array(labels)

    # 将CBH值映射到序数类别
    classes = np.array([50., 100., 200., 300., 600., 1000., 1500., 2000., 2500.])
    y = np.digitize(labels, classes) - 1  # 类别从0开始


    labels.extend(batch['CLD_BASE_HT'].numpy())

     
    # 训练序数回归模型
    model = LogisticAT()
    model.fit(encodings, y)

    # 保存模型
    dump(model, or_model_save_path)
    print('序数回归模型已保存至', or_model_save_path)

if __name__ == '__main__':
    main()
