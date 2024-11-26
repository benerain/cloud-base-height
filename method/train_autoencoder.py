# train_autoencoder.py

import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from method.models.models import ConvAutoEncoder
from method.utils.pytorch_class import ModisGlobalTilesDataset
import pandas as pd


from torch.utils.data import Dataset

class CustomDataset(Dataset):
    def __init__(self, data_file):
        # 加载和预处理数据
        self.data = load_and_preprocess_data(data_file)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        sample = self.data[idx]
        return sample
    

dataset = CustomDataset('data/test_set/cbh_preprocess_pytorch_filtered_obs_20082016_cldbaseht_test.csv')



def main():
    # 设置参数
    data_dir = 'data/tiles/'  # 瓦片数据目录
    means_stds_file = 'data/means_stds_save.txt'  # 均值和标准差文件
    model_save_dir = 'models_save/'  # 模型保存目录
    num_epochs = 80
    batch_size = 64
    learning_rate = 1e-3
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 加载均值和标准差
    def load_means_stds(file):
        df = pd.read_csv(file, header=0, skiprows=2, sep=' ', usecols=[0, 1, 2, 3, 4, 5])
        Means = df.iloc[0].values
        Stds = df.iloc[1].values
        Mins = df.iloc[2].values
        Maxs = df.iloc[3].values
        return Means, Stds, Mins, Maxs

    means, stds, mins, _ = load_means_stds(means_stds_file)
    param_cols = ['cloud_top_height', 'cloud_optical_thickness', 'cloud_water_path']
    subset = [1, 3, 4]  # 对应于上述参数的索引

    # 创建数据集和数据加载器
    dataset = ModisGlobalTilesDataset(
        ddir=data_dir,
        ext='nc',
        subset=subset,
        subset_cols=param_cols,
        transform=None,
        subscale=False,
        grid_size=128,
        normalize=True,
        mean=means[subset],
        std=stds[subset],
        min=mins[subset],
        log_transform=False,
        get_calipso_cbh=False
    )

    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=4)

    # 初始化模型
    model = ConvAutoEncoder(n_channels=3, input_grid_size=128, latent_dim=256)
    model = model.to(device)

    # 定义损失函数和优化器
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    # 训练模型
    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        for batch in dataloader:
            inputs = batch['data'].to(device, dtype=torch.float)
            optimizer.zero_grad()
            _, outputs = model(inputs)
            loss = criterion(outputs, inputs)
            loss.backward()
            optimizer.step()
            running_loss += loss.item() * inputs.size(0)
        epoch_loss = running_loss / len(dataset)
        print(f'Epoch {epoch+1}/{num_epochs}, Loss: {epoch_loss:.4f}')

        # 每10个epoch保存一次模型
        if (epoch + 1) % 10 == 0:
            checkpoint_path = os.path.join(model_save_dir, f'ae_ocean_savecheckpoint{epoch+1}.pt')
            torch.save(model.state_dict(), checkpoint_path)

    # 保存最终模型
    final_model_path = os.path.join(model_save_dir, 'ae_ocean_savecheckpoint80.pt')
    torch.save(model.state_dict(), final_model_path)
    print('训练完成，模型已保存至', final_model_path)

if __name__ == '__main__':
    main()
