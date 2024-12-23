import os
import numpy as np
import xarray as xr
import pandas as pd
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import pandas as pd

# from method.utils.pytorch_copy import ModisGlobalTilesDataset
from method.utils.utils import sample_tiles_swath, plot_tile, tile_predict_cbh, CloudBaseHeightPrediction,load_means_stds
home_dir = os.getcwd()

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from method.models.models_jinzhi import ConvAutoEncoder
from method.utils.pytorch_class import ModisGlobalTilesDataset

from torch.profiler import profile, record_function, ProfilerActivity
import torch.autograd.profiler as profiler
import pdb

def count_parameters(model):
    """
    Count the total number of trainable parameters in the model.
    :param model: The PyTorch model
    :return: Total number of trainable parameters
    """
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def main():

    # data_dir = home_dir + '/data/example/tiles/'
    # means_stds_file = home_dir+'/data/example/means_stds_save.txt'  
    # model_save_dir = home_dir + '/method/models/models_save/'



    home_dir = os.getcwd()
    data_dir = home_dir + '/data/example/tiles/'
    means_stds_file = home_dir + '/data/example/means_stds_save.txt'
    model_save_dir = home_dir + '/method/models/models_save_jinzhi/'



    learning_rate = 1e-3
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(device)
    # 加载均值和标准差
    means, stds, mins, _ = load_means_stds(means_stds_file,log_transform = False)
    param_cols = ['cloud_top_height', 'cloud_optical_thickness', 'cloud_water_path']
    subset = [0, 1, 2]  # 对应于上述参数的索引




    param_cols = ['cloud_top_height', 'cloud_optical_thickness', 'cloud_water_path']
    subset = [0, 1, 2]  # Adjust index to match the data

    # Initialize dataset
    dataset = ModisGlobalTilesDataset(
        ddir=data_dir,
        ext='nc',
        tile='', 
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

        # 调整批量大小
    batch_size = 32  # 根据GPU内存调整
    num_epochs = 500  #
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=32)


    # 初始化模型
    n_channels = 3
    input_grid_size = 128
    latent_dim = 128
    layer_channels = [32, 128, 256] 
    n_residual_blocks = 3
    model = ConvAutoEncoder(n_channels=n_channels, 
                            input_grid_size=input_grid_size, 
                            latent_dim=latent_dim, 
                            layer_channels = layer_channels, 
                            n_residual_blocks = n_residual_blocks)

    total_params = count_parameters(model)
    print(f"Total Trainable Parameters: {total_params}")

    model = model.to(device)
    
    # 定义损失函数和优化器
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    # 训练模型
    for epoch in range(num_epochs):
        
        model.train()
        running_loss = 0.0

        # 记录模型推理过程
        
        for batch in dataloader:
            
            inputs = batch['data'].to(device, dtype=torch.float)
            optimizer.zero_grad()
            _, outputs = model(inputs)
            loss = criterion(outputs, inputs)
            loss.backward()
            #print('loss: ', loss.item())
            optimizer.step()
            running_loss += loss.item() * inputs.size(0)

        # 计算当前epoch的损失
        epoch_loss = running_loss / len(dataset)
        print(f"Epoch {epoch + 1}/{num_epochs}, Loss: {epoch_loss:.4f}")

        # 每10个epoch保存模型
        if (epoch + 1) % 50 == 0:
            checkpoint_path = os.path.join(model_save_dir, f"ae_ocean_savecheckpoint_latent_mytrain{epoch + 1}.pt")
            torch.save(model.state_dict(), checkpoint_path)

        # 打印性能分析结果
        #print(prof.key_averages().table(sort_by="cpu_time_total", row_limit=10))

    # 保存最终模型
    final_model_path = os.path.join(model_save_dir, 'ae_ocean_savecheckpoint_latent_mytrain.pt')
    torch.save(model.state_dict(), final_model_path)
    print('训练完成，模型已保存至', final_model_path)














if __name__ == "__main__":
    main()







