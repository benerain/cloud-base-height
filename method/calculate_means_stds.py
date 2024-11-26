# generate_means_stds.py

import os
import glob
import xarray as xr
import numpy as np
import pandas as pd

def main():
    # 设置瓦片数据目录
    data_dir = 'data/tiles/'  # 根据你的实际数据目录修改

    # 定义要处理的云属性通道
    channels = ['cloud_top_pressure', 'cloud_top_height', 'cloud_top_temperature',
                'cloud_optical_thickness', 'cloud_water_path', 'cloud_mask']

    # 初始化列表来存储每个通道的所有像素值
    channel_values = {channel: [] for channel in channels}

    # 获取所有瓦片文件
    file_paths = glob.glob(os.path.join(data_dir, '*.nc'))

    print(f'共找到 {len(file_paths)} 个瓦片文件，开始计算统计量...')

    # 遍历每个瓦片文件，提取通道数据
    for idx, file_path in enumerate(file_paths):
        ds = xr.open_dataset(file_path)
        for channel in channels:
            data = ds[channel].values.flatten()
            # 过滤掉NaN和无效值（例如，填充值）
            data = data[~np.isnan(data)]
            # 对于云光学厚度和云水路径，过滤掉非正值（用于对数变换）
            if channel in ['cloud_optical_thickness', 'cloud_water_path']:
                data = data[data > 0]
            channel_values[channel].extend(data)
        ds.close()
        if (idx + 1) % 100 == 0:
            print(f'已处理 {idx + 1} / {len(file_paths)} 个文件')

    # 计算每个通道的统计量
    means = []
    stds = []
    mins = []
    maxs = []
    means_log = []
    stds_log = []

    for channel in channels:
        values = np.array(channel_values[channel])
        mean = np.mean(values)
        std = np.std(values)
        min_val = np.min(values)
        max_val = np.max(values)
        means.append(mean)
        stds.append(std)
        mins.append(min_val)
        maxs.append(max_val)

        # 对数变换后的均值和标准差（仅适用于云光学厚度和云水路径）
        if channel in ['cloud_optical_thickness', 'cloud_water_path']:
            log_values = np.log(values)
            mean_log = np.mean(log_values)
            std_log = np.std(log_values)
            means_log.append(mean_log)
            stds_log.append(std_log)
        else:
            means_log.append(0.0)  # 对于其他通道，填充0
            stds_log.append(0.0)

    # 创建DataFrame保存结果
    df = pd.DataFrame({
        'Channel': channels,
        'Mean': means,
        'Std': stds,
        'Min': mins,
        'Max': maxs,
        'Mean_log': means_log,
        'Std_log': stds_log
    })

    # 保存到means_stds_save.txt
    output_file = 'means_stds_save.txt'
    with open(output_file, 'w') as f:
        f.write(f'RUN time {pd.Timestamp.now()}\n')
        f.write(f'Means, Standard deviations, Minimums and Maximums for each MODIS channel extracted in {data_dir}.\n')
        df.to_csv(f, index=False, sep=' ', columns=['Channel', 'Mean', 'Std', 'Min', 'Max', 'Mean_log', 'Std_log'])
    print(f'统计量已保存至 {output_file}')

if __name__ == '__main__':
    main()
