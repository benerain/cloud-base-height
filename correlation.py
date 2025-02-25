import os
import csv
import re
import glob
import numpy as np
import xarray as xr
import pandas as pd
# import torch
# import torch.nn as nn
# from torch.utils.data import DataLoader
import time

from datetime import datetime, timedelta


from method.utils.utils import get_channels_cloud_mask, fill_in_values


from concurrent.futures import ProcessPoolExecutor, as_completed



def deg_to_index(lat_arr, lon_arr, target_lat, target_lon):
    """找到与目标经纬度最接近的像素索引"""
    dist = (lat_arr - target_lat)**2 + (lon_arr - target_lon)**2
    ij = np.unravel_index(dist.argmin(), dist.shape)
    return ij  # (i, j) 对应 (Height, Width)




def generate_filenames(start_time, hours=2):
    # 根据起始时间生成前后两小时的文件名
    filenames = []
    time_point = start_time - timedelta(hours=1)  # 开始时间前一小时
    end_time = start_time + timedelta(hours=hours-1)  # 结束时间后一小时

    while time_point <= end_time:
        day_of_year = time_point.strftime('%j')  # '001'
        hhmm = time_point.strftime('%H%M')  # '0005', '0010', ...
        filename = f"A2008.{day_of_year}.{hhmm}.nc"
        filenames.append(filename)
        time_point += timedelta(minutes=5)  # 每5分钟一个文件

    return filenames



def fill_in_values(data, fill_in=True):
	"""
	Change fill-in values in grids for channels:
	CTP - CTH - CTT - COT - CWP

	:param data: A data array from modis L2 channels.
	:param fill_in: To fill in the values with NaNs or with pre-defined fill in values.
	:returns: Filled data array and filling values.
	"""
	if fill_in:
		filling_in = np.array([1013.25, 0., 140.00999687053263187, 0., 0.], dtype=np.float64)
		# Fill cloud_top_pressure with 1atm = 1013.25hPa
		data[0, :, :] = np.where(np.isnan(data[0, :, :]), filling_in[0], data[0, :, :])
		# Fill cloud_top_height with 0m
		data[1, :, :] = np.where(np.isnan(data[1, :, :]), filling_in[1], data[1, :, :])
		# Fill cloud_top_temperature with fill_value = 140.01K by default on MODIS
		data[2, :, :] = np.where(np.isnan(data[2, :, :]), filling_in[2], data[2, :, :])
		# Fill cloud_optical_thickness with 0.
		data[3, :, :] = np.where(np.isnan(data[3, :, :]), filling_in[3], data[3, :, :])
		# Fill cloud_water_path with 0.g/m^2
		data[4, :, :] = np.where(np.isnan(data[4, :, :]), filling_in[4], data[4, :, :])
	# Else do nothing because data array already filled with NaNs here needed
	else:
		filling_in = np.zeros(shape=(5,)) * np.nan
	# Return filled grid values
	return data, filling_in


#读取特定的云属性通道数据
def read_channel(file, channel):
	"""
	Read in from the loaded file the corresponding channel.

	:param file: xarray.Dataset object.
	:param channel: Name of the channel.
	:returns: Data array of the channel.
	"""
	# Get data
	return file[channel].data[:MAX_WIDTH, :MAX_HEIGHT]


def get_channels_cloud_mask(filename): 
    """
    从二级文件中返回以下内容：
    - 通道数据：大小为 (n_channels, HEIGHT, WIDTH) 的 numpy 数组，包含所有二级通道
    - 云掩膜：大小为 (HEIGHT, WIDTH) 的 numpy 数组，表示云掩膜

    :param filename: 文件名字符串。
    :returns: 通道数据、云掩膜、纬度、经度、填充值、标签掩膜及云层底部高度（如有）。
    """
    level_data = None
    with xr.open_dataset(filename) as origin_data:
        level_data = origin_data.isel(time=0)

    if level_data is not None:
        # 纬度和经度
        latitude = read_channel(level_data, 'latitude')
        longitude = read_channel(level_data, 'longitude')

        # 提取对应通道数据
        ctp = read_channel(level_data, 'cloud_top_pressure')
        cth = read_channel(level_data, 'cloud_top_height')
        ctt = read_channel(level_data, 'cloud_top_temperature')
        cot = read_channel(level_data, 'cloud_optical_thickness')
        lwp = read_channel(level_data, 'cloud_water_path')
        # print('JJJJJJJJJJJJ',MAX_WIDTH,MAX_HEIGHT)
        cbh = level_data['cloud_layer_base'].min(axis=2).data[:MAX_HEIGHT, :MAX_WIDTH] * 1e3
        # print(cbh.shape)
        label_mask = (cbh > 0.)

        # 将通道数据堆叠为numpy数组
        channels = np.stack([ctp, cth, ctt, cot, lwp])

        # 替换填充值
        channels, filling_in = fill_in_values(channels, fill_in=True)

        # 获取云掩膜
        cloud_mask = ~np.isnan(read_channel(level_data, 'cloud_mask'))

        return channels.astype(np.float64), cloud_mask.astype(np.intp), \
               latitude.astype(np.float64), longitude.astype(np.float64), \
               filling_in.astype(np.float16), label_mask.astype(np.intp), cbh.astype(np.float64)
    else:
        # 文件无法打开时返回None
        return None, None, None, None, None, None, None




def setup_xarray_tile(tile, cbh=None, ground_cbh =None , center=None, tile_size=128):
    """创建包含所有元数据的xarray数据集"""
    ds = xr.Dataset({
        'cloud_top_pressure': (['lat', 'lon'], tile[0, :, :]),
        'cloud_top_height': (['lat', 'lon'], tile[1, :, :]),
        'cloud_top_temperature': (['lat', 'lon'], tile[2, :, :]),
        'cloud_optical_thickness': (['lat', 'lon'], tile[3, :, :]),
        'cloud_water_path': (['lat', 'lon'], tile[4, :, :]),
        'cloud_mask': (['lat', 'lon'], tile[5, :, :]),
        'cloud_base_height': (['lat', 'lon'], cbh[0]), # modis中自带的cbh
		'cloud_base_height_center': cbh[1],
        'ground_cloud_base_height': ([], ground_cbh),
        'center': (['center'], np.array(center, dtype=np.float64))
    },
    coords={
			'lat': np.arange(tile_size),
			'lon': np.arange(tile_size)
		})

    return ds


import numpy as np

MAX_WIDTH = 2030
MAX_HEIGHT = 1354

def extract_tile_by_center(swath_array, cloud_mask, latitude, longitude, center_w, center_h, fill_values, cbh, 
                           tile_size=128, cf_threshold=0.3, verbose=False):
    """
    从数据数组中提取以指定中心为中心的 tile（大小为 tile_size x tile_size）。
    并检查边界以防止超出 swath_array 范围。

    :param swath_array: 形如 (channels, width, height) 的数据数组
    :param cloud_mask: 形如 (width, height) 的云掩膜
    :param latitude: 形如 (width, height) 的纬度数据
    :param longitude: 形如 (width, height) 的经度数据
    :param center_w: 提取子区域的中心点（纵向）索引
    :param center_h: 提取子区域的中心点（横向）索引
    :param fill_values: 形如 (channels,) 的填充值数组
    :param cbh: 形如 (width, height) 的云底高度数据
    :param tile_size: 所需 tile 的大小
    :param cf_threshold: 云掩膜阈值，用于过滤云覆盖不足的区域
    :param verbose: 是否打印调试信息
    :return: (tiles, positions, centers, centers_lat_lon, cbh_values, cbh_values_center)
             或者 None，如果越界或不满足阈值条件
    """
    
    offset = tile_size // 2
    offset_2 = offset if tile_size % 2 else (offset - 1)
    
    center = (center_w, center_h)
    i1 = center_w - offset
    i2 = center_w + offset_2 + 1
    j1 = center_h - offset
    j2 = center_h + offset_2 + 1

    positions, centers, centers_lat_lon, tiles, cbh_values, cbh_values_center = [], [], [], [], [], []

    if i1 < 0 or i2 > swath_array.shape[1] or j1 < 0 or j2 > swath_array.shape[2]:
        if verbose:
            print(f"[WARN] Tile out of boundary: ({i1}:{i2}, {j1}:{j2})")
        return None

    cf = cloud_mask[i1:i2, j1:j2].sum() / (tile_size * tile_size)
    mv = (swath_array[:, i1:i2, j1:j2] == fill_values[:, np.newaxis, np.newaxis]).sum(axis=(1, 2)) / (
            tile_size * tile_size)

    if (cf >= cf_threshold) and all(mv < 1.):
        tile = swath_array[:, i1:i2, j1:j2]
        tile_position = ((i1, i2), (j1, j2))
        tile_cbh_value_center = cbh[center_w, center_h]
        tile_cbh_value = cbh[i1:i2, j1:j2]

        # Stack with cloud mask
        tile = np.concatenate([tile, cloud_mask[np.newaxis, i1:i2, j1:j2]], axis=0)

        # 返回结果
        return {
            'tile': tile,
            'position': tile_position,
            'center': (center_w, center_h),
            'center_lat_lon': (latitude[center_w, center_h], longitude[center_w, center_h]),
            'cbh_value': tile_cbh_value,
            'cbh_value_center': tile_cbh_value_center
        }
    else:
        if verbose:
            print(f"Tile at ({center_w}, {center_h}) does not meet cloud fraction or missing value threshold. Skipping.")
        return None


# 原始不删除的
# def process_observation(row, cumulo_dir,output_dir, tile_size=128, cf_threshold=0.3, verbose=True):
#     """
#     处理单个观测记录，生成相应的文件名，提取tile并返回结果。
    
#     :param row: pandas Series，单行观测数据
#     :param cumulo_dir: str，NetCDF文件存放目录
#     :param tile_size: int，tile的大小
#     :return: list of dicts，提取到的tile信息
#     """
#     # ob_time_stamp = row['OB_TIME']
#     ob_time = row['OB_TIME'].to_pydatetime()
#     lat = row['LATITUDE']
#     lon = row['LONGITUDE']
#     cbh_label = row['CLD_BASE_HT']
#     obs_id = row['id']
    
#     potential_files = generate_filenames(ob_time)

    
#     for filename in potential_files:
#         full_path = os.path.join(cumulo_dir, filename)
#         if not os.path.exists(full_path):
#             if verbose: print('    错误 - 文件 {} 不存在...'.format(filename))
#             continue

#         # print('xxxxxxxx',filename)
#         swath_array, cloud_mask,latitude, longitude, fill_values, label_mask, cbh = get_channels_cloud_mask(
# filename=full_path)

#         if verbose: print('    提取扫描带 {} 的瓦片...'.format(filename))

        
#         if swath_array is None:
#             if verbose: print('    错误 - 文件 {} 无法打开...'.format(filename))
#             continue


#         center_h, center_w = deg_to_index(latitude, longitude, lat, lon)

#         # print(type(cbh),cbh.shape,cbh)
#         res = extract_tile_by_center(
#             swath_array=swath_array,
#             cloud_mask=cloud_mask,
#             latitude=latitude,
#             longitude=longitude,
#             center_w = center_w,
#             center_h = center_h,
#             fill_values=fill_values,
#             cbh=cbh,
#             tile_size=tile_size,
#             cf_threshold=cf_threshold,
#             verbose=verbose)
        
#         if res is None:
#             # 不满足阈值或越界了，直接跳过
#             continue


            
#         # 如果不是 None，就从字典里取需要的东西
#         tile = res['tile']
#         positions = res['position']
#         centers = res['center']
#         centers_lat_lon = res['center_lat_lon']
#         cbh_values = res['cbh_value']
#         cbh_values_center = res['cbh_value_center']

#         if tile is not None:
#             if verbose: print('    将瓦片保存到输出目录...')
#             base_filename = os.path.splitext(filename)[0]

#             setup_xarray_tile(
# 			tile=tile, cbh=[cbh_values, cbh_values_center] if cbh is not None else None, ground_cbh=cbh_label ,
# 			 center=(lat, lon), tile_size=128).to_netcdf(os.path.join(output_dir , "obs_{}_{}_{}_{}.nc".format(obs_id,base_filename,center_w,center_h )))

#     # return local_results


def process_observation(row, cumulo_dir, output_dir, tile_size=128, cf_threshold=0.3, verbose=True):
    """
    处理单条观测数据，如果成功生成文件则返回 True，否则 False。
    """
    ob_time = row['OB_TIME'].to_pydatetime()
    lat = row['LATITUDE']
    lon = row['LONGITUDE']
    cbh_label = row['CLD_BASE_HT']
    obs_id = row['id']

    try:
        potential_files = generate_filenames(ob_time)

        found_tile = False  # 用于指示是否找到并生成瓦片
        for filename in potential_files:
            full_path = os.path.join(cumulo_dir, filename)
            if not os.path.exists(full_path):
                if verbose:
                    print(f"    错误 - 文件 {filename} 不存在...")
                continue

            swath_array, cloud_mask, latitude_arr, longitude_arr, fill_values, label_mask, cbh = \
                get_channels_cloud_mask(full_path)

            if verbose:
                print(f"    提取扫描带 {filename} 的瓦片...")

            if swath_array is None:
                if verbose:
                    print(f"    错误 - 文件 {filename} 无法打开...")
                continue

            # 找到该观测点在该 swath 上的像素索引
            center_h, center_w = deg_to_index(latitude_arr, longitude_arr, lat, lon)

            res = extract_tile_by_center(
                swath_array=swath_array,
                cloud_mask=cloud_mask,
                latitude=latitude_arr,
                longitude=longitude_arr,
                center_w=center_w,
                center_h=center_h,
                fill_values=fill_values,
                cbh=cbh,
                tile_size=tile_size,
                cf_threshold=cf_threshold,
                verbose=verbose
            )

            if res is None:
                # 不满足阈值或越界，跳过
                continue

            tile = res['tile']
            cbh_values = res['cbh_value']
            cbh_values_center = res['cbh_value_center']

            if tile is not None:
                if verbose:
                    print(f"    将瓦片保存到输出目录: obs_{obs_id}_{filename}_{center_w}_{center_h}.nc")
                base_filename = os.path.splitext(filename)[0]
                ds = setup_xarray_tile(
                    tile=tile,
                    cbh=[cbh_values, cbh_values_center] if cbh is not None else None,
                    ground_cbh=cbh_label,
                    center=(lat, lon),
                    tile_size=tile_size
                )
                out_file = os.path.join(output_dir, f"obs_{obs_id}_{base_filename}_{center_w}_{center_h}.nc")
                ds.to_netcdf(out_file)
                found_tile = True

        return found_tile  # 如果至少生成了一个瓦片就返回 True

    except Exception as e:
        if verbose:
            print(f"处理 {obs_id} 时出现异常: {e}")
        return False



def append_row_to_csv(filename, row_dict):
    """将 row_dict 追加写入 CSV 文件，如果文件不存在则先写入表头。"""
    file_exists = os.path.exists(filename)
    with open(filename, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(row_dict.keys()))
        if not file_exists:
            writer.writeheader()
        writer.writerow(row_dict)
        f.flush()


# ========== 并行主程序入口 ==========

if __name__ == '__main__':
    csv_path = "/home/jinzhi/hdd/jinzhi/lyj/Cloud_base_height_Method_Lenhardt2024_v2/data/plots_data/synop_cbh_colocation_count_shuffle.csv"
    cumulo_dir = "/home/jinzhi/hdd/jinzhi/lyj/YS-WD2/CUMULO"
    output_dir = "/home/jinzhi/hdd/jinzhi/lyj/CUMULO_data/output"

    # 读取 CSV
    df = pd.read_csv(csv_path, parse_dates=['OB_TIME'])


    # # 1) 打乱csv顺序
    # df = df.sample(frac=1, random_state=42).reset_index(drop=True)

    # 确保输出目录存在
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    max_workers = 48

    start_time = time.time()

    # 文件名设置
    success_csv = "/home/jinzhi/hdd/jinzhi/lyj/Cloud_base_height_Method_Lenhardt2024_v2/data/plots_data/success.csv"
    fail_csv = "/home/jinzhi/hdd/jinzhi/lyj/Cloud_base_height_Method_Lenhardt2024_v2/data/plots_data/fail.csv"


    # 使用 ProcessPoolExecutor 并行处理，并把 row 以 dict 存入 futures 映射中
    futures = {}
    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        for idx, row in df.iterrows():
            row_dict = row.to_dict()
            future = executor.submit(
                process_observation,
                row_dict,  # 把行转换为 dict
                cumulo_dir,
                output_dir,
                128,    # tile_size
                0.3,    # cf_threshold
                True    # verbose
            )
            futures[future] = row_dict  # 存储整个字典

        # 逐个处理完成的 future
        for fut in as_completed(futures):
            row_dict = futures[fut]
            try:
                result = fut.result()  # 返回 True 或 False
                if result:
                    # 成功立即写入 success.csv
                    append_row_to_csv(success_csv, row_dict)
                    # print(f"记录 id={row_dict['id']} 成功处理并写入 {success_csv}")
                else:
                    append_row_to_csv(fail_csv, row_dict)
                    # print(f"记录 id={row_dict['id']} 处理失败，写入 {fail_csv}")
            except Exception as e:
                append_row_to_csv(fail_csv, row_dict)
                # print(f"记录 id={row_dict['id']} 异常，写入 {fail_csv}，error: {e}")

    end_time = time.time()
    elapsed = end_time - start_time
    print(f"\n代码执行时间：{elapsed:.4f} 秒")
    print("并行处理完成！")

# if __name__ == '__main__':
#     # 测试

#     csv_path = "/home/jinzhi/hdd/jinzhi/lyj/Cloud_base_height_Method_Lenhardt2024_v2/data/plots_data/synop_cbh_colocation_count copy.csv"
#     cumulo_dir = "/home/jinzhi/hdd/jinzhi/lyj/YS-WD2/CUMULO"  # A2008.001.0000.nc 等都在这里
#     output_dir = "/home/jinzhi/hdd/jinzhi/lyj/CUMULO_data/output"


#     df = pd.read_csv(csv_path, parse_dates=['OB_TIME'])
#     df = df.head(100)

#     if not os.path.exists(output_dir):
#         os.makedirs(output_dir)


#     max_workers = 16  # 根据自己的 CPU 核数和 I/O 情况来决定

#     import time
#     start_time = time.time()  # 记录开始时间

#     with ProcessPoolExecutor(max_workers=max_workers) as executor:
#         # 提交所有观测记录的处理任务
#         futures = []
#         for idx, row in df.iterrows():
#             # 提交一个任务: 调用 process_observation
#             # 注意: 需要把 row 转换成可直接被子进程使用的对象 (row 是 pandas.Series)
#             # 一般直接传 row 即可；如果出现 pickling 问题，可以转成 dict
#             futures.append(
#                 executor.submit(
#                     process_observation,
#                     row, 
#                     cumulo_dir, 
#                     output_dir, 
#                     128,      # tile_size
#                     0.3,      # cf_threshold
#                     True      # verbose
#                 )
#             )

#     end_time = time.time()  # 记录结束时间
#     elapsed = end_time - start_time
#     print(f"代码执行时间：{elapsed:.4f} 秒")
#     print("并行处理完成！")
        


#     # for idx, row in df.iterrows():
#     #     process_observation(row, cumulo_dir, output_dir, verbose=True)