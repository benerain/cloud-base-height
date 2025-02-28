import xarray as xr
import os
import glob
import pandas as pd

import numpy as np
np.int = int

import itertools
import matplotlib.pyplot as plt
from method.utils.FYutils import FYGlobalTilesDataset

import torch
import torch.nn as nn
from torch.utils.data import Dataset
from torch.utils.data import DataLoader


from method.models.models import ConvAutoEncoder
from joblib import load



MIN_N_TILES = 3000
MAX_WIDTH, MAX_HEIGHT = 1250,1750

def get_tile_offsets(tile_size=128):
	"""
	Returns index of center depending on the tile size.
	In the case of an even tile size, the center is defined by (tile_size//2, tile_size//2-1).

	:param tile_size: Square size of the desired tile.
	:returns: Offset sizes.
	"""
	offset = tile_size // 2
	offset_2 = offset
	if not tile_size % 2:
		offset_2 -= 1
	return offset, offset_2


def get_sampling_mask(mask_shape=(MAX_WIDTH, MAX_HEIGHT), tile_size=128):
	"""
	Returns a mask of allowed centers for the tiles to be sampled.
	The center of an even size tile is considered to be the point
	at the position (size // 2, size // 2 - 1) within the tile.

	:param mask_shape: Size of the input file ie mask size.
	:param tile_size: Size of the desired tiles.
	:returns: mask
	"""
	mask = np.ones(mask_shape, dtype=np.uint8)
	offset, offset_2 = get_tile_offsets(tile_size)
	# must not sample tile centers in the borders, so that tiles keep to required shape
	mask[:, :offset] = 0
	mask[:, -offset_2:] = 0
	mask[:offset, :] = 0
	mask[-offset_2:, :] = 0
	return mask



def fill_in_values(data, fill_in=True):
	"""
	Change fill-in values in grids for channels:
	 CTH - COT - CWP

	:param data: A data array from modis L2 channels.
	:param fill_in: To fill in the values with NaNs or with pre-defined fill in values.
	:returns: Filled data array and filling values.
	"""
	if fill_in:
		filling_in = np.array([ 0., 0., 0.], dtype=np.float64)
        # Fill cloud_top_height with 0m
		data[0, :, :] = np.where(np.isnan(data[0, :, :]), filling_in[0], data[0, :, :])
		# Fill cloud_optical_thickness with 0.
		data[1, :, :] = np.where(np.isnan(data[1, :, :]), filling_in[0], data[1, :, :])
		# Fill cloud_water_path with 0.g/m^2
		data[2, :, :] = np.where(np.isnan(data[2, :, :]), filling_in[0], data[2, :, :])
	# Else do nothing because data array already filled with NaNs here needed
	else:
		filling_in = np.zeros(shape=(3,)) * np.nan
	# Return filled grid values
	return data, filling_in

def read_channel(file, channel):
	"""
	Read in from the loaded file the corresponding channel.

	:param file: xarray.Dataset object.
	:param channel: Name of the channel.
	:returns: Data array of the channel.
	"""
	# Get data
	return file[channel].data


###预定义填充值的选择:
# filling_in 数组定义了每个通道应该使用的填充值。这些值通常基于科学或技术的标准或常识。例如：
	# •	1013.25 hPa 代表平均海平面气压，可能用于填充缺失的云顶压力数据。
	# •	0 可能用于云高和云光学厚度，表明缺失或最低值。
	# •	140.01K 代表一个非常低的温度，可能用于云顶温度，表明极端条件或数据缺失。
# 填充缺失值或无效值。
def fill_in_values(data, fill_in=True):
	"""
	Change fill-in values in grids for channels:
	 CTH - COT - CWP

	:param data: A data array from modis L2 channels.
	:param fill_in: To fill in the values with NaNs or with pre-defined fill in values.
	:returns: Filled data array and filling values.
	"""
	if fill_in:
		filling_in = np.array([ 0., 0., 0.], dtype=np.float64)
        # Fill cloud_top_height with 0m
		data[0, :, :] = np.where(np.isnan(data[0, :, :]), filling_in[0], data[0, :, :])
		# Fill cloud_optical_thickness with 0.
		data[1, :, :] = np.where(np.isnan(data[1, :, :]), filling_in[0], data[1, :, :])
		# Fill cloud_water_path with 0.g/m^2
		data[2, :, :] = np.where(np.isnan(data[2, :, :]), filling_in[0], data[2, :, :])
	# Else do nothing because data array already filled with NaNs here needed
	else:
		filling_in = np.zeros(shape=(3,)) * np.nan
	# Return filled grid values
	return data, filling_in



def get_channels_cloud_mask(filename): 
    """
    从二级文件中返回以下内容：
    - 通道数据：大小为 (n_channels, HEIGHT, WIDTH) 的 numpy 数组，包含所有二级通道
    - 云掩膜：大小为 (HEIGHT, WIDTH) 的 numpy 数组，表示云掩膜

    :param filename: 文件名字符串。
    :returns: 通道数据、云掩膜、纬度、经度、填充值、标签掩膜及云层底部高度（如有）。
    """
    level_data = None
    try:
        # 打开文件
        level_data = xr.open_dataset(filename)
    except IndexError as err:
        print('    无法打开文件: {}'.format(err))

    if level_data is not None:
        # 纬度和经度
        latitude = read_channel(level_data, 'lat')
        longitude = read_channel(level_data, 'lon')

        # 提取对应通道数据

        cth = read_channel(level_data, 'CTH')
        cot = read_channel(level_data, 'COT')
        lwp = read_channel(level_data, 'LWP')
        cbh = level_data['cloud_base'].data
        label_mask = (cbh > 0.)

        # 将通道数据堆叠为numpy数组
        channels = np.stack([ cth, cot, lwp])

        # 替换填充值
        channels, filling_in = fill_in_values(channels, fill_in=True)

        # 获取云掩膜
        # cloud_mask = ~np.isnan(read_channel(level_data, 'CLM'))
		
        # 获取云掩膜
        # cloud_mask = ~np.isnan(read_channel(level_data, 'CLM')) # 这是modis的
        # clm_data = read_channel(level_data, 'CLM')
        raw_clm = read_channel(level_data, 'CLM')
        # cloud_mask = (clm_data == 1.0) 
        cloud_mask = np.logical_and(~np.isnan(raw_clm), raw_clm == 1.0)

        return channels.astype(np.float64), cloud_mask.astype(np.intp), \
               latitude.astype(np.float64), longitude.astype(np.float64), \
               filling_in.astype(np.float16), label_mask.astype(np.intp), cbh.astype(np.float64)
    else:
        # 文件无法打开时返回None
        return None, None, None, None, None, None, None 
	

def setup_xarray_tile(tile, cbh=None, center=None, tile_size=128):
    """
    Setup the tile as an xarray.Dataset object.
    """
    tile_xr = xr.Dataset({    
        'CTH': (['lat', 'lon'], tile[0, :, :]),
        'COT': (['lat', 'lon'], tile[1, :, :]),
        'LWP': (['lat', 'lon'], tile[2, :, :]),
        'CLM': (['lat', 'lon'], tile[3, :, :]),
       
        # 如果 cbh 数据可用，则加入相应数据
        'cloud_base_height': (['lat', 'lon'], cbh[0]) if cbh is not None else None,
        'cloud_base_height_center': cbh[1] if cbh is not None else None,
        'center': center
    },
    coords={
        'lat': np.arange(tile_size),
        'lon': np.arange(tile_size)
    })
    return tile_xr



def save_tiles_nc(swath_name, tiles, ddir, center, cbh=None):
	"""
	Save tiles to destination directory as xarray.Dataset/netCDF objects.

	:param swath_name: Swath id name.
	:param tiles: List containing the extracted tiles.
	:param ddir: Destination directory.
	:returns: None
	"""
	if not os.path.exists(ddir):
		os.makedirs(ddir)

	# Remove existing tiles
	list_tiles = glob.glob(ddir + '*.nc')
	for t in list_tiles:
		os.remove(t)

	for i, tile in enumerate(tiles, 1):
		setup_xarray_tile(
			tile=tile, cbh=[cbh[0][i - 1], cbh[1][i - 1]] if cbh is not None else None,
			center=center[i - 1]).to_netcdf(ddir + "{}_{}.nc".format(swath_name, i))
		


def extract_cloudy_tiles_swath(
		swath_array, cloud_mask, latitude, longitude,
		fill_values, cbh, regular_sampling=False, sampling_step='wide',
		n_tiles=20, tile_size=128, cf_threshold=0.3, verbose=False):
	"""
	The script will use a cloud_mask channel to mask away all non-cloudy data.
	The script will then select all tiles from the cloudy areas where the cloud fraction is at least cf_threshold.

	:param swath_array: input numpy array from MODIS of size (nb_channels, w, h)
	:param cloud_mask: 2d array of size (w, h) marking the cloudy pixels
	:param latitude: Latitude array.
	:param longitude: Longitude array.
	:param fill_values: Values to fill in the missing values of the cloud properties.
	:param cbh: input array of the cbh retrieval (dimension (w, h)) if available.
	:param regular_sampling: Tiles to be sampled regularly (according to some step size) or randomly.
	:param sampling_step: Which step size to use when sampling regularly the tiles. 'wide' = 128 km, 'regular' = 64 km
	and 'fine' = 10 km. Values can be adapted if necessary.
	:param n_tiles: Number of tiles to sample from the swath.
	:param tile_size: size of the tile selected from within the image
	:param cf_threshold: cloud fraction threshold to apply when filtering cloud scenes.
	:param verbose: Display information or not.
	:return: a 4-d array (nb_tiles, nb_channels, w, h) of sampled tiles and corresponding cbh label (nb_tiles,)
	"""
	# Compute distances from tile center of tile upper left and lower right corners
	offset, offset_2 = get_tile_offsets(tile_size)

	if regular_sampling:

		tile_centers = []
		# Sampling centers
		step_size = tile_size // 2 if sampling_step == 'wide' else (64 if sampling_step == 'regular' else 10)
		idx_w = np.arange(start=2 * tile_size, stop=cloud_mask.shape[0] - 2 * tile_size, step=step_size)
		idx_h = np.arange(start=2 * tile_size, stop=cloud_mask.shape[1] - 2 * tile_size, step=step_size)
		for c_w, c_h in itertools.product(idx_w, idx_h):
			tile_centers.append([c_w, c_h])
			# tile_centers.append([latitude[c_w, c_h], longitude[c_w, c_h]])
		tile_centers = np.array(tile_centers)

	else:

		# Mask out borders not to sample outside the swath
		allowed_pixels = get_sampling_mask(swath_array.shape[1:], tile_size)

		# Tile centers will be sampled from the cloudy pixels that are not in the borders of the swath
		cloudy_label_pixels = np.logical_and.reduce([allowed_pixels.astype(bool), cloud_mask.astype(bool)])
		cloudy_label_pixels_idx = np.where(cloudy_label_pixels == 1)
		cloudy_label_pixels_idx = list(zip(*cloudy_label_pixels_idx))

		# Number of tiles to sample from
		number_of_tiles = min(MIN_N_TILES, len(cloudy_label_pixels_idx)) # MIN_N_TILES  控制采样数量
		# Sample without replacement
		tile_centers_idx = np.random.choice(np.arange(len(cloudy_label_pixels_idx)), number_of_tiles, False)
		cloudy_pixels_idx = np.array(cloudy_label_pixels_idx)
		tile_centers = cloudy_pixels_idx[tile_centers_idx]

	positions, centers, centers_lat_lon, tiles, cbh_values, cbh_values_center = [], [], [], [], [], []

	for center in tile_centers:   # 以中心点为中心，提取大小为 tile_size 的瓦片，并检查其云覆盖度和缺失值比例。

		center_w, center_h = center

		w1 = center_w - offset
		w2 = center_w + offset_2 + 1
		h1 = center_h - offset
		h2 = center_h + offset_2 + 1

		# Check cloud fraction in tile
		cf = cloud_mask[w1:w2, h1:h2].sum() / (tile_size * tile_size)

		# Check missing values in tile
		mv = (swath_array[:, w1:w2, h1:h2] == fill_values[:, np.newaxis, np.newaxis]).sum(axis=(1, 2)) / (
				tile_size * tile_size)

		# If cloud fraction in the tile is higher than cf_threshold then store it
		if (cf >= cf_threshold) and all(mv < 1.):
			tile = swath_array[:, w1:w2, h1:h2]
			tile_position = ((w1, w2), (h1, h2))
			tile_cbh_value_center = cbh[center_w, center_h]
			tile_cbh_value = cbh[w1:w2, h1:h2]

			# Stack with cloud mask
			tile = np.concatenate([tile, cloud_mask[np.newaxis, w1:w2, h1:h2]], axis=0)

			positions.append(tile_position)
			centers.append(center)
			centers_lat_lon.append((latitude[center_w, center_h], longitude[center_w, center_h]))
			tiles.append(tile)
			cbh_values.append(tile_cbh_value)
			cbh_values_center.append(tile_cbh_value_center)

	if len(tiles) > 0:
		n_tiles = len(tiles) if regular_sampling else min(n_tiles, len(tiles))
		print('    {} extracted tiles'.format(n_tiles)) if verbose else None
		tiles = np.stack(tiles[:n_tiles])
		positions = np.stack(positions[:n_tiles])
		centers = np.stack(centers[:n_tiles])
		centers_lat_lon = np.stack(centers_lat_lon[:n_tiles])
		cbh_values = np.stack(cbh_values[:n_tiles])
		cbh_values_center = np.stack(cbh_values_center[:n_tiles])

		return tiles, positions, centers, centers_lat_lon, cbh_values, cbh_values_center #extract_cloudy_tiles_swath 返回了云底高

	else:
		print('    No valid tiles could be extracted from the swath.') if verbose else None
		return None, None, None, None, None, None



def sample_tiles_swath(filename, dest_dir, regular_sampling=False, sampling_step='wide', n_tiles=20, tile_size=128, cf_threshold=0.3, verbose=True):

    """
    从扫描带中创建瓦片。
    将瓦片保存到相应文件夹。

    :param filename: 扫描带文件名。
    :param dest_dir: 保存提取瓦片的目标目录。
    :param regular_sampling: 瓦片是否规则采样（根据步长）或随机采样。
    :param sampling_step: 规则采样时的步长。'wide' = 128公里，'regular' = 64公里，'fine' = 10公里。
    :param n_tiles: 从扫描带中采样的瓦片数量。
    :param tile_size: 从图像中选取的瓦片大小。
    :param cf_threshold: 过滤云场景时的云分数阈值。
    :param verbose: 是否显示信息。
    :returns: 提取瓦片中心的数组（纬度，经度）。
    """

    # 扫描带名称
    swath_name = filename.split('/')[-1][:-3]
    # print(swath_name)
    if verbose: print('扫描带文件 {}'.format(swath_name))

    # 从MODIS文件提取通道数据和云掩膜
    if verbose: print('    提取通道和云掩膜数据...')
    swath_array, cloud_mask, latitude, longitude, fill_values, label_mask, cbh = get_channels_cloud_mask(
        filename=filename)

    # 如果文件无法打开
    if swath_array is None:
        if verbose: print('    错误 - 文件 {} 无法打开...'.format(swath_name))
        return None

    # 从扫描带数据中提取云瓦片
    if verbose: print('    提取扫描带 {} 的瓦片...'.format(swath_name))

    # 添加的权宜之计
    latitude = np.transpose(latitude)  # 变为 (1250, 1750)
    longitude = np.transpose(longitude)

    print(latitude.shape, longitude.shape)
    
    tiles, positions, centers, centers_lat_lon, cbh_values, cbh_values_center = extract_cloudy_tiles_swath(
        swath_array=swath_array,
        cloud_mask=cloud_mask,
        latitude=latitude,
        longitude=longitude,
        fill_values=fill_values,
        cbh=cbh,
        regular_sampling=regular_sampling,
        sampling_step=sampling_step,
        n_tiles=n_tiles,
        tile_size=tile_size,
        cf_threshold=cf_threshold,
        verbose=verbose)

    # 保存瓦片
    if tiles is not None:
        if verbose: print('    将瓦片保存到输出目录...')
        print(tiles.shape, centers_lat_lon.shape, cbh_values.shape, cbh_values_center.shape)
        save_tiles_nc(
            swath_name=swath_name, tiles=tiles, center=centers_lat_lon,
            cbh=[cbh_values, cbh_values_center], ddir=dest_dir)

    return centers_lat_lon




def plot_tile(tile):
	"""
	Plot an extracted tile.

	:param tile: Tile filename to plot.
	:returns: None
	"""
	tile = xr.open_dataset(tile)

	fig, axs = plt.subplots(ncols=3, nrows=1, figsize=(20, 8))

	p = axs[0].imshow(tile.CTH*1000, aspect='equal', vmin=0., cmap='Blues_r')
	cbar = plt.colorbar(p, ax=axs[0], orientation='horizontal', extend='max')
	cbar.set_label('Cloud top height ($m$)')
	axs[0].set_title('Cloud top height ($m$)', weight='bold')
	axs[0].axis('off')

	p = axs[1].imshow(tile.COT, aspect='equal', vmin=0., vmax=20., cmap='Blues_r')
	cbar = plt.colorbar(p, ax=axs[1], orientation='horizontal', extend='max')
	cbar.set_label('Cloud optical thickness (a.u.)')
	axs[1].set_title('Cloud optical thickness (a.u.)', weight='bold')
	axs[1].axis('off')

	p = axs[2].imshow(tile.LWP, aspect='equal', vmin=0., vmax=1000., cmap='Blues_r')
	cbar = plt.colorbar(p, ax=axs[2], orientation='horizontal', extend='max')
	cbar.set_label('Cloud water path ($g.m^{-2}$)')
	axs[2].set_title('Cloud water path ($g.m^{-2}$)', weight='bold')
	axs[2].axis('off')

	plt.show()

	tile.close()

# endregion

# region Encoding and prediction


def load_means_stds(file, log_transform, subset=[1, 3, 4]):
	"""
	Load means and standard deviations of the different cloud properties.
	:param file: File in which the quantities are stored.
	:param log_transform: Boolean to indicate if some cloud properties are to be log-transformed (only COT or CWP).
	:param subset: Subset of cloud properties to load. Channels are stored in order: CTP, CTH, CTT, COT, CWP.
	:returns: numpy.arrays of means and standard deviations.
	"""
	df_means_stds = pd.read_csv(
		file, header=0, skiprows=2, sep=' ', usecols=[0, 1, 2, 3, 4, 5])
	Means = df_means_stds.iloc[0].values
	Stds = df_means_stds.iloc[1].values
	Mins = df_means_stds.iloc[2].values
	Maxs = df_means_stds.iloc[3].values
	Means_log = df_means_stds.iloc[4].values
	Stds_log = df_means_stds.iloc[5].values
	if log_transform:
		# Replace mean/std by the mean/std of logs for COT and CWP
		Means[3:5] = Means_log[3:5]
		Stds[3:5] = Stds_log[3:5]
	# Select subset channels
	means = Means[subset]
	stds = Stds[subset]
	mins = Mins[subset]
	maxs = Maxs[subset]
	return means, stds, mins, maxs



# 带有ground truth的数据集	
def load_data_tiles(ddir, tile_names, means_stds_file, get_ground_cbh=False):
    """
    Load pytorch.Dataset object for the tiles.

    :param ddir: Data directory of the tile files.
    :param tile_names: Regex for tiles to use from the data directory.
    :param means_stds_file: File containing means and standard deviations of the different cloud properties.
    :param get_ground_cbh: Boolean. If True, will load ground-based CBH from netCDF.
    :returns: (dataset, dataloader)
    """

	
    param_cols = ['CTH', 'COT', 'LWP']
    subset = [1, 3, 4]

    # Additional parameters for the pytorch.Dataset class FYGlobalTilesDataset
    ext = 'nc'
    subscale = False
    grid_size = 128
    log_transform = False
    normalize = True

    means, stds, mins, _ = load_means_stds(file=means_stds_file, log_transform=log_transform)

    # dataset = FYGlobalTilesDataset(
    #     ddir=ddir, ext=ext, tile=tile_names, subset=subset, subset_cols=param_cols,
    #     transform=None, subscale=subscale, grid_size=grid_size,
    #     normalize=normalize, mean=means, std=stds, min=mins, log_transform=log_transform
    # )
    dataset = FYGlobalTilesDataset(
        ddir=ddir, 
        ext=ext, 
        tile=tile_names, 
        subset=subset, 
        subset_cols=param_cols,
        transform=None, 
        subscale=subscale, 
        grid_size=grid_size,
        normalize=normalize, 
        mean=means, 
        std=stds, 
        min=mins, 
        log_transform=log_transform,
        get_ground_cbh=get_ground_cbh   # <== 这里!
    )

    batch_size = 64 if len(dataset) >= 64 else len(dataset)
    
    dataloader = DataLoader(dataset, batch_size=batch_size,
                            shuffle=False, drop_last=False, num_workers=1)

    return dataset, dataloader


def load_ae_model(ae_model, device, grid_size=128, subscale=False):
	"""
	Load convolutional auto-encoder model.

	:param ae_model: Model file.
	:param device: Device to load the model on.
	:param grid_size: Dimension of the tile.
	:param subscale: If size of the tiles is reduced.
	:returns: Initialized model object ConvVarAutoEncoder.
	"""
	# Model parameters
	in_channels = 3
	input_grid_size = grid_size if subscale else 128
	latent_dim = 256
	# Initialize model
	model = ConvAutoEncoder(n_channels=in_channels, input_grid_size=input_grid_size, latent_dim=latent_dim)
	model = nn.DataParallel(model)
	# Load model weights
	model.load_state_dict(torch.load(ae_model, map_location=device))
	# Send the model to the device
	model = model.to(device)
	model = model.double()
	# Set nn.BatchNorm2d running stats to None
	for m in model.modules():
		for child in m.children():
			if type(child) == torch.nn.BatchNorm2d:
				child.track_running_stats = False
				child.running_mean = None
				child.running_var = None
	# Put model in evaluation mode, no gradient computations necessary
	model.train(False)
	return model

def encoding_tiles(ddir, tile_names, ae_model, means_stds_file):
    """
    Encode tiles with the convolutional auto-encoder model.

    :param ddir: Data directory of the tile files.
    :param tile_names: Regex for tiles to use from the data directory.
    :param ae_model: File for the convolutional auto-encoder.
    :param means_stds_file: File containing means and standard deviations of the different cloud properties.
    :returns: numpy.array of the encoded tiles and of corresponding (latitude, longitude) centers.
    """
    # 1) 加载 tiles 数据
    dataset, dataloader = load_data_tiles(
        ddir=ddir,
        tile_names=tile_names,
        means_stds_file=means_stds_file,
        get_ground_cbh=True  # 获取地基雷达 CBH
    )

    # 2) 定义 device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 3) 加载 Autoencoder 模型
    model = load_ae_model(ae_model=ae_model, device=device)

    # 4) 准备数组来保存编码向量、瓦片中心和真值 CBH
    n_samples = len(dataset)
    latent_dim = model.module.latent_dim  # 确保 model.module 存在（如果是 nn.DataParallel）

    encodings = np.empty(shape=(n_samples, latent_dim), dtype=np.float64)
    centers = np.empty(shape=(n_samples, 2), dtype=np.float64)
    ground_cbh_arr = np.empty(shape=(n_samples,), dtype=np.float64)

    idx_start = 0

    with torch.no_grad():
        for batch_data in dataloader:
            inputs = batch_data['data'].to(device)  # 瓦片数据
            b_size = inputs.shape[0]

            # 5) AE 编码
            latent = model.module.encode(inputs)  # shape: (batch_size, latent_dim)
            latent_np = latent.cpu().numpy()

            # 6) 保存 encodings
            encodings[idx_start:idx_start + b_size, :] = latent_np

            # 7) 保存中心坐标
            centers_np = batch_data['center'].cpu().numpy()
            centers[idx_start:idx_start + b_size, :] = centers_np

            # 8) 保存地基雷达 CBH (点值)
            if 'ground_cbh_center' in batch_data:
                ground_cbh_np = batch_data['ground_cbh_center'].cpu().numpy()
                ground_cbh_arr[idx_start:idx_start + b_size] = ground_cbh_np
            else:
                ground_cbh_arr[idx_start:idx_start + b_size] = np.nan  # 处理缺失值

            idx_start += b_size

    return encodings, centers, ground_cbh_arr  # 返回所有 CBH 数据



def predicting_cbh_or(encodings, or_model):
	"""
	Predict cloud base height from the tile encodings.

	:param encodings: numpy.array of the tile encodings.
	:param or_model: File for the ordinal regression prediction model.
	:returns: numpy.array of cloud base height predictions.
	"""
	classes = np.array([50., 100., 200., 300., 600., 1000., 1500., 2000., 2500.])
	# Load OR model
	model = load(or_model)
	# Predictions
	preds = model.predict(encodings)
	# Convert classes to cloud base height values
	preds = np.array([classes[p] for p in preds])
	return preds




def tile_predict_cbh(ddir, tile_names, ae_model, or_model, means_stds_file, verbose):
    """
    1. 利用 AE 对瓦片进行编码
    2. 利用 OR 模型（回归）预测 CBH
    3. 对比 ground_cbh_center（真值）和 preds（预测值），计算评估指标
    4. 返回预测值、中心坐标和 ground truth
    """
    print('    Running encoding of sampled tiles...') if verbose else None
    # 新的 encoding_tiles 会返回 3 个值
    encodings, centers, ground_cbh_arr = encoding_tiles(
        ddir=ddir, tile_names=tile_names, 
        ae_model=ae_model, means_stds_file=means_stds_file
    )

    print('    Predicting cloud base height...') if verbose else None
    preds = predicting_cbh_or(encodings=encodings, or_model=or_model)

    # =============== 计算模型误差 ===============
    # 先判断：ground_cbh_arr 是否有效 (如果读取失败可能是 -1) # -1 是在 dataset 中定义的
    mask_valid = ground_cbh_arr != -1
    valid_preds = preds[mask_valid]
    valid_gt = ground_cbh_arr[mask_valid]

    if len(valid_gt) > 0:
        # 计算 RMSE
        rmse = np.sqrt(np.mean((valid_preds - valid_gt)**2))
        # 计算 MAE
        mae = np.mean(np.abs(valid_preds - valid_gt))
        # 计算相关系数 (Pearson)
        corr = np.corrcoef(valid_preds, valid_gt)[0, 1]

        print(f'    Evaluation metrics (on {len(valid_gt)} samples):')
        print(f'    RMSE = {rmse:.2f},  MAE = {mae:.2f},  Corr = {corr:.2f}')
    else:
        print('    No valid ground truth CBH found for comparison.')

    return preds, centers, ground_cbh_arr



class FYCloudBaseHeightPrediction:
	"""
	Method wrapper for the cloud base height prediction from cloud properties.

	Method presented/detailed in:
	"Marine cloud base height retrieval from MODIS cloud properties using supervised machine learning"
	J. Lenhardt, J. Quaas, D. Sejdinovic

	__init__
		Initialize the class object.

	sample_tiles
		Sample tiles from the given swath files.

	predict_cbh
		Predict cloud base height from the given tiles.

	run_cbh
		Sample tiles from the given swath files and then predicts
		the cloud base height for the corresponding extracted tiles.

	"""

	def __init__(self, par):
		self.filename = par['filename']
		self.tile_names = self.filename.split('/')[-1][:-3]
		self.ddir = par['ddir']

		self.regular_sampling = par['regular_sampling']
		self.sampling_step = par['sampling_step']
		self.n_tiles = par['n_tiles']
		self.tile_size = par['tile_size']
		self.cf_threshold = par['cf_threshold']

		self.ae_model = par['ae_model']
		self.or_model = par['or_model']
		self.means_stds_file = par['means_stds_file']

		self.verbose = par['verbose']

		self.centers = []
		self.preds = []

	def sample_tiles(self):
		self.centers = sample_tiles_swath(
			filename=self.filename, dest_dir=self.ddir,
			regular_sampling=self.regular_sampling, sampling_step=self.sampling_step,
			n_tiles=self.n_tiles, tile_size=self.tile_size,
			cf_threshold=self.cf_threshold, verbose=self.verbose)

	def predict_cbh(self):
		self.preds, self.centers, self.ground_cbh_arr  = tile_predict_cbh(
			ddir=self.ddir, tile_names=self.tile_names,
			ae_model=self.ae_model, or_model=self.or_model,
			means_stds_file=self.means_stds_file,
			verbose=self.verbose)

	def run_cbh(self):
		"""
		Run the whole prediction pipeline.

		:returns: numpy.array predictions of the cloud base height, numpy.array of the tiles centers.
		"""
		print('Running cloud base height prediction pipeline...') if self.verbose else None
		print('\n-> Sampling tiles from swath files') if self.verbose else None
		self.sample_tiles()
		print('\n-> Encoding sampled tiles and predicting cloud base height') if self.verbose else None
		self.predict_cbh()
		print('\nDone') if self.verbose else None
		return self.preds, self.centers




# endregion