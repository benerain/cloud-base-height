"""

Utility functions for the method implementation described in the paper:
Marine cloud base height retrieval from MODIS cloud properties using supervised machine learning.

Edited by Julien LENHARDT

Parts of the tile processing are based on code from:
https://github.com/FrontierDevelopmentLab/CUMULO

"""

import xarray as xr
import os
import glob
import pandas as pd
import numpy as np
import itertools
import matplotlib.pyplot as plt
from method.utils.pytorch_class import ModisGlobalTilesDataset
from method.models.models import ConvAutoEncoder
from joblib import load
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

# region Processing tiles from swath file

MIN_N_TILES = 1000
MAX_WIDTH, MAX_HEIGHT = 1354, 2030


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

# 填充缺失值或无效值。
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


# 读取卫星数据文件，提取云属性通道和云掩码。
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
        # 打开文件并去掉时间维度
        level_data = xr.open_dataset(filename).isel(time=0)
    except IndexError as err:
        print('    无法打开文件: {}'.format(err))

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
        cbh = level_data['cloud_layer_base'].min(axis=2).data[:MAX_WIDTH, :MAX_HEIGHT] * 1e3
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
	

def setup_xarray_tile(tile, cbh=None, center=None, tile_size=128):
	"""
	Setup the tile as an xarray.Dataset object.

	:param tile: Cloud properties to include in the tile.
	:param cbh: If provided, cloud base height retrievals from co-located active satellite.
	:param center: (Latitude, Longitude) of the tile's center.
	:param tile_size: Size of the tile.
	:returns: xarray.Dataset object
	"""
	tile_xr = xr.Dataset({
		'cloud_top_pressure': (['lat', 'lon'], tile[0, :, :]),
		'cloud_top_height': (['lat', 'lon'], tile[1, :, :]),
		'cloud_top_temperature': (['lat', 'lon'], tile[2, :, :]),
		'cloud_optical_thickness': (['lat', 'lon'], tile[3, :, :]),
		'cloud_water_path': (['lat', 'lon'], tile[4, :, :]),
		'cloud_mask': (['lat', 'lon'], tile[5, :, :]),
		'cloud_base_height': (['lat', 'lon'], cbh[0]),
		'cloud_base_height_center': cbh[1],
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
		number_of_tiles = min(MIN_N_TILES, len(cloudy_label_pixels_idx))
		# Sample without replacement
		tile_centers_idx = np.random.choice(np.arange(len(cloudy_label_pixels_idx)), number_of_tiles, False)
		cloudy_pixels_idx = np.array(cloudy_label_pixels_idx)
		tile_centers = cloudy_pixels_idx[tile_centers_idx]

	positions, centers, centers_lat_lon, tiles, cbh_values, cbh_values_center = [], [], [], [], [], []

	for center in tile_centers:

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

		return tiles, positions, centers, centers_lat_lon, cbh_values, cbh_values_center

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

	p = axs[0].imshow(tile.cloud_top_height, aspect='equal', vmin=0., cmap='Blues_r')
	cbar = plt.colorbar(p, ax=axs[0], orientation='horizontal', extend='max')
	cbar.set_label('Cloud top height ($m$)')
	axs[0].set_title('Cloud top height ($m$)', weight='bold')
	axs[0].axis('off')

	p = axs[1].imshow(tile.cloud_optical_thickness, aspect='equal', vmin=0., vmax=20., cmap='Blues_r')
	cbar = plt.colorbar(p, ax=axs[1], orientation='horizontal', extend='max')
	cbar.set_label('Cloud optical thickness (a.u.)')
	axs[1].set_title('Cloud optical thickness (a.u.)', weight='bold')
	axs[1].axis('off')

	p = axs[2].imshow(tile.cloud_water_path, aspect='equal', vmin=0., vmax=1000., cmap='Blues_r')
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


def load_data_tiles(ddir, tile_names, means_stds_file):
	"""
	Load pytorch.Dataset object for the tiles.

	:param ddir: Data directory of the tile files.
	:param tile_names: Regex for tiles to use from the data directory.
	:param means_stds_file: File containing means and standard deviations of the different cloud properties.
	:returns: Dataset object from ModisGlobalTilesDataset and corresponding pytorch.DataLoader.
	"""
	# Cloud properties to use in input
	param_cols = ['cloud_top_height', 'cloud_optical_thickness', 'cloud_water_path']
	subset = [1, 3, 4]
	# Additional parameters for the pytorch.Dataset class ModisGlobalTilesDataset
	ext = 'nc'
	subscale = False
	grid_size = 128
	log_transform = False
	normalize = True
	means, stds, mins, _ = load_means_stds(file=means_stds_file, log_transform=log_transform)
	dataset = ModisGlobalTilesDataset(
		ddir=ddir, ext=ext, tile=tile_names, subset=subset, subset_cols=param_cols,
		transform=None, subscale=subscale, grid_size=grid_size,
		normalize=normalize, mean=means, std=stds, min=mins, log_transform=log_transform)
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
	# Load tiles
	dataset, dataloader = load_data_tiles(ddir=ddir, tile_names=tile_names, means_stds_file=means_stds_file)
	# Define torch device to use (if cuda/gpu is available)
	device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
	# Load model
	model = load_ae_model(ae_model=ae_model, device=device)
	# Encodings
	encodings = np.empty(shape=(len(dataset), model.module.latent_dim), dtype=np.float64)
	centers = np.empty(shape=(len(dataset), 2), dtype=np.float64)
	with torch.no_grad():
		for i, data in enumerate(dataloader, 0):
			# Fetch input data
			inputs = data['data']
			inputs = inputs.to(device)
			b_size = inputs.shape[0]
			# Encode
			latent = model.module.encode(inputs)
			# Save encodings
			encodings[i * dataloader.batch_size:i * dataloader.batch_size + b_size, :] = latent.to(
				'cpu').detach().numpy()
			# Save centers
			centers[i * dataloader.batch_size:i * dataloader.batch_size + b_size, :] = data['center'].to(
				'cpu').detach().numpy()
	return encodings, centers


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
	Predict the cloud base height of given tiles following:
	- convolutional auto-encoder processing of tiles
	- ordinal regression prediction of the cloud base height from the encodings

	:param ddir: Data directory of the tile files.
	:param tile_names: Regex for tiles to use from the data directory.
	:param ae_model: File for the convolutional auto-encoder.
	:param or_model: File for the ordinal regression prediction model.
	:param means_stds_file: File containing means and standard deviations of the different cloud properties.
	:param verbose: Display information or not.
	:returns: numpy.array of cloud base height predictions and of corresponding (latitude, longitude) centers.
	"""
	print('    Running encoding of sampled tiles...') if verbose else None
	encodings, centers = encoding_tiles(
		ddir=ddir, tile_names=tile_names, ae_model=ae_model, means_stds_file=means_stds_file)
	print('    Predicting cloud base height...') if verbose else None
	preds = predicting_cbh_or(encodings=encodings, or_model=or_model)
	return preds, centers


# endregion

# region Method wrapper

class CloudBaseHeightPrediction:
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
		self.preds, self.centers = tile_predict_cbh(
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
