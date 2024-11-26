"""

Pytorch dataset class for processing tiles for the auto-encoder.

"""

import os
import glob
import xarray as xr
import numpy as np
import torch
from torch.utils.data import Dataset


# region Utils


def read_nc(filename):
	"""
	Read netcdf file and return xarray.Dataset.

	:param filename: Tile filename.
	:returns: xarray.Dataset of the tile
	"""
	file = xr.open_dataset(filename)
	return file


def subscalegrid(output_size, sample):
	"""
	Subscale the input grid in a sample to a given size.

	:param output_size: Desired output size for the grid element.
	:param sample: Data array.
	:returns: Subscaled input grid element.
	"""
	assert output_size < sample.shape[1]
	size = sample.shape[1]
	if output_size % 2 == 0:
		image = sample[:,
		        size // 2 - output_size // 2: size // 2 + output_size // 2,
		        size // 2 - output_size // 2: size // 2 + output_size // 2]
	else:
		image = sample[:,
		        size // 2 - output_size // 2: size // 2 + output_size // 2 + 1,
		        size // 2 - output_size // 2: size // 2 + output_size // 2 + 1]
	return image


def standardize_channel(array, mean, std, channel, epsilon, log_transform):
	"""
	Standardize input array along channels with mean/std.

	:param array: Data array.
	:param mean: Mean of channel.
	:param std: Standard deviation of channel.
	:param channel: Channel name.
	:param epsilon: Minimum value of the corresponding channel.
	:param log_transform: If channel is to be normalized with the log.
	:returns: Standardized data array.
	"""
	if channel in ['cloud_top_pressure', 'cloud_top_height', 'cloud_top_temperature']:
		return (array - mean) / std
	elif channel in ['cloud_optical_thickness', 'cloud_water_path']:
		if log_transform:
			return (np.where(array == 0., np.log(epsilon - 1e-10), np.log(array)) - mean) / std
		else:
			return (array - mean) / std
	else:
		print('Unknown channel {}'.format(channel))
		return None


# endregion

# region Dataset class

class ModisGlobalTilesDataset(Dataset):
	"""
	Pytorch class dataset for the MODIS cloud properties tiles.
	"""

	def __init__(self, ddir, ext, tile=None, subset=None, subset_cols=None,
	             transform=None, log_transform=False, normalize=False, mean=None, std=None,min=None,
	             subscale=False, grid_size=None, get_calipso_cbh=False):
		"""
		Initialization method for the dataset class.

		:param ddir: Data directory containing the tile files.
		:param ext: Extension of the tile files. Should be netCDF files in the implementation here.
		:param tile: File name format for the tile files.
		:param subset: Subset int of cloud properties to use: 0 CTP, 1 CTH, 2 CTT, 3 COT, 4 CWP.
		:param subset_cols: Subset names of cloud properties to use.
		:param transform: pytorch transformations to apply to the samples.
		:param log_transform: Log transform cloud properties (COT and CWP).
		:param normalize: Standardize or not the cloud properties.
		:param mean: List of cloud properties means.
		:param std: List of cloud properties standard deviations.
		:param min: List of cloud properties mins.
		:param subscale: Reduce or not the initial size of the tiles.
		:param grid_size: Size of the tiles.
		:param get_calipso_cbh: Include cloud base height retrievals from co-located active satellite measurements.
		"""
		# Directory with tiles files
		self.ddir = ddir
		# Extension to use for the files
		self.ext = ext
		# Filename to use
		self.tile = 'tile' if tile is None else tile
		# Column subset
		self.subset = subset
		self.subset_cols = subset_cols
		# Transforms to apply
		self.transform = transform
		# Define parameters to transform of input data
		self.subscale = subscale
		self.grid_size = grid_size
		self.normalize = normalize
		self.mean = mean
		self.std = std
		self.min = min
		self.log_transform = log_transform
		self.get_calipso_cbh = get_calipso_cbh
		self.file_paths = glob.glob(os.path.join(self.ddir, self.tile + '*.' + self.ext))
		print('xxxxxxxx', self.file_paths)
		self.file_paths.sort()

		if len(self.file_paths) == 0:
			print('xxxxx',self.file_paths)
			raise FileNotFoundError("No {} files found in directory {} ...".format(self.ext, self.ddir))

	def __len__(self):
		return len(self.file_paths)

	def _standardize_input(self, values):
		for i, col in enumerate(self.subset_cols):
			values[i, :, :] = standardize_channel(values[i, :, :], self.mean[i], self.std[i], col, self.min[i], self.log_transform)
		return values

	def __getitem__(self, idx):

		if torch.is_tensor(idx):
			idx = idx.tolist()

		filename = self.file_paths[idx]
		id = filename.split('/')[-1]

		data = read_nc(filename=filename)
		if self.subset:
			grid_values = np.array(data[self.subset_cols].to_array().values)
		else:
			grid_values = np.array(data.to_array().values)
		# Get Cloud_Mask
		cld_mask = np.array(data['cloud_mask'].values)
		# Get center
		center = np.array(data['center'].values)
		# Get cloud base height from calipso/cloudsat retrievals
		if self.get_calipso_cbh:
			calipso_cbh = np.array(data['cloud_base_height'].values)
			calipso_cbh_center = np.array(data['cloud_base_height_center'].values)

		# Apply subscaling if necessary
		if self.subscale:
			grid_values = subscalegrid(self.grid_size, grid_values)
			cld_mask = subscalegrid(self.grid_size, cld_mask)
		# Apply normalization if necessary
		if self.normalize:
			grid_values = self._standardize_input(grid_values)

		sample = {'data': grid_values.astype(np.float64),
		          'cld_mask': cld_mask.astype(int),
		          'center': center.astype(np.float64),
		          'id': id,
		          'calipso_cbh': calipso_cbh.astype(np.float64) if self.get_calipso_cbh else -1,
		          'calipso_cbh_center': calipso_cbh_center.astype(np.float64) if self.get_calipso_cbh else -1}

		if self.transform:
			sample['data'] = self.transform(sample['data'])

		return sample

# endregion
