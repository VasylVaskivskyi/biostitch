import copy
import gc
import os
from datetime import datetime

import numpy as np
import pandas as pd
import tifffile as tif
from tifffile import TiffWriter

from .adaptive_estimation import AdaptiveShiftEstimation
from .image_positions import load_necessary_xml_tags, get_image_sizes_scan_auto, get_image_sizes_scan_manual, \
    get_path_for_each_plane_and_field_per_channel
from .image_processing import stitch_z_projection, create_z_projection_for_fov, stitch_plane, stitch_images
from .ome_tags import create_ome_metadata, get_channel_metadata
from .saving_loading import load_parameters, save_parameters


class ImageStitcher:
    def __init__(self):
        # user input
        self._img_dir = ''
        self._xml_path = None
        self._out_dir = ''
        self._reference_channel = ''
        self._stitch_only_ch = ['all']
        self._scan = ''
        self._stitching_mode = ''
        self._ill_cor_ch = ['none']
        self._is_adaptive = True
        self._make_preview = True
        self._save_param = ''
        self._load_param_path = 'none'
        self._img_name = ''
        self._fovs = None
        self._extra_meta = None
        # working variables
        self._channel_names = []
        self._nchannels = 0
        self._dtype = np.uint16
        self._measurement_time = ''
        self._ome_meta = ''
        self._preview_ome_meta = ''
        self._channel_ids = {}
        self._y_pos = None
        self._default_img_shape = tuple()


    def stitch(self):
        st = datetime.now()
        print('\nstarted', st)

        self.check_dir_exist()
        self.check_scan_modes()

        tag_Images, field_path_list, plane_path_list = self.load_metadata()
        self._default_img_shape = (int(tag_Images[0].find('ImageSizeY').text), int(tag_Images[0].find('ImageSizeX').text))

        ids, x_size, y_size = self.estimate_image_sizes(tag_Images, field_path_list)
        self.generate_ome_meta(self._channel_ids, x_size, y_size, tag_Images, plane_path_list)
        self.perform_stitching(ids, x_size, y_size, plane_path_list, field_path_list, self._ome_meta)
        self.write_separate_ome_xml()

        fin = datetime.now()
        print('\nelapsed time', fin - st)

    def check_dir_exist(self):
        # check if input and output directories exist
        if not os.path.isdir(self._img_dir):
            raise ValueError('img_dir does not exist')
        if not os.path.exists(self._out_dir):
            os.makedirs(self._out_dir)

        if not self._out_dir.endswith('/'):
            self._out_dir = self._out_dir + '/'
        if not self._img_dir.endswith('/'):
            self._img_dir = self._img_dir + '/'

        if self._xml_path is None:
            self._xml_path = self._img_dir + 'Index.idx.xml'

    def check_scan_modes(self):
        available_scan_modes = ('auto', 'manual')
        if self._scan not in available_scan_modes:
            raise ValueError('Incorrect scan mode. Available scan modes ' + ', '.join(available_scan_modes))

        available_stitching_modes = ('stack', 'maxz')
        if self._stitching_mode not in available_stitching_modes:
            raise ValueError(
                'Incorrect stitching mode. Available stitching modes ' + ', '.join(available_stitching_modes))

    def load_metadata(self):
        tag_Images, tag_Name, tag_MeasurementStartTime = load_necessary_xml_tags(self._xml_path)
        if self._fovs is not None:
            self._fovs = [int(f) for f in self._fovs.split(',')]

        plane_path_list, field_path_list = get_path_for_each_plane_and_field_per_channel(tag_Images, self._img_dir, self._fovs)
        nchannels = len(plane_path_list.keys())
        channel_names = list(plane_path_list.keys())
        channel_ids = {ch: i for i, ch in enumerate(channel_names)}

        if isinstance(self._stitch_only_ch, str):
            self._stitch_only_ch = [self._stitch_only_ch]

        if self._stitch_only_ch == ['all']:
            self._stitch_only_ch = channel_names
            if self._reference_channel == 'none':
                self._reference_channel = channel_names[0]
        elif self._stitch_only_ch != ['all']:
            # if user specified custom number of channels check if they are correct
            for i in self._stitch_only_ch:
                if i not in channel_names:
                    raise ValueError('There is no channel with name ' + i + ' in the XML file. ' +
                                     'Available channels ' + ', '.join(channel_names))
            if self._reference_channel == 'none':
                self._reference_channel = self._stitch_only_ch[0]
            nchannels = len(self._stitch_only_ch)
            channel_names = self._stitch_only_ch

        if isinstance(self._ill_cor_ch, str):
            self._ill_cor_ch = [self._ill_cor_ch]
        if self._ill_cor_ch == ['all']:
            self._ill_cor_ch = {ch: True for ch in channel_names}
        elif self._ill_cor_ch == ['none']:
            self._ill_cor_ch = {ch: False for ch in channel_names}
        else:
            self._ill_cor_ch = {ch: (True if ch in self._ill_cor_ch else False) for ch in channel_names}

        self._channel_ids = {k: v for k, v in channel_ids.items() if k in channel_names}
        self._channel_names = channel_names
        self._nchannels = nchannels
        self._measurement_time = tag_MeasurementStartTime
        if self._img_name == '':
            self._img_name = tag_Name
        if not self._img_name.endswith(('.tif', '.tiff')):
            self._img_name += '.tif'
        return tag_Images, field_path_list, plane_path_list

    def estimate_image_sizes(self, tag_Images, field_path_list):
        if self._load_param_path == 'none':

            if self._scan == 'auto':
                ids, x_size, y_size, ids_in_clusters, self._y_pos = get_image_sizes_scan_auto(tag_Images, self._reference_channel, self._fovs)
                
            elif self._scan == 'manual':
                ids, x_size, y_size = get_image_sizes_scan_manual(tag_Images, self._reference_channel, self._fovs)
                if self._is_adaptive == False:
                    ids = pd.DataFrame(ids)
                    x_size = pd.DataFrame(x_size)
                    y_size = pd.DataFrame(y_size)
                    for j in ids.columns:
                        for i in ids.index:
                            try:
                                val = ids.loc[i, j]
                                val = int(val)
                                ids.loc[i, j] = val
                            except ValueError:
                                pass

            if self._is_adaptive:
                print('estimating image shifts')
                z_max_img_list = create_z_projection_for_fov(self._reference_channel, field_path_list)
                estimator = AdaptiveShiftEstimation()
                estimator.scan = self._scan
                estimator.micro_ids = ids
                estimator.micro_x_size = x_size
                estimator.micro_y_size = y_size
                if self._scan == 'auto':
                    estimator.ids_in_clusters = ids_in_clusters
                    estimator.y_pos = self._y_pos
                ids, x_size, y_size, self._y_pos = estimator.estimate(z_max_img_list)

                if self._make_preview:
                    self.generate_preview(ids, x_size, y_size, self._y_pos, self._preview_ome_meta, z_max_img_list)
                del z_max_img_list
                gc.collect()
        else:
            # loading previously estimated stitching parameters from files
            print('using parameters from loaded files')
            if not self._load_param_path.endswith('/'):
                self._load_param_path = self._load_param_path + '/'
            ids, x_size, y_size, self._y_pos = load_parameters(self._load_param_path, self._scan)

        # saving estimated parameters to files
        if self._save_param:
            print('saving_parameters')
            save_parameters(self._out_dir, self._scan, ids, x_size, y_size, self._y_pos)

        return ids, x_size, y_size

    def generate_ome_meta(self, channel_ids, x_size, y_size, tag_Images, plane_path_list):
        # width and height of single plain
        if self._scan == 'auto':
            width = max([sum(row) for row in x_size])
            height = max(self._y_pos) + self._default_img_shape[0]
        elif self._scan == 'manual':
            width = sum(x_size.iloc[0, :])
            height = sum(y_size.iloc[:, 0])
        nplanes = len(plane_path_list[self._reference_channel])

        channels_meta = get_channel_metadata(tag_Images, channel_ids)
        final_meta = dict()
        for i, channel in enumerate(self._channel_names):
            final_meta[channel] = channels_meta[channel].replace('Channel', 'Channel ID="Channel:0:' + str(i) + '"')

        if self._stitching_mode == 'stack':
            self._ome_meta = create_ome_metadata(self._img_name, width, height, self._nchannels, nplanes, 1, 'uint16', final_meta,
                                                 tag_Images, self._measurement_time, self._extra_meta)
        elif self._stitching_mode == 'maxz':
            self._ome_meta = create_ome_metadata(self._img_name, width, height, self._nchannels, 1, 1, 'uint16', final_meta,
                                                 tag_Images, self._measurement_time, self._extra_meta)

        preview_meta = {self._reference_channel: final_meta[self._reference_channel]}
        self._preview_ome_meta = create_ome_metadata(self._img_name, width, height, 1, 1, 1, 'uint16',
                                                     preview_meta, tag_Images, self._measurement_time, self._extra_meta)

    def generate_preview(self, ids, x_size, y_size, y_pos, metadata, images=None):
        print('generating max z preview')
        z_proj = stitch_images(images, ids, x_size, y_size, y_pos, self._scan)
        tif.imwrite(self._out_dir + 'preview.tif', z_proj, description=metadata)
        print('preview is available at ' + self._out_dir + 'preview.tif')

    def perform_stitching(self, ids, x_size, y_size, plane_path_list, field_path_list, metadata):
        nplanes = len(plane_path_list[self._reference_channel])
        gc.collect()

        if self._stitching_mode == 'stack':
            output_path_stack = self._out_dir + self._img_name
            delete = '\b'*25
            with TiffWriter(output_path_stack, bigtiff=True) as TW:
                for i, channel in enumerate(self._channel_names):
                    print('\nprocessing channel no.{0}/{1} {2}'.format(i + 1, self._nchannels, channel))
                    print('started at', datetime.now())
                    for j, plane in enumerate(plane_path_list[channel]):
                        print('{0}plane {1}/{2}'.format(delete, j + 1, nplanes), end='', flush=True)
                        TW.save(stitch_plane(plane, ids, x_size, y_size, self._y_pos, self._ill_cor_ch[channel], self._scan),
                                photometric='minisblack', contiguous=True, description=metadata)

        elif self._stitching_mode == 'maxz':
            output_path_maxz = self._out_dir + self._img_name
            with TiffWriter(output_path_maxz, bigtiff=True) as TW:
                for i, channel in enumerate(self._channel_names):
                    print('\nprocessing channel no.{0}/{1} {2}'.format(i + 1, self._nchannels, channel))
                    print('started at', datetime.now())

                    TW.save(stitch_z_projection(
                            channel, field_path_list, ids, x_size, y_size, self._y_pos, self._ill_cor_ch[channel], self._scan),
                            photometric='minisblack', contiguous=True, description=metadata)

    def write_separate_ome_xml(self):
        with open(self._out_dir + 'ome_meta.xml', 'w', encoding='utf-8') as f:
            f.write(self._ome_meta)

    @property
    def image_directory(self):
        """ Directory where images are stored """
        return self._img_dir
    @image_directory.setter
    def image_directory(self, value):
        self._img_dir = value

    @property
    def xml_path(self):
        """ Path to xml file with metadata from microscope, Index.idx.xml"""
        return self._xml_path
    @xml_path.setter
    def xml_path(self, value):
        self._xml_path = value

    @property
    def output_directory(self):
        """ Directory in which to store output """
        return self._out_dir
    @output_directory.setter
    def output_directory(self, value):
        self._out_dir = value

    @property
    def reference_channel(self):
        """ Channel that will be used as reference for stitching """
        return self._reference_channel
    @reference_channel.setter
    def reference_channel(self, value):
        self._reference_channel = value

    @property
    def stitch_following_channels(self):
        """ Specify channel names that you want to stitch. Default to stitch all. """
        return self._stitch_only_ch
    @stitch_following_channels.setter
    def stitch_following_channels(self, value):
        self._stitch_only_ch = value

    @property
    def scan_mode(self):
        """ Microscope scanning method used: auto or manual"""
        return self._scan
    @scan_mode.setter
    def scan_mode(self, value):
        self._scan = value

    @property
    def stitching_mode(self):
        """ Stitching mode: stack - generate z stacks, maxz - generate max intensity projection of z stack"""
        return self._stitching_mode
    @stitching_mode.setter
    def stitching_mode(self, value):
        self._stitching_mode = value

    @property
    def correct_illumination_in_channels(self):
        """ Select channels to correct illumination """
        return self._ill_cor_ch
    @correct_illumination_in_channels.setter
    def correct_illumination_in_channels(self, value):
        self._ill_cor_ch = value

    @property
    def load_stitching_parameters_from(self):
        """ Specify directory to load previously estimated stitching parameters """
        return self._load_param_path
    @load_stitching_parameters_from.setter
    def load_stitching_parameters_from(self, value):
        self._load_param_path = value

    @property
    def image_name(self):
        """ Specify name for stitched image. Default is to get it from Index.idx.xml """
        return self._img_name
    @image_name.setter
    def image_name(self, value):
        self._img_name = value

    @property
    def use_adaptive_stitching(self):
        """ Specify if you want to use adaptive stitching """
        return self._is_adaptive
    @use_adaptive_stitching.setter
    def use_adaptive_stitching(self, value):
        self._is_adaptive = value

    @property
    def make_preview(self):
        """ Specify if you want to get preview of reference channel """
        return self._make_preview
    @make_preview.setter
    def make_preview(self, value):
        self._make_preview = value

    @property
    def save_stitching_parameters(self):
        """ Specify if you want to save calculated stitching parameters """
        return self._save_param
    @save_stitching_parameters.setter
    def save_stitching_parameters(self, value):
        self._save_param = value

    @property
    def fovs(self):
        return self._fovs
    @fovs.setter
    def fovs(self, value):
        self._fovs = value

    @property
    def extra_meta(self):
        return self.extra_meta
    @extra_meta.setter
    def extra_meta(self, value):
        self._extra_meta = value
