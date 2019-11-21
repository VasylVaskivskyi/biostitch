#!/usr/bin/env python
# -*- coding: utf-8 -*-

import argparse
import tifffile as tif
from tifffile import TiffWriter
import gc
from datetime import datetime
import os
import cv2 as cv 

from ome_tags import create_ome_metadata, get_channel_metadata
from adaptive_estimation import AdaptiveShiftEstimation
from image_positions import load_necessary_xml_tags, get_image_sizes_auto, get_image_sizes_manual, get_image_paths_for_fields_per_channel, get_image_paths_for_planes_per_channel
from image_processing import stitch_z_projection, create_z_projection_for_fov, stitch_series_of_planes, stitch_plane2
from saving_loading import load_param, save_param

def main():

    parser = argparse.ArgumentParser(
        description="Phenix image stitcher.\nPLEASE DO NOT USE SINGLE QUOTES FOR ARGS")
    parser.add_argument('--xml', type=str, required=True,
                        help='path to the xml file typically ../Images/Index.idx.xml')
    parser.add_argument('--img_dir', type=str, required=True,
                        help='path to the directory with images.')
    parser.add_argument('--out_dir', type=str, required=True,
                        help='path to output directory.')
    parser.add_argument('--reference_channel', type=str, default='none',
                        help='select channel that will be used for estimating stitching parameters. Default is to use first channel.')
    parser.add_argument('--preview_channel', type=str, default='none',
                        help='will generate z-max projection of specified channel in the out_dir.')
    parser.add_argument('--stitch_channels', type=str, nargs='+', default=['all'],
                        help='specify space separated channel names to stitch (e.g. "DAPI" "ALEXA 657"); \nall: will stitch all channels. Default to stitch all channels.')
    parser.add_argument('--channels_to_correct_illumination', type=str, nargs='+', default=['none'],
                        help='specify space separated channel names that require correction of bad illumination (e.g. "DAPI"), RNA spot channels usually do not need correction.\nall: will apply correction to all channels. \nnone: will not apply to any.')
    parser.add_argument('--mode', type=str, default='regular_channel', 
                        help='stack: produce z-stacks.\nmaxz: produce max z-projections instead of z-stacks.')
    parser.add_argument('--adaptive', action='store_true',
                        help='turn on adaptive estimation of image translation')
    parser.add_argument('--save_param', action='store_true', default=False,
                        help='will save parameters estimated during stitching into 3 csv files (image_ids, x_sizes, y sizes)')
    parser.add_argument('--load_param', type=str, default='none',
                        help='specify folder that contais the following csv files: image_ids.csv, x_size.csv, y_sizes.csv, that contain previously estimated parameters')
    parser.add_argument('--scan', type=str, default='none', required=True,
                        help='specify scanning mode (auto or manual)')
    args = parser.parse_args()

    xml_path = args.xml
    img_dir = args.img_dir
    out_dir = args.out_dir
    reference_channel = args.reference_channel
    preview_ch = args.preview_channel
    stitch_only_ch = args.stitch_channels
    ill_cor_ch = args.channels_to_correct_illumination
    stitching_mode = args.mode
    is_adaptive = args.adaptive
    do_save_param = args.save_param
    param_path = args.load_param
    scan = args.scan

    # check if specified directories exist
    if not os.path.isdir(img_dir):
        raise ValueError('img_dir does not exist')
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)
    
    if not out_dir.endswith('/'):
        out_dir = out_dir + '/'
    if not img_dir.endswith('/'):
        img_dir = img_dir + '/'
    
    st = datetime.now()
    print('\nstarted', st)


# ----------- loading xml metadata ------------- #
    tag_Images, tag_Name, tag_MeasurementStartTime = load_necessary_xml_tags(xml_path)
    fields_path_list = get_image_paths_for_fields_per_channel(img_dir, tag_Images)
    planes_path_list = get_image_paths_for_planes_per_channel(img_dir, tag_Images)
    nchannels = len(planes_path_list.keys())
    channel_names = list(planes_path_list.keys())
    channel_ids = {ch: i for i, ch in enumerate(channel_names)}

# ----------- checking parsed arguments ---------- #
    if stitch_only_ch == ['all']:
        if reference_channel == 'none':
            reference_channel = channel_names[0]
    elif stitch_only_ch != ['all']:
        # if user specified custom number of channels check if they are correct
        for i in stitch_only_ch:
            if i not in channel_names:
                raise ValueError('There is no channel with name ' + i + ' in the XML file. ' +
                                 'Available channels ' + ', '.join(channel_names))
        if reference_channel == 'none':
            reference_channel = stitch_only_ch[0]
        nchannels = len(stitch_only_ch)
        channel_names = stitch_only_ch

    if preview_ch != 'none' and preview_ch not in channel_names:
        raise ValueError('Channel specified in --preview_channel is not found. ' +
                         'Available channels ' + ', '.join(channel_names))

    channel_ids = {k: v for k, v in channel_ids.items() if k in channel_names}

    if ill_cor_ch == ['all']:
        ill_cor_ch = channel_names
    elif ill_cor_ch == ['none']:
        ill_cor_ch = []

    available_scan_modes = ('auto', 'manual')
    assert scan in available_scan_modes, 'Incorrect scan mode. Available scan modes ' + ', '.join(available_scan_modes)

    available_stitching_modes = ('stack', 'maxz')
    assert stitching_mode in available_scan_modes, 'Incorrect stitching mode. Available stitching modes ' + ', '.join(available_stitching_modes)

# ----------- estimating image sizes ------------- #
    if param_path == 'none':
        if scan == 'auto':
            micro_img_sizes = get_image_sizes_auto(tag_Images, reference_channel)
            ids = []
            x_size = []
            y_size = []
            for row in micro_img_sizes:
                x_size.append([i[0] for i in row])
                y_size.append([i[1] for i in row])
                ids.append([i[2] for i in row])

        elif scan == 'manual':
            ids, x_size, y_size = get_image_sizes_manual(tag_Images, reference_channel)

        if is_adaptive:
            print('estimating image translation')
            z_max_img_list = create_z_projection_for_fov(reference_channel, fields_path_list)
            estimator = AdaptiveShiftEstimation()
            estimator.scan = scan
            estimator.micro_ids = ids
            estimator.micro_x_size = x_size
            estimator.micro_y_size = y_size
            x_size, y_size = estimator.estimate(z_max_img_list)
            del z_max_img_list
            gc.collect()
    else:
        # ----------- loading previously estimated stitching parameters from files ------------- #
        print('using parameters from loaded files')
        if not param_path.endswith('/'):
            param_path = param_path + '/'
        ids, x_size, y_size = load_param(param_path, scan)

# ---------- saving estimated parameters to files ------------- #
    if do_save_param:
        print('saving_parameters')
        save_param(out_dir, scan, ids, x_size, y_size)

# -------- generating ome metadata ------------ #
    # width and height of single plain
    if scan == 'auto':
        width = sum(x_size[0])
        height = sum([row[0] for row in y_size])
    elif scan == 'manual':
        width = sum(x_size.iloc[0, :])
        height = sum(y_size.iloc[:, 0])
    nplanes = len(planes_path_list[reference_channel])

    channels_meta = get_channel_metadata(tag_Images, channel_ids)
    final_meta = dict()
    for i, channel in enumerate(channel_names):
        final_meta[channel] = channels_meta[channel].replace('Channel', 'Channel ID="Channel:0:' + str(i) + '"')
    ome = create_ome_metadata(tag_Name, width, height, nchannels, nplanes, 1, 'uint16', final_meta, tag_Images, tag_MeasurementStartTime)
    ome_maxz = create_ome_metadata(tag_Name, width, height, nchannels, 1, 1, 'uint16', final_meta, tag_Images, tag_MeasurementStartTime)

# ---------- image stitching ------------- #
    if preview_ch != 'none':
        print('generating max z preview')
        z_proj = stitch_z_projection(preview_ch, fields_path_list, ids, x_size, y_size, False, scan)
        preview_meta = {preview_ch: final_meta[preview_ch]}
        ome_preview = create_ome_metadata(tag_Name, width, height, 1, 1, 1,
                                          'uint16', preview_meta, tag_Images, tag_MeasurementStartTime)
        tif.imwrite(out_dir + 'preview.tif', z_proj, description=ome_preview)
        print('preview is available at ' + out_dir + 'preview.tif')
        del z_proj

    gc.collect()
    if stitching_mode == 'stack':
        final_path_reg = out_dir + tag_Name + '.tif'
        delete = '\b'*20
        grid_size = [int(round(max((x_size.iloc[0, 0], y_size.iloc[0, 0])) / 20))] * 2
        grid_size = tuple(i if i % 2 != 0 else i + 1 for i in grid_size)
        contrast_limit = 256
        clahe = cv.createCLAHE(contrast_limit, grid_size)
        with TiffWriter(final_path_reg, bigtiff=True) as TW:
            for i, channel in enumerate(channel_names):
                print('\nprocessing channel no.{0}/{1} {2}'.format(i+1, nchannels, channel))
                print('started at', datetime.now())
                if channel in ill_cor_ch:
                    do_illum_cor = True
                else:
                    do_illum_cor = False
                    
                for j, plane in enumerate(planes_path_list[channel]):
                    print('{0}plane {1}/{2}'.format(delete, j+1, nplanes), end='', flush=True)
                    TW.save(stitch_plane2(plane, clahe, ids, x_size, y_size, do_illum_cor, scan),
                            photometric='minisblack', contiguous=True, description=ome)
                
    elif stitching_mode == 'maxz':
        final_path_maxz = out_dir + 'maxz_' + tag_Name + '.tif'
        with TiffWriter(final_path_maxz, bigtiff=True) as TW:
            for i, channel in enumerate(channel_names):
                print('\nprocessing channel no.{0}/{1} {2}'.format(i+1, nchannels, channel))
                print('started at', datetime.now())
                
                if channel in ill_cor_ch:
                    do_illum_cor = True
                else:
                    do_illum_cor = False
                
                TW.save(stitch_z_projection(channel, fields_path_list, ids, x_size, y_size, do_illum_cor, scan),
                        photometric='minisblack', contiguous=True, description=ome_maxz)

# ----------- saving ome metadata to a separate XML file ----------- #
    with open(out_dir + 'ome_meta.xml', 'w', encoding='utf-8') as f:
        if stitching_mode == 'regular_plane' or stitching_mode == 'regular_channel':
            f.write(ome)
        if stitching_mode == 'maxz':
            f.write(ome_maxz)

    fin = datetime.now()
    print('\nelapsed time', fin-st)


if __name__ == '__main__':
    main()
