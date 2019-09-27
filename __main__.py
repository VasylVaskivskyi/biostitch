#!/usr/bin/env python
# -*- coding: utf-8 -*-

import argparse
import tifffile as tif
from tifffile import TiffWriter
import gc
import numpy as np
from datetime import datetime
import os
import cv2 as cv 

from ome_tags import create_ome_metadata, get_channel_metadata
from adaptive_estimation import AdaptiveShiftEstimation
from image_positions import load_necessary_xml_tags, get_image_sizes, get_image_paths_for_fields_per_channel, get_image_paths_for_planes_per_channel
from image_processing import create_z_projection, create_z_projection_for_fov, equalize_histograms, stitch_images, stitch_series_of_planes, stitch_plane2


def main():

    parser = argparse.ArgumentParser(
        description="Phenix image stitcher.\nPLEASE DO NOT USE SINGLE QUOTES FOR ARGS")
    parser.add_argument('--xml', type=str, required=True,
                        help='path to the xml file typically ../Images/Index.idx.xml')
    parser.add_argument('--img_dir', type=str, required=True,
                        help='path to the directory with images')
    parser.add_argument('--out_dir', type=str, required=True,
                        help='path to output directory')
    parser.add_argument('--make_preview', action='store_true', default=False,
                        help='will generate z-max projection of main_channel')
    parser.add_argument('--stitch_channels', type=str, nargs='+', default=['all'], help='specify space separated channel names to stitch (e.g. "DAPI" "ALEXA 657") default is to use all channels. \nall: will stitch all channels.')
    parser.add_argument('--channels_to_correct_illumination', type=str, nargs='+', default=['all'], help='specify space separated channel names that require correction of bad illumination (e.g. "DAPI"), RNA spot channels usually do not need correction.\nall: will apply correction to all channels. \nnone: will not apply to any.')
    parser.add_argument('--mode', type=str, default='regular_channel', help='regular_channel: produce z-stacks, save by channel.\nregular_plane: produce z-stacks, save by plane.\nmaxz: produce z-projections instead of z-stacks.')
    args = parser.parse_args()

    xml_path = args.xml
    img_dir = args.img_dir
    img_out_dir = args.out_dir
    make_preview = args.make_preview
    stitch_only_ch = args.stitch_channels
    ill_cor_ch = args.channels_to_correct_illumination
    stitching_mode = args.mode

    # check if specified directories exist
    if not os.path.isdir(img_dir):
        raise ValueError('img_dir do not exist')
        exit(1)
    if not os.path.isdir(img_out_dir):
        raise ValueError('img_out_dir do not exist')
        exit(1)
    
    if not img_out_dir.endswith('/'):
        img_out_dir = img_out_dir + '/'
    if not img_dir.endswith('/'):
        img_dir = img_dir + '/'
    
    st = datetime.now()
    print('\nstarted', st)

    '''
    xml_path = 'C:/Users/vv3/Desktop/image/images/Hiplex_run1_cycle1_MsPos__2019-03-05T10_52_04-Measurement_2/Index.idx.xml'
    img_dir = 'C:/Users/vv3/Desktop/image/images/Hiplex_run1_cycle1_MsPos__2019-03-05T10_52_04-Measurement_2/Images/'
    img_out_dir = 'C:/Users/vv3/Desktop/image/stitched/'
    main_channel = 'DAPI'
    '''
    
    tag_Images, tag_Name, tag_MeasurementStartTime = load_necessary_xml_tags(xml_path)
    fields_path_list = get_image_paths_for_fields_per_channel(img_dir, tag_Images)
    planes_path_list = get_image_paths_for_planes_per_channel(img_dir, tag_Images)
    nchannels = len(planes_path_list.keys())
    channel_names = list(planes_path_list.keys())
    
    if stitch_only_ch == ['all']:
        main_channel  = channel_names[0]
    elif stitch_only_ch != ['all']:
        # if user specified custom number of channels check if they are correct
        for i in stitch_only_ch:
            if i not in channel_names:
                raise ValueError('There is no channel with name ' + i +  ' in the XML file')
                exit(1)
                
        main_channel = stitch_only_ch[0]
        nchannels = len(stitch_only_ch)
        channel_names = stitch_only_ch
    
    if ill_cor_ch == ['all']:
        ill_cor_ch = channel_names
    elif ill_cor_ch == ['none']:
        ill_cor = []
    
    ids, x_size, y_size = get_image_sizes(tag_Images, main_channel)
    z_max_img_list = create_z_projection_for_fov(main_channel, fields_path_list)
    x_size, y_size = AdaptiveShiftEstimation().estimate_image_sizes(z_max_img_list, ids, 0.1, 0.1)

    ncols = sum(x_size.iloc[0, :])
    nrows = sum(y_size.iloc[:, 0])
    nplanes = len(planes_path_list[main_channel])

    channels_meta = get_channel_metadata(tag_Images, channel_names)
    
    if make_preview:
        print('generating z-max preview')
        z_proj = create_z_projection(main_channel, fields_path_list) 
        tif.imwrite(img_out_dir + 'preview.tif', z_proj)
        print('preview is available at ' + img_out_dir + 'preview.tif')
        del z_proj
        gc.collect()

    
    final_meta = dict()
    for i, channel in enumerate(channel_names):
        final_meta[channel] = channels_meta[channel].replace('Channel', 'Channel ID="Channel:0:' + str(i) + '"')
    ome = create_ome_metadata(tag_Name, 'XYCZT', ncols, nrows, nchannels, nplanes, 1, 'uint16', final_meta, tag_Images, tag_MeasurementStartTime)
    ome_maxz = create_ome_metadata(tag_Name, 'XYCZT', ncols, nrows, nchannels, 1, 1, 'uint16', final_meta, tag_Images, tag_MeasurementStartTime) 
    
    if stitching_mode == 'regular_channel':
        final_path_reg = img_out_dir + tag_Name + '.tif'
        with TiffWriter(final_path_reg, bigtiff=True) as TW:
            for i, channel in enumerate(channel_names):
                print('\nprocessing channel no.{0}/{1} {2}'.format(i+1, nchannels, channel))
                print('started at', datetime.now())
                
                if channel in ill_cor_ch:
                    do_illum_cor = True
                else:
                    do_illum_cor = False
                               
                TW.save(stitch_series_of_planes(channel, planes_path_list, ids, x_size, y_size, do_illum_cor), photometric='minisblack', contiguous=True, description=ome)
    
    elif stitching_mode == 'regular_plane':
        final_path_reg = img_out_dir + tag_Name + '.tif'
        delete = '\b'*20
        contrast_limit = 127
        grid_size = (41, 41)
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
                    TW.save(stitch_plane2(plane, clahe, ids, x_size, y_size, do_illum_cor), photometric='minisblack', contiguous=True, description=ome)
                
    elif stitching_mode == 'maxz':
        final_path_maxz = img_out_dir + 'maxz_' + tag_Name + '.tif'
        with TiffWriter(final_path_maxz, bigtiff=True) as TW:
            for i, channel in enumerate(channel_names):
                print('\nprocessing channel no.{0}/{1} {2}'.format(i+1, nchannels, channel))
                print('started at', datetime.now())
                
                if channel in ill_cor_ch:
                    do_illum_cor = True
                else:
                    do_illum_cor = False
                
                TW.save(create_z_projection(channel, fields_path_list, ids, x_size, y_size, do_illum_cor), photometric='minisblack',contiguous=True, description=ome_maxz)
                

    del ids, x_size, y_size, channels_meta
    gc.collect()
    
    with open(img_out_dir + 'ome_meta.xml', 'w', encoding='utf-8') as f:
        if stitching_mode == 'regular_plane' or stitching_mode == 'regular_channel':
            f.write(ome)
        if stitching_mode == 'maxz':
            f.write(ome_maxz)
    

    fin = datetime.now()
    print('\nelapsed time', fin-st)


if __name__ == '__main__':
    main()
