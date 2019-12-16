#!/usr/bin/env python
# -*- coding: utf-8 -*-

import argparse
from biostitch import ImageStitcher


def main():
    parser = argparse.ArgumentParser(
        description="Phenix image stitcher.\nPLEASE DO NOT USE SINGLE QUOTES FOR ARGS")
    parser.add_argument('--img_dir', type=str, required=True,
                        help='path to the directory with images.')
    parser.add_argument('--xml', type=str, default=None,
                        help='path to the xml file typically ../Images/Index.idx.xml')
    parser.add_argument('--out_dir', type=str, required=True,
                        help='path to output directory.')
    parser.add_argument('--scan', type=str, default='none', required=True,
                        help='specify scanning mode (auto or manual)')
    parser.add_argument('--mode', type=str, required=True,
                        help='stack: produce z-stacks.\nmaxz: produce max z-projections instead of z-stacks.')
    parser.add_argument('--reference_channel', type=str, default='none',
                        help='select channel that will be used for estimating stitching parameters. Default is to use first channel.')
    parser.add_argument('--make_preview', action='store_true',
                        help='will generate z-max projection of reference channel in the out_dir.')
    parser.add_argument('--stitch_channels', type=str, nargs='+', default='all',
                        help='specify space separated channel names to stitch (e.g. "DAPI" "ALEXA 657"). Default to stitch all channels.')
    parser.add_argument('--correct_illumination_in_channels', type=str, nargs='+', default='none',
                        help='specify space separated channel names that require correction of bad illumination (e.g. "DAPI"), RNA spot channels usually do not need correction.\nall: will apply correction to all channels. \nnone: will not apply to any.')
    parser.add_argument('--adaptive', action='store_true',
                        help='turn on adaptive estimation of image translation')
    parser.add_argument('--save_param', action='store_true', default=False,
                        help='will save parameters estimated during stitching into 3 csv files (image_ids, x_sizes, y sizes)')
    parser.add_argument('--load_param', type=str, default='none',
                        help='specify folder that contais the following csv files: image_ids.csv, x_size.csv, y_sizes.csv, that contain previously estimated parameters')
    parser.add_argument('--output_name', type=str, default='',
                        help='specify name of the output image. Default ot use name from Index.idx.xml')
    parser.add_argument('--fovs', type=str, default=None,
                        help='specify a comma separated, without spaces, subset of fields of view you want to use for stitching')
    parser.add_argument('--extra_meta', type=str, default=None,
                        help='JSON formatted extra metadata ("channel_names")')
    args = parser.parse_args()


    stitcher = ImageStitcher()
    stitcher.image_directory = args.img_dir
    stitcher.xml_path = args.xml
    stitcher.output_directory = args.out_dir
    stitcher.image_name = args.output_name
    stitcher.reference_channel = args.reference_channel
    stitcher.stitch_following_channels = args.stitch_channels
    stitcher.scan_mode = args.scan
    stitcher.stitching_mode = args.mode
    stitcher.correct_illumination_in_channels = args.correct_illumination_in_channels
    stitcher.load_stitching_parameters_from = args.load_param
    stitcher.use_adaptive_stitching = args.adaptive
    stitcher.make_preview = args.make_preview
    stitcher.save_stitching_parameters = args.save_param
    stitcher.fovs = args.fovs
    stitcher.stitch()


if __name__ == '__main__':
    main()
