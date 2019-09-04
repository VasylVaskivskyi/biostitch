import argparse

parser = argparse.ArgumentParser(description="Phenix image stittcher.\nPLEASE DON'T USE SINGLE QUOTES FOR ARGS")
parser.add_argument('--xml', type=str, nargs='?', help='path to the xml file typically ../Images/Index.idx.xml')
parser.add_argument('--img_dir', type=str, nargs='?', help='path to the directory with images')
parser.add_argument('--out_dir', type=str, nargs='?', help='path to output directory')
parser.add_argument('--main_channel', type=str, nargs='?', help='channel that will be used as basis for stitching e.g. DAPI')
parser.add_argument('--make_preview', type=str, nargs='?', default='TRUE', help='BOOL, if true, will generate z-max projection of main_channel')
parser.add_argument('--stitch_only', type=str, nargs='+', default=['ALL'], help='specify space separated channel names to stitch (e.g. "DAPI" "ALEXA 657") default is to use all cahnnels')

arguments = parser.parse_args()
