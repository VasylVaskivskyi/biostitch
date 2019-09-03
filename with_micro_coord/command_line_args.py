import argparse

parser = argparse.ArgumentParser(description="Phenix image stittcher.\nPLEASE DON'T USE SINGLE QUOTES FOR ARGS")
parser.add_argument('--xml', type=str, nargs='?', help='path to the xml file typically ../Images/Index.idx.xml')
parser.add_argument('--img_dir', type=str, nargs='?', help='path to the directory with images')
parser.add_argument('--out_dir', type=str, nargs='?', help='path to output directory')
parser.add_argument('--main_channel', type=str, nargs='?', help='channel that will be used as basis for stitching')
arguments = parser.parse_args()
