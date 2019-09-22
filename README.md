# Image stitcher for Opera Phenix HCS System 

This script uses microscope coordinates from `Index.idx.xml` to stitch images.
Current version allows to process image datasets that are bigger than memory by appending stitched images to the file every channel or every plane.

## Command line arguments:
**`--xml`**   path to the xml file with microscope metadata, typically ../Images/Index.idx.xml

**`--img_dir`**   path to the directory with images

**`--out_dir`**   path to output directory

**`--make_preview`**  adding this argument will generate z-max projection of the first channel

**`--stitch_channels`**   specify space separated channel names to stitch (e.g. "DAPI" "ALEXA 657") default is to use all channels

**`--channels_to_correct_illumination`**  specify space separated channel names that require correction of bad illumination (e.g. "DAPI"), RNA spot channels usually do not need correction

**`--mode`**  regular_channel: produce z-stacks, save by channel
            regular_plane: produce z-stacks, save by plane
            maxz: produce z-projections instead of z-stacks
            


## Dependecies

`numpy, pandas, imagecodecs-lite, opencv-contrib-python, dask`

mod_lib_tifffile is modified version of package tifffile.py. This modified version restricts number of samples per pixels to 1. Such restrictions prevents rendering multichannel grayscale images as RGB or CMYK images, and so the OME metadata does not interfere with TIFF metadata.
