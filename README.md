# Image stitcher for Opera Phenix HCS System 

This script uses microscope coordinates from `Index.idx.xml` to stitch images.
Current version allows to process image datasets that are bigger than memory by writing stitched images to the file every channel or every plane.

## Command line arguments:
**`--xml`**   path to the xml file with microscope metadata, typically ../Images/Index.idx.xml

**`--img_dir`**   path to the directory with images

**`--out_dir`**   path to output directory

**`--make_preview`**  adding this argument will generate z-max projection of the first channel (typically first channel)

**`--stitch_channels`**   specify space separated channel names to stitch (e.g. "DAPI" "ALEXA 657") default is to use all channels

**`--channels_to_correct_illumination`**  specify space separated channel names that require correction of bad illumination (e.g. "DAPI"), RNA spot channels usually do not need correction

**`--mode`**  **regular_channel**: produce z-stacks, save by channel (uses more memory, a bit faster); 
            **regular_plane**: produce z-stacks, save by plane (uses less memory, a bit slower); 
            **maxz**: produce max z-projections instead of z-stacks.
            
**`--adaptive`**    estimation of stitching parameters will be perfomed using Fourier transfomation based registration. If you are enabling this parameter you have to specify expected overlap between images.

**`--overlap`**     two values that correspond to horizontal and vertical overlap of images in fractions of 1. Default overalp: horizontal 0.1, vertical 0.1


## Dependecies

`numpy, pandas, imagecodecs-lite, tifffile, opencv-contrib-python, scikit-image, dask`

This program developed and tested in the **conda** environment and some packages may fail to install correctly without it.
