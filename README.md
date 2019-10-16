# Image stitcher for Opera Phenix HCS System 

This script uses microscope coordinates from `Index.idx.xml` to stitch images.
Current version allows to process image datasets that are bigger than memory by writing stitched images to the file every channel or every plane.

## Command line arguments:
**`-h, --help`**    will print help.

**`--xml`**   path to the xml file with microscope metadata, typically ../Images/Index.idx.xml.

**`--img_dir`**   path to the directory with images.

**`--out_dir`**   path to output directory.

**`--stitch_channels`**   specify space separated channel names to stitch (e.g. "DAPI" "ALEXA 657"). Default is to use all channels.

**`--channels_to_correct_illumination`**  specify space separated channel names that require correction of bad illumination (e.g. "DAPI"), RNA spot channels usually do not need correction. Default is to apply correction to all channels. Specify `none` if you do not want to do correction at all.

**`--mode`**  **regular_channel**: produce z-stacks, save by channel (uses more memory, a bit faster); 
            **regular_plane**: produce z-stacks, save by plane (uses less memory, a bit slower); 
            **maxz**: produce max z-projections instead of z-stacks.
            
**`--adaptive`**    flag that enables estimation of stitching parameters using Fourier transformation based registration. If you are enabling this parameter you have to specify expected overlap between images.

**`--overlap`**     if adaptive flag is enable you have to specify two values that correspond to horizontal and vertical overlap of images in fractions of 1. Default overlap: horizontal 0.1, vertical 0.1.

**`--make_preview`**  enabling this flag will generate z-max projection of the first channel (usually DAPI) to the out_dir.

## Example usage

`python image_stitcher --xml "/path/to/dataset/Images/Index.idx.xml" --img_dir "/path/to/dataset/Images/" --out_dir "/path/to/out/directory/" --mode "maxz" --adaptive --overlap 0.1 0.1 --make_preview --stitch_channels "DAPI" "ALEXA 568" --channels_to_correct_illumination "DAPI"`

You start with telling python to run the folder `image_stitcher`, then you have to specify paths where Index.idx.xml (`--xml`) and images (`--img_dir`) are stored. 
Then add name of the `--mode`**:** `maxz` will write maximum z-projections of every channel, `regular_channel` will be processing and writing image to the disk by channels, and `regular_plane` will be processing and writing image to disk by z-planes. 
The flag `--adaptive` enables adaptive estimation of overlap between images, however when enabling in you have to specify **maximum percent of overlap** between images in the `--overlap` command **in fractions of 1**. The overlap in theory should not be a big value, typically 10-20 percent (0.1-0.2). 
The flag `--make_preview` will allow to save max z-projection of the first channel which is usually a DAPI channel, to assess if stitching was successful or use later for registration. 
If you want to select specific channels to be stitched you can specify them in `--stitch_cahnnels` command, separated by space, the default value is to stitch all channels. Use double quotes (**"Alto 490LS"**) if names have spaces. 
If you want to correct uneven illumination in one of the channels you can specify it in the `--channels_to_correct_illumination` command, default value is apply correction to all channels. You can provide `none` to prevent applying illumination correction.


## Dependencies

`numpy, pandas, imagecodecs-lite, tifffile, opencv-contrib-python, scikit-image, dask`

This program developed and tested in the **conda** environment and some packages may fail to install correctly without it.

**Installation example for conda**
`conda create -n stitching python=3.7 numpy pandas dask`
`source activate stitching`
`pip install imagecodecs-lite tifffile opencv-contrib-python scikit-image`
