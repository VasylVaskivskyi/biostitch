# Image stitcher for Opera Phenix HCS System 

This program uses microscope coordinates from `Index.idx.xml` to stitch images. There is also adaptive mode that improves stitching quality by using phase correlation based image registration. Current version allows to process image datasets that are bigger than memory by writing stitched images to the file every channel or every plane. You can stitch dataset of any size as long as you have enough physical memory and amount of RAM at least the size of a single stitched plane.

## Command line arguments:
**`-h, --help`**    will print help.

**`--xml`**   path to the xml file with microscope metadata, typically ../Images/Index.idx.xml.

**`--img_dir`**   path to the directory with images.

**`--out_dir`**   path to output directory.

**`--reference_channel`**   select channel that will be used for estimating stitching parameters. Default is to use first channel.

**`--stitch_channels`**   specify space separated channel names to stitch (e.g. "DAPI" "ALEXA 657"). Default is to use all channels.

**`--channels_to_correct_illumination`**  specify space separated channel names that require correction of bad illumination (e.g. "DAPI"), RNA spot channels usually do not need correction. Default is `none`. Use `all` to apply do correction at all.

**`--mode`**  **regular_channel**: produce z-stacks, save by channel (uses more memory, a bit faster); 
            **regular_plane**: produce z-stacks, save by plane (uses less memory, a bit slower); 
            **maxz**: produce max z-projections instead of z-stacks.
            
**`--adaptive`**    flag that enables estimation of stitching parameters using Fourier transformation based registration. If you are enabling this parameter you have to specify expected overlap between images.

**`--overlap`**     if adaptive flag is enable you have to specify two values that correspond to horizontal and vertical overlap of images in fractions of 1. Default overlap: horizontal 0.1, vertical 0.1.

**`--preview_channel`**  will generate z-max projection of specified channel in the out_dir.

**`--save_params`**     will save parameters estimated during stitching into 3 csv files (image_ids, x_sizes, y_sizes)

**`--load_params`**     specify folder that contains the following csv files: image_ids.csv, x_size.csv, y_sizes.csv, that contain previously estimated parameters

## Example usage

`python image_stitcher --xml "/path/to/dataset/Images/Index.idx.xml" --img_dir "/path/to/dataset/Images/" --out_dir "/path/to/out/directory/" --mode "maxz" --adaptive --overlap 0.1 0.1 --make_preview --stitch_channels "DAPI" "ALEXA 568" --channels_to_correct_illumination "DAPI"`


You start by telling python to run the folder `image_stitcher`, then you have to specify paths where Index.idx.xml (`--xml`) and images (`--img_dir`) are stored, and (`--out_dir`) where you want to save the stitched image. If output directory does not exist it will be created. The output file will have the name from the xml file tag name.

Then add name of the `--mode`**:** `maxz` will write maximum z-projections of every channel, `regular_channel` will be processing and writing image to the disk by channels, and `regular_plane` will be processing and writing image to disk by z-planes. 

The flag `--adaptive` enables adaptive estimation of overlap between images, however when enabling in you have to specify **maximum percent of overlap** between images in the `--overlap` command **in fractions of 1**. The overlap in theory should not be a big value, typically 10-20 percent (0.1-0.2). 

The parameter `--preview_channel` allows to save max z-projection of specified channel, to assess if stitching was successful or use later for registration.

If you want to select specific channels to be stitched you can specify them in `--stitch_channels` command, separated by space, the default value is to stitch all channels. Use double quotes (**"Alto 490LS"**) if names have spaces.

If you want to correct uneven illumination in one of the channels you can specify it in the `--channels_to_correct_illumination` command, default value is apply correction to all channels. You can provide `none` to prevent applying illumination correction.

Finally, if you want to save estimated stitching parameters use `--save_params` flag, and if you want to load previously estimated files use `--load_params` specifying path to the directory which contains 3 csv files: image_ids.csv, x_sizes.csv, y_sizes.csv


## Dependencies

`numpy, pandas, imagecodecs-lite, tifffile, opencv-contrib-python, scikit-image, dask`

This program developed and tested in the **conda** environment and some packages may fail to install correctly without it.

**Installation example for conda**

`conda create -n stitching python=3.7 numpy pandas dask`

`source activate stitching`

`pip install imagecodecs-lite tifffile opencv-contrib-python scikit-image`

