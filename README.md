# Image stitcher for Opera Phenix HCS System 

This program uses microscope coordinates from `Index.idx.xml` to stitch images. There is also adaptive mode that improves stitching quality by using phase correlation based image registration. Current version allows to process image datasets that are bigger than memory by writing stitched images to the file every channel or every plane. You can stitch dataset of any size as long as you have enough physical memory and amount of RAM at least the size of a single stitched plane.

## Command line arguments:
**`-h, --help`**    will print help.

**`--xml`**   path to the xml file with microscope metadata, typically ../Images/Index.idx.xml.

**`--img_dir`**   path to the directory with images.

**`--out_dir`**   path to output directory.

**`--reference_channel`**   select channel that will be used for estimating stitching parameters. Default is to use first channel.

**`--stitch_channels`**   specify space separated channel names to stitch (e.g. "DAPI" "ALEXA 657"). Default is to use all channels.

**`--correct_illumination_in_channels`**  specify space separated channel names that require correction of bad illumination (e.g. "DAPI"), RNA spot channels usually do not need correction. Default is `none`. Use `all` to apply do correction at all.

**`--mode`**  **stack**: produce z-stacks; 
            **maxz**: produce max z-projections instead of z-stacks.
            
**`--adaptive`**    flag that enables estimation of stitching parameters using Fourier transformation based registration. If you are enabling this parameter you have to specify expected overlap between images.

**`--make_preview`**  will generate z-max projection of specified channel in the out_dir.

**`--save_param`**     will save parameters estimated during stitching into 3 csv files (image_ids, x_sizes, y_sizes)

**`--load_param`**     specify folder that contains the following csv files: image_ids.csv, x_size.csv, y_sizes.csv, that contain previously estimated parameters

**`--scan`**    specify type of microscope scanning used (manual or auto)

**`--output_name`** specify name of the output image. Default ot use name from Index.idx.xml


## Example usage

`python image_stitcher --xml "/path/to/dataset/Images/Index.idx.xml" --img_dir "/path/to/dataset/Images/" --out_dir "/path/to/out/directory/" --scane "manual" --mode "maxz" --adaptive --overlap 0.1 0.1 --make_preview --stitch_channels "DAPI" "ALEXA 568" --correct_illumination_in_channels "DAPI"`


You start by telling python to run the folder `image_stitcher`, then you have to specify paths where Index.idx.xml (`--xml`) and images (`--img_dir`) are stored, and (`--out_dir`) where you want to save the stitched image. If output directory does not exist it will be created. The output file will have the name from the xml file tag name.

Then add name of the `--mode`**:** `maxz` will write maximum z-projections of every channel, `stack` will create z-stack of images.

The flag `--adaptive` enables adaptive estimation of overlap between images using phase correlation.

The parameter `--make_preview` allows to save max z-projection of reference channel, to assess if stitching was successful or to use later for registration.

If you want to select specific channels to be stitched you can specify them in `--stitch_channels` command, separated by space. Default action is to stitch all channels. Use double quotes (**"Alto 490LS"**) if names have spaces.

If you want to correct uneven illumination in one of the channels you can specify it in the `--correct_illumination_in_channels` command, default value is `none`  - no correction to in all channels, `all`  will apply illumination correction to all channels.

Finally, if you want to save estimated stitching parameters use `--save_params` flag, and if you want to load previously estimated files use `--load_params` specifying path to the directory which contains 3 csv files: image_ids.csv, x_sizes.csv, y_sizes.csv


## Dependencies

`numpy, pandas, imagecodecs-lite, tifffile, opencv-contrib-python, scikit-image, dask`

This program developed and tested in the **conda** environment and some packages may fail to install correctly without it.

**Installation example for conda**

`conda create -n stitching python=3.7 numpy pandas dask`

`source activate stitching`

`pip install imagecodecs-lite tifffile opencv-contrib-python scikit-image`

