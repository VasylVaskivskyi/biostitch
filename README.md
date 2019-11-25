# Image stitcher for Opera Phenix HCS System 

This program uses microscope coordinates from `Index.idx.xml` to stitch images. There is also adaptive mode that improves stitching quality by using phase correlation based image registration. Current version allows to process image datasets that are bigger than memory by writing stitched images to the file every channel or every plane. You can stitch dataset of any size as long as you have enough physical memory and amount of RAM at least the size of a single stitched plane.

## Command line arguments:
**Required**

**`--img_dir`**   path to the directory with images.

**`--out_dir`**   path to output directory.

**`--mode`**  **stack**: output image is z-stack; 
            **maxz**: output image is max intensity projections across z-stack.
            
**`--scan`**    specify type of microscope scanning used (manual or auto)


**Optional**


**`-h, --help`**    will print help.

**`--xml`**   path to the xml file with microscope metadata, typically ../Images/Index.idx.xml.

**`--reference_channel`**   select channel that will be used for estimating stitching parameters. Default is to use first channel.

**`--stitch_channels`**   specify space separated channel names to stitch (e.g. "DAPI" "ALEXA 657"). Default is to use all channels.

**`--correct_illumination_in_channels`**  specify space separated channel names that require correction of bad illumination (e.g. "DAPI"), RNA spot channels usually do not need correction. Default is `none`. Use `all` to apply do correction at all.

**`--adaptive`**    flag that enables estimation of stitching parameters using Fourier transformation based registration. If you are enabling this parameter you have to specify expected overlap between images.

**`--make_preview`**  will generate z-max projection of specified channel in the out_dir.

**`--save_param`**     will save parameters estimated during stitching into 3 csv files (image_ids, x_sizes, y_sizes)

**`--load_param`**     specify folder that contains the following csv files: image_ids.csv, x_size.csv, y_sizes.csv, that contain previously estimated parameters

**`--output_name`** specify name of the output image. Default ot use name from Index.idx.xml

**`--fovs`**    specify a comma separated, without spaces, subset of fields of view you want to use for stitching


## Example usage
From repository directory run:
`python biostitch --img_dir "/path/to/dataset/Images/" --out_dir "/path/to/out/directory/" --scan "manual" --mode "maxz" --adaptive --make_preview --stitch_channels "DAPI" "ALEXA 568" --correct_illumination_in_channels "DAPI"` 


## Dependencies

`numpy, pandas, imagecodecs-lite, tifffile, opencv-contrib-python, dask`

This program developed and tested in the **conda** environment and some packages may fail to install correctly without it.
If you want to install `dask` with `pip`, use command `pip install "dask[complete]"`.  

**Installation example for conda**

`conda create -n stitching python=3.7 numpy pandas dask`

`source activate stitching`

`pip install imagecodecs-lite tifffile opencv-contrib-python`

