# Feature based image registrator

The programs uses `Fast` feature finder and `Daisy` feature descriptor for registration. It can align images of different size by padding them with 0 values. The image registrator can work with multichannel grayscale TIFFs and TIFFs with multiple z-planes. Images must have OME-TIFF XML in their description.

## Command line arguments

**`--maxz_images`**     specify, separated by space, paths to maxz images of anchor channels you want to use for **estimating registration parameters**.
They should also include reference image.

**`--maxz_ref_image`**  specify path to **reference maxz image**, the one that will be used as reference for aligning all other images.

**`--register_images`**   specify, separated by space, paths to the **images you want to apply registration to** (e.g. z-stacked images, multichannel images, maxz images).
They should be in the same order as images specified in `--maxz_images` argument. If not specified `--maxz_images` will be used.

**`--out_dir`**     directory to output registered image

**`--estimate_only`** add this flag if you want to get only registration parameters and do not want to process images

**`--load_params`**  specify path to csv file that store registration parameters


## Example usage

`python reg.py --maxz_images "/path/to/image1/img1.tif" "/path/to/image2/img2.tif" --maxz_ref_image "/path/to/image2/img2.tif" --register_images "/path/to/image1/img1_zstack.tif" "/path/to/image2/img2_zstack.tif" --out_dir "/path/to/output/"`

## Dependencies

`numpy pandas tifffile opencv-contrib-python`
