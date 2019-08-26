# phenix_image_stitcher



Scripts in folder **with_micro_coord** use microscope coordinates to stitch images


Note for Unix systems:

You need to install imagecodecs-lite in order to read tiff files
`pip install imagecodecs-lite`

There is a problem with allocation of big chunks of memory
https://stackoverflow.com/questions/57507832/unable-to-allocate-array-with-shape-and-data-type
You need to change overcommit value to 1

`echo 1 > /proc/sys/vm/overcommit_memory`

