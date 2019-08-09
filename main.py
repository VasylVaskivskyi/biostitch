import tifffile as tif
from datetime import datetime
from preprocess_images import preprocess_images
from get_image_positions import get_image_postions
from stitch_images import stitch_images

st = datetime.now()

xml_path = 'C:/Users/vv3/Desktop/image/images/Hiplex_run1_cycle1_MsPos__2019-03-05T10_52_04-Measurement_2/Index.idx.xml'
img_dir = 'C:/Users/vv3/Desktop/image/out_zmax/' #'C:/Users/vv3/Desktop/image/hiplex_zmax/'
img_output_path = 'C:/Users/vv3/Desktop/image/stitched/img_fin.tif'

image_positions = get_image_postions(xml_path)  # get image positions from xml file
images = preprocess_images(img_dir)  # import images and correct uneven illumination
result = stitch_images(images, image_positions)  # stitch images
tif.imwrite(img_output_path, result)  # write image to file

fin = datetime.now()
print('finished \n time elapsed ', fin - st)
