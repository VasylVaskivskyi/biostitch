import tifffile as tif
from datetime import datetime
from preprocess_images import preprocess_images
from get_image_positions import get_image_postions
from secondary_stitching import import_homography, stitch_images2

st = datetime.now()

xml_path = 'C:/Users/vv3/Desktop/image/images/Hiplex_run1_cycle1_MsPos__2019-03-05T10_52_04-Measurement_2/Index.idx.xml'
img_dir = 'C:/Users/vv3/Desktop/image/out_zmax/' #'C:/Users/vv3/Desktop/image/hiplex_zmax/'
img_output_path = 'C:/Users/vv3/Desktop/image/stitched/secondary.tif'

h_path_horizontal = 'C:/Users/vv3/Desktop/image/stitched/homography_horizontal.tsv'
h_path_vertical = 'C:/Users/vv3/Desktop/image/stitched/homography_vertical.tsv'

image_positions = get_image_postions(xml_path)  # get image positions from xml file
images = preprocess_images(img_dir)  # import images and correct uneven illumination

homography_horizontal_list, homography_vertical_list = import_homography(h_path_horizontal, h_path_vertical)

result = stitch_images2(images, image_positions, homography_horizontal_list, homography_vertical_list)
tif.imwrite(img_output_path, result)  # write image to file

fin = datetime.now()
print('finished \n time elapsed ', fin - st)
