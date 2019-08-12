import tifffile as tif
import numpy as np
from datetime import datetime
from preprocess_images import read_images, create_z_projection_for_initial_stitching, equalize_histograms
from get_image_positions import get_image_postions, get_image_paths_for_fields_per_channel, get_image_paths_for_planes_per_channel
from stitch_images import stitch_images
from secondary_stitching import import_homography, stitch_images2

st = datetime.now()

xml_path = 'C:/Users/vv3/Desktop/image/images/Hiplex_run1_cycle1_MsPos__2019-03-05T10_52_04-Measurement_2/Index.idx.xml'
img_dir = 'C:/Users/vv3/Desktop/image/images/Hiplex_run1_cycle1_MsPos__2019-03-05T10_52_04-Measurement_2/Images/'
img_output_path = 'C:/Users/vv3/Desktop/image/stitched/initial.tif'

img_out_dir = 'C:/Users/vv3/Desktop/image/stitched/'
h_path_horizontal = 'C:/source/stitching/homography_horizontal.tsv'
h_path_vertical = 'C:/source/stitching/homography_vertical.tsv'



# -------- Initial stitching ------------
image_positions = get_image_postions(xml_path)
fields_path_list = get_image_paths_for_fields_per_channel(img_dir, xml_path)

main_channel = 'DAPI'
z_max_img_list = create_z_projection_for_initial_stitching(main_channel, fields_path_list)
images = equalize_histograms(z_max_img_list)
result = stitch_images(images, image_positions)

del images
del result


homography_horizontal_list, picture_size_horizontal_list, homography_vertical_list, picture_size_vertical_list = import_homography(h_path_horizontal, h_path_vertical)


path_list = get_image_paths_for_planes_per_channel(img_dir, xml_path)


for channel in path_list.keys():
    print('processing channel ', channel)
    result_channel = []
    j = 0
    for plane in path_list[channel]:
        print(j, '/', len(path_list[channel]))
        img_list = read_images(plane, is_dir=False)
        images = equalize_histograms(img_list, contrast_limit=127,
                                     grid_size=(41, 41))  # import images and correct uneven illumination
        result_plane = stitch_images2(images, image_positions, homography_horizontal_list, picture_size_horizontal_list,
                                      homography_vertical_list, picture_size_vertical_list)

        result_channel.append(result_plane)
        j += 1
    tif.imwrite(img_out_dir + channel + '.tif', np.stack(result_channel, axis=0))  # write multilayer image to file

del result_plane
del result_channel

st = datetime.now()
tif.imwrite(
    img_out_dir + 'stitching_result.tif',
    np.stack(list(map(tif.imread, [img_out_dir + ch_name + '.tif' for ch_name in path_list.keys()])), axis=3)
    )


fin = datetime.now()
print('finished \n time elapsed ', fin - st)
