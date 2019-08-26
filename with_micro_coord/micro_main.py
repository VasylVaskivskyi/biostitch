import tifffile as tif
import gc 
import numpy as np
import dask
import dask.array as da


from datetime import datetime
from image_positions import get_image_sizes, get_image_paths_for_fields_per_channel, get_image_paths_for_planes_per_channel
from preprocess_images import read_images, create_z_projection_for_initial_stitching, equalize_histograms
from stitch_images import stitch_images


st = datetime.now()
print('started', st)
xml_path = 'C:/Users/vv3/Desktop/image/images/Hiplex_run1_cycle1_MsPos__2019-03-05T10_52_04-Measurement_2/Index.idx.xml'
img_dir = 'C:/Users/vv3/Desktop/image/images/Hiplex_run1_cycle1_MsPos__2019-03-05T10_52_04-Measurement_2/Images/'
img_output_path = 'C:/Users/vv3/Desktop/image/stitched/coord_test.tif'

img_out_dir = 'C:/Users/vv3/Desktop/image/stitched/'



# -------- Initial stitching ------------
main_channel = 'DAPI'

fields_path_list = get_image_paths_for_fields_per_channel(img_dir, xml_path)
planes_path_list = get_image_paths_for_planes_per_channel(img_dir, xml_path)

z_max_img_list = create_z_projection_for_initial_stitching(main_channel, fields_path_list)
images = equalize_histograms(z_max_img_list)

ids, x_size, y_size = get_image_sizes(xml_path, main_channel, images)
y_size.iloc[1:,:] = y_size.iloc[1:,:] - 1

z_proj = stitch_images(images, ids, x_size, y_size)
tif.imwrite('C:/Users/vv3/Desktop/image/stitched/coord_test.tif', z_proj)

del images, z_proj, z_max_img_list, fields_path_list
gc.collect()


for channel in planes_path_list.keys():
    print('processing channel ', channel)
    result_channel = []
    j = 0
    for plane in planes_path_list[channel]:
        print(j, '/', len(planes_path_list[channel]))
        print('reading images')
        img_list = read_images(plane, is_dir=False)
        print('equalizing histograms')
        images = equalize_histograms(img_list, contrast_limit=101,
                                    grid_size=(37, 37))  # import images and correct uneven illumination
        print('stitching')
        result_plane = stitch_images(images, ids, x_size, y_size)
        result_channel.append(result_plane)
        #result_channel.append(equalize_histograms([result_plane], contrast_limit= 256, grid_size= (71, 81))[0] )
        j += 1
    print('writing channel')
    tif.imwrite(img_out_dir + channel + '.tif', np.stack(result_channel, axis=0))

    # write multilayer image to file

del img_list, images, result_plane, result_channel, ids, x_size, y_size
gc.collect()


paths = [img_out_dir + ch_name + '.tif' for ch_name in planes_path_list.keys()]

lazy_arrays = [dask.delayed(tif.imread(p)) for p in paths]
final_path = img_out_dir + 'stitching_result.tif'

tif.imwrite(
    final_path,
    da.stack(
        [da.from_delayed(x, shape=x._obj.shape, dtype=x._obj.dtype) for x in lazy_arrays], axis=3
    )
)


'''
tif.imwrite(img_out_dir + 'stitching_result.tif',
            np.moveaxis(np.array([tif.imread(p) for p in paths], ndmin=4), 0,3)
            )
'''

fin = datetime.now()
print('elapsed time', fin-st)
