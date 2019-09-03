import tifffile as tif
import gc 
import numpy as np

from datetime import datetime
from image_positions import load_xml_tag_Images, get_image_sizes, get_image_paths_for_fields_per_channel, get_image_paths_for_planes_per_channel
from preprocess_images import create_z_projection_for_initial_stitching, equalize_histograms
from stitch_images import stitch_images, stitch_big_image


def main():
    st = datetime.now()
    print('started', st)
    xml_path = 'C:/Users/vv3/Desktop/image/images/Hiplex_run1_cycle1_MsPos__2019-03-05T10_52_04-Measurement_2/Index.idx.xml'
    img_dir = 'C:/Users/vv3/Desktop/image/images/Hiplex_run1_cycle1_MsPos__2019-03-05T10_52_04-Measurement_2/Images/'
    img_out_dir = 'C:/Users/vv3/Desktop/image/stitched/'
    main_channel = 'DAPI'

    tag_Images = load_xml_tag_Images(xml_path)
    fields_path_list = get_image_paths_for_fields_per_channel(img_dir, tag_Images)
    planes_path_list = get_image_paths_for_planes_per_channel(img_dir, tag_Images)
    ids, x_size, y_size = get_image_sizes(tag_Images, main_channel)

    z_max_img_list = create_z_projection_for_initial_stitching(main_channel, fields_path_list)
    images = equalize_histograms(z_max_img_list)
    z_proj = stitch_images(images, ids, x_size, y_size)
    tif.imwrite(img_out_dir + 'coord_test_1.tif', z_proj)

    nrows,ncols = z_proj.shape
    n_planes = len(planes_path_list[main_channel])
    n_channels = len(planes_path_list.keys())
    
    del images, z_proj, z_max_img_list, fields_path_list
    gc.collect()
    
    final_image = np.zeros((n_planes, nrows, ncols, n_channels), dtype=np.uint16)
    c = 0
    for channel in planes_path_list.keys():
        final_image[:, :, :, c] = stitch_big_image(channel, planes_path_list, ids, x_size, y_size, img_out_dir)
        c += 1

    del ids, x_size, y_size
    gc.collect()
    final_path = img_out_dir + 'stitching_result.tif'
    print('writing final image')
    tif.imwrite(final_path, final_image)

    '''
    tif.imwrite(img_out_dir + 'stitching_result.tif',
                np.moveaxis(np.array([tif.imread(p) for p in paths], ndmin=4), 0,3)
                )
    '''

    fin = datetime.now()
    print('elapsed time', fin-st)


if __name__ == '__main__':
    main()
