import tifffile as tif
import gc 
import numpy as np
import dask
import dask.array as da
import platform

from datetime import datetime
from image_positions import load_xml_tag_Images, get_image_sizes, get_image_paths_for_fields_per_channel, get_image_paths_for_planes_per_channel
from preprocess_images import read_images, create_z_projection_for_initial_stitching, equalize_histograms
from stitch_images import stitch_images


# for linux limit memory usage
def memory_limit():
    soft, hard = resource.getrlimit(resource.RLIMIT_AS)
    resource.setrlimit(resource.RLIMIT_AS, ((get_memory() * 1024) - 2048, hard)) #reserve 2 gigs

def get_memory():
    with open('/proc/meminfo', 'r') as mem:
        free_memory = 0
        for i in mem:
            sline = i.split()
            if str(sline[0]) == 'MemAvailable:':
                free_memory = int(sline[1])
                break
    return free_memory


def stitch_big_image(channel, planes_path_list, ids, x_size, y_size, img_out_dir):
    # write channel multilayer image to file
    print('\nprocessing channel ', channel)
    result_channel = []
    j = 0
    for plane in planes_path_list[channel]:
        print(j, '/', len(planes_path_list[channel]))
        print('reading images')
        img_list = read_images(plane, is_dir=False)
        print('equalizing histograms')
        # import images and correct uneven illumination
        images = equalize_histograms(img_list, contrast_limit=101, grid_size=(37, 37))
        print('stitching')
        result_plane = stitch_images(images, ids, x_size, y_size)
        result_channel.append(result_plane)
        j += 1
    print('writing channel')
    tif.imwrite(img_out_dir + channel + '.tif', np.stack(result_channel, axis=0))


def main():
    st = datetime.now()
    print('started', st)
    xml_path = 'C:/Users/vv3/Desktop/image/images/Hiplex_run1_cycle1_MsPos__2019-03-05T10_52_04-Measurement_2/Index.idx.xml'
    img_dir = 'C:/Users/vv3/Desktop/image/images/Hiplex_run1_cycle1_MsPos__2019-03-05T10_52_04-Measurement_2/Images/'
    img_out_dir = 'C:/Users/vv3/Desktop/image/stitched/'



    # -------- Initial stitching ------------
    main_channel = 'DAPI'
    tag_Images = load_xml_tag_Images(xml_path)
    fields_path_list = get_image_paths_for_fields_per_channel(img_dir, tag_Images)
    planes_path_list = get_image_paths_for_planes_per_channel(img_dir, tag_Images)

    z_max_img_list = create_z_projection_for_initial_stitching(main_channel, fields_path_list)
    images = equalize_histograms(z_max_img_list)

    ids, x_size, y_size = get_image_sizes(tag_Images, main_channel, images)
    y_size.iloc[1:, :] = y_size.iloc[1:, :] - 1

    z_proj = stitch_images(images, ids, x_size, y_size)
    tif.imwrite(img_out_dir + 'coord_test.tif', z_proj)

    del images, z_proj, z_max_img_list, fields_path_list
    gc.collect()


    for channel in planes_path_list.keys():
        stitch_big_image(channel, planes_path_list, ids, x_size, y_size, img_out_dir)


    del ids, x_size, y_size
    gc.collect()


    paths = [img_out_dir + ch_name + '.tif' for ch_name in planes_path_list.keys()]

    lazy_arrays = [dask.delayed(tif.imread(p)) for p in paths]
    lazy_arrays = [da.from_delayed(x, shape=x._obj.shape, dtype=x._obj.dtype) for x in lazy_arrays]
    final_path = img_out_dir + 'stitching_result.tif'

    tif.imwrite(final_path, da.stack(lazy_arrays, axis=3))


    '''
    tif.imwrite(img_out_dir + 'stitching_result.tif',
                np.moveaxis(np.array([tif.imread(p) for p in paths], ndmin=4), 0,3)
                )
    '''

    fin = datetime.now()
    print('elapsed time', fin-st)





if __name__ == '__main__':
    if platform.system() == 'Windows':
        main()
    elif platform.system() == 'Linux':
        import resource
        memory_limit()
        main()
