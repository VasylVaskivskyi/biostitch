import tifffile as tif
import gc
import numpy as np
from datetime import datetime

#from aicsimageio.writers import ome_tiff_writer

from command_line_args import arguments
from ome_tags import create_ome_metadata, get_channel_metadata
from image_positions import load_xml_tag_Images, get_image_sizes, get_image_paths_for_fields_per_channel, get_image_paths_for_planes_per_channel
from preprocess_images import create_z_projection_for_preview, equalize_histograms
from stitch_images import stitch_images, stitch_big_image


def main():
    st = datetime.now()
    print('\nstarted', st)

    xml_path = arguments.xml
    img_dir = arguments.img_dir
    img_out_dir = arguments.out_dir
    main_channel = arguments.main_channel
    make_preview = arguments.make_preview.upper()
    stitch_only = arguments.stitch_only

    if stitch_only != ['ALL'] and len(stitch_only) == 1:
        main_channel = stitch_only[0]
    if not img_out_dir.endswith('/'):
        img_out_dir = img_out_dir + '/'
    if not img_dir.endswith('/'):
        img_dir = img_dir + '/'

    '''
    xml_path = 'C:/Users/vv3/Desktop/Index.idx.xml'
    xml_path = 'C:/Users/vv3/Desktop/image/images/Hiplex_run1_cycle1_MsPos__2019-03-05T10_52_04-Measurement_2/Index.idx.xml'
    img_dir = 'C:/Users/vv3/Desktop/image/images/Hiplex_run1_cycle1_MsPos__2019-03-05T10_52_04-Measurement_2/Images/'
    img_out_dir = 'C:/Users/vv3/Desktop/image/stitched/'
    main_channel = 'DAPI'
    '''
    tag_Images = load_xml_tag_Images(xml_path)
    fields_path_list = get_image_paths_for_fields_per_channel(img_dir, tag_Images)
    planes_path_list = get_image_paths_for_planes_per_channel(img_dir, tag_Images)
    ids, x_size, y_size = get_image_sizes(tag_Images, main_channel)

    if make_preview == 'TRUE':
        print('generating z-max preview')
        z_max_img_list = create_z_projection_for_preview(main_channel, fields_path_list)
        images = equalize_histograms(z_max_img_list)
        z_proj = stitch_images(images, ids, x_size, y_size)
        tif.imwrite(img_out_dir + 'preview.tif', z_proj)
        print('preview is available at ' + img_out_dir + 'preview.tif')
        del images, z_proj, z_max_img_list, fields_path_list
        gc.collect()

    ncols = sum(x_size.iloc[0, :])
    nrows = sum(y_size.iloc[:, 0])
    nplanes = len(planes_path_list[main_channel])
    nchannels = len(planes_path_list.keys())
    channel_names = list(planes_path_list.keys())
    channels_meta = get_channel_metadata(tag_Images, channel_names)

    if stitch_only != ['ALL']:
        nchannels = len(stitch_only)
        channel_names = stitch_only


    final_path = img_out_dir + 'stitching_result.tif'
    final_image = np.zeros((1, nchannels, nplanes, nrows, ncols), dtype=np.uint16)

    final_meta = dict()
    c = 0
    for channel in channel_names:
        print('\nprocessing channel no.{0}/{1} {2}'.format(c+1, nchannels, channel))
        print('started at', datetime.now())

        final_image[0,c,:, :, :] = stitch_big_image(channel, planes_path_list, ids, x_size, y_size)
        final_meta[channel] = channels_meta[channel].replace('Channel', 'Channel ID="Channel:0:' + str(c) + '"')
        c += 1

    final_image = np.einsum('TCZYX -> TZCYX', final_image)

    ome = create_ome_metadata('stitching_result.tif', 'XYCZT', ncols, nrows, nchannels, nplanes, 1, 'uint16', final_meta, tag_Images)

    del ids, x_size, y_size, channels_meta
    gc.collect()

    tif.imwrite(final_path, final_image, description=ome, metadata=None)

    """
    writer = ome_tiff_writer.OmeTiffWriter(final_path, overwrite_file=True)
    writer.save(final_image, channel_names=channel_names, image_name='stitching_result.tif')
    writer.close()



    with open(img_out_dir + 'ome_meta.xml', 'w', encoding='utf-8') as f:
        f.write(ome)
    """

    fin = datetime.now()
    print('\nelapsed time', fin-st)


if __name__ == '__main__':
    main()
