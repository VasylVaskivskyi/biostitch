import re
from itertools import chain
import xml.etree.ElementTree as ET
import numpy as np
import pandas as pd


def load_necessary_xml_tags(xml_path):
    """ xml tag Images contain information about image size, resolution, binning, position, wave length, objective information.
        xml tag Name - experiment name.
        xml tag MeasurementStartTime - image acquisition time.
    """
    with open(xml_path, 'r', encoding='utf-8') as f:
        xml_file = f.read()
        f.close()
    # remove this tag to avoid dealing with formatting like this {http://www.perkinelmer.com/PEHH/HarmonyV5}Image
    xml_file = re.sub(r'xmlns="http://www.perkinelmer.com/PEHH/HarmonyV\d"', '',xml_file)
    xml = ET.fromstring(xml_file)
    tag_Images = xml.find('Images')
    tag_Name = xml.find('Plates').find('Plate').find('Name').text.replace(' ', '_')
    tag_MeasurementStartTime = xml.find('Plates').find('Plate').find('MeasurementStartTime').text
    return tag_Images, tag_Name, tag_MeasurementStartTime


def get_positions_from_xml(tag_Images, reference_channel, fovs):
    """read xml metadata and find image metadata (position, channel name) """
    x_resol = round(float(tag_Images[0].find('ImageResolutionX').text), 20)
    y_resol = round(float(tag_Images[0].find('ImageResolutionY').text), 20)

    x_pos = []
    y_pos = []

    img_pos = []
    if fovs is not None:
        for img in tag_Images:
            if img.find('ChannelName').text == reference_channel and img.find('PlaneID').text == '1' and int(img.find('FieldID').text) in fovs:
                x_coord = round(float(img.find('PositionX').text), 9)  # limit precision to nm
                y_coord = round(float(img.find('PositionY').text), 9)

                # convert position to pixels by dividing on resolution in nm
                x_pos.append(round(x_coord / x_resol))  # x_resol[:cut_resol_x]
                y_pos.append(round(y_coord / y_resol))
                img_pos.append((round(x_coord / x_resol),
                                round(y_coord / y_resol),
                                int(img.find('FieldID').text) - 1))  # ids - 1, original data starts from 1
    else:
        for img in tag_Images:
            if img.find('ChannelName').text == reference_channel and img.find('PlaneID').text == '1':
                x_coord = round(float(img.find('PositionX').text), 9)  # limit precision to nm
                y_coord = round(float(img.find('PositionY').text), 9)

                # convert position to pixels by dividing on resolution in nm
                x_pos.append(round(x_coord / x_resol))  # x_resol[:cut_resol_x]
                y_pos.append(round(y_coord / y_resol))
                img_pos.append((round(x_coord / x_resol),
                                round(y_coord / y_resol),
                                int(img.find('FieldID').text) - 1))  # ids - 1, original data starts from 1

    return x_pos, y_pos, img_pos


def get_image_positions_scan_manual(tag_Images, reference_channel, fovs):
    """ Specify path to read xml file.
        Computes position of each picture and zero padding.
    """
    # get microscope coordinates of images from xml file
    x_pos, y_pos, img_pos = get_positions_from_xml(tag_Images, reference_channel, fovs)

    # centering coordinates to 0,0
    leftmost = min(x_pos)
    top = max(y_pos)

    if leftmost < 0:
        leftmost = abs(leftmost)
        if top < 0:
            top = abs(top)
            img_pos = [(pos[0] + leftmost, abs(pos[1]) - top, pos[2]) for pos in img_pos]
            x_pos = [i + leftmost for i in x_pos]
            y_pos = [abs(i) - top for i in y_pos]
        else:
            img_pos = [(pos[0] + leftmost, top - pos[1], pos[2]) for pos in img_pos]
            x_pos = [i + leftmost for i in x_pos]
            y_pos = [top - i for i in y_pos]
    else:
        if top < 0:
            top = abs(top)
            img_pos = [(pos[0] - leftmost, abs(pos[1]) - top, pos[2]) for pos in img_pos]
            x_pos = [i - leftmost for i in x_pos]
            y_pos = [abs(i) - top for i in y_pos]
        else:
            img_pos = [(pos[0] - leftmost, top - pos[1], pos[2]) for pos in img_pos]
            x_pos = [i - leftmost for i in x_pos]
            y_pos = [top - i for i in y_pos]

    y_range = sorted(set(y_pos))  # because set is unordered
    x_range = sorted(set(x_pos))
    # sort image coordinates int rows 
    row_list = []
    for y in y_range:
        row = [i for i in img_pos if i[1] == y]
        if len(row) < len(x_range):
            row_x_range = [i[0] for i in row]
            row_y = row[0][1]
            for x in x_range:
                if x not in row_x_range:
                    row.append((x, row_y, 'zeros'))

        row = sorted(row, key=lambda x: x[0])  # sort by x coordinate
        row_list.append(row)

    ids = []
    x_pos = []
    y_pos = []
    for row in row_list:
        x_pos.append([i[0] for i in row])
        y_pos.append([i[1] for i in row])
        ids.append([i[2] for i in row])

    return ids, x_pos, y_pos


def get_image_sizes_scan_manual(tag_Images, reference_channel, fovs):
    ids, x_pos, y_pos = get_image_positions_scan_manual(tag_Images, reference_channel, fovs)

    ids = np.array(ids)
    x_pos = np.array(x_pos)
    y_pos = np.array(y_pos)

    # assuming that all images are of the same size, so default image size from first image in xml are taken
    default_img_width = int(tag_Images[0].find('ImageSizeX').text)
    default_img_height = int(tag_Images[0].find('ImageSizeY').text)

    # set size of first column and first row to the size of single picture
    x_size = np.diff(x_pos, axis=1)
    y_size = np.diff(y_pos, axis=0)

    last_x_col = np.array([[default_img_width]] * ids.shape[0])
    last_y_row = np.array([[default_img_height] * ids.shape[1]])
    x_size = np.concatenate((x_size, last_x_col), axis=1)
    y_size = np.concatenate((y_size, last_y_row), axis=0)

    return ids, x_size, y_size


def get_image_sizes_scan_auto(tag_Images, reference_channel, fovs):
    """specify path to read xml file
    function finds metadata about image location, computes
    relative location to central image, and size in pixels of each image
    """
    # get microscope coordinates of images from xml file
    x_pos, y_pos, img_pos = get_positions_from_xml(tag_Images, reference_channel, fovs)
    default_img_width = int(tag_Images[0].find('ImageSizeX').text)
    default_img_height = int(tag_Images[0].find('ImageSizeY').text)
    # centering coordinates to 0,0
    leftmost = min(x_pos)
    top = max(y_pos)

    if leftmost < 0:
        leftmost = abs(leftmost)
        if top < 0:
            top = abs(top)
            img_pos = [(pos[0] + leftmost, abs(pos[1]) - top, pos[2]) for pos in img_pos]
            x_pos = [i + leftmost for i in x_pos]
            y_pos = [abs(i) - top for i in y_pos]
        else:
            img_pos = [(pos[0] + leftmost, top - pos[1], pos[2]) for pos in img_pos]
            x_pos = [i + leftmost for i in x_pos]
            y_pos = [top - i for i in y_pos]
    else:
        if top < 0:
            top = abs(top)
            img_pos = [(pos[0] - leftmost, abs(pos[1]) - top, pos[2]) for pos in img_pos]
            x_pos = [i - leftmost for i in x_pos]
            y_pos = [abs(i) - top for i in y_pos]
        else:
            img_pos = [(pos[0] - leftmost, top - pos[1], pos[2]) for pos in img_pos]
            x_pos = [i - leftmost for i in x_pos]
            y_pos = [top - i for i in y_pos]

    from scipy.cluster.hierarchy import fclusterdata
    z = np.array(list(zip(x_pos, y_pos)))

    clusters = fclusterdata(z, t=default_img_height, criterion='distance')

    c = clusters[0]
    c_ids = []
    this_cluster_ids = []
    this_cluster_ypos = []
    c_ypos = []

    for i in range(0, len(clusters)):
        this_val = clusters[i]
        if fovs is None:
            f_id = i
        else:
            f_id = fovs[i]
        if this_val == c:
            this_cluster_ids.append(f_id)
            this_cluster_ypos.append(y_pos[i])
        else:
            c = this_val
            c_ids.append(this_cluster_ids)
            c_ypos.append(this_cluster_ypos)
            this_cluster_ids = [f_id]
            this_cluster_ypos = [y_pos[i]]
        if i == len(clusters) - 1:
            c_ids.append(this_cluster_ids)
            c_ypos.append(this_cluster_ypos)

    ids_in_clusters = [set(c) for c in c_ids]
    y_pos_in_clusters = [sorted(set(c)) for c in c_ypos]

    y_range = sorted(set(y_pos))  # because set is unordered

    """
    for i in range(0, len(y_pos_in_clusters)):
        first_img_size = y_pos_in_clusters[i][1] - y_pos_in_clusters[i][0]
        diff = default_img_height - first_img_size
        for j in range(1, len(y_pos_in_clusters[i])):
            y_pos_in_clusters[i][j] += diff
    """


    y_sizes = []
    for cluster in y_pos_in_clusters:
        y_sizes.extend(list(np.diff(cluster)))
        y_sizes.append(default_img_height)

    y_range = []
    for cluster in y_pos_in_clusters:
        y_range.append(sorted(set(cluster)))

    #y_pos_in_clusters = list(chain.from_iterable(y_pos_in_clusters))

    # image coordinates arranged in rows by same y-coordinate

    row_list = []
    for cluster in y_range:
        for y in cluster:
            row = [j for j in img_pos if j[1] == y]
            if row == []:
                continue
            else:
                row = sorted(row, key=lambda x: x[0])  # sort by x coordinate
                row_list.append(row)
    """           
    for i in range(0, len(y_range)):
        row = [j for j in img_pos if j[1] == y_range[i]]
        row = sorted(row, key=lambda x: x[0])  # sort by x coordinate
        row_list.append(row)
    """
    print(row_list)
    # create image sizes based on difference in coordinates
    img_sizes = []
    row_sizes = []
    for row in row_list:
        img_coords = [i[0] for i in row]
        img_ids = [i[2] for i in row]

        # start each row with zero padding and full width image
        img_size = [(img_coords[0], 'zeros'), (img_coords[1] - img_coords[0], img_ids[0])]
        # detect gaps between images
        for i in range(1, len(img_coords)):
            size = img_coords[i] - img_coords[i - 1]
            # if difference between two adjacent pictures is bigger than width of default picture,
            # then consider this part as a gap, subtract img size from it, and consider the rest as size of a gap
            if size > default_img_width:
                image_size = default_img_width
                gap_size = size - default_img_width
                img_size.extend([(gap_size, 'zeros'), (image_size, img_ids[i])])
            else:
                img_size.append((size, img_ids[i]))

        row_width = sum([i[0] for i in img_size])
        row_sizes.append(row_width)

        img_sizes.append(img_size)

    # add zero padding to the end of each row
    max_width = max(row_sizes)
    for i in range(0, len(row_sizes)):
        diff = max_width - row_sizes[i]
        img_sizes[i].append((diff, 'zeros'))

    # adding y_coordinate to tuple
    for row in range(0, len(img_sizes)):
        img_sizes[row] = [(i[0], y_sizes[row], i[1]) for i in img_sizes[row]]

    # total_height = sum(y_sizes)
    # total_width = max_width

    ids = []
    x_size = []
    y_size = []
    for row in img_sizes:
        x_size.append([i[0] for i in row])
        y_size.append([i[1] for i in row])
        ids.append([i[2] for i in row])
    
    y_pos = list(chain.from_iterable(y_pos_in_clusters))

    return ids, x_size, y_size, ids_in_clusters, y_pos


# ----------- Get full path of each image im xml ----------


def get_target_per_channel_arrangement(tag_Images, target):
    """ target is either 'plane' or 'field' """

    metadata_list = []
    for i in tag_Images:
        # field id, plane id, channel name, file name
        metadata_list.append(
            [int(i.find('FieldID').text), int(i.find('PlaneID').text), i.find('ChannelName').text, i.find('URL').text])

    metadata_table = pd.DataFrame(metadata_list, columns=['field_id', 'plane_id', 'channel_name', 'file_name'])

    # get information about channels names and number of planes
    if target == 'plane':
        target_val = list(metadata_table.loc[:, 'plane_id'].unique())
    elif target == 'field':
        target_val = list(metadata_table.loc[:, 'field_id'].unique())

    channel_names = list(metadata_table.loc[:, 'channel_name'].unique())

    # organize images into list of channels
    channel_list = []
    for i in channel_names:
        channel_list.append(metadata_table.loc[metadata_table.loc[:, 'channel_name'] == i, :])

    # organize images into dictionary of a kind {channel_name: [list of images divided into planes]}
    # e.g. {'DAPI': [25 dataframes]}
    targets_per_channel = {}
    for c, channel in enumerate(channel_list):
        target_val_list = []
        for i in target_val:
            target_val_list.append(channel.loc[channel.loc[:, target + '_id'] == i, :])
        targets_per_channel.update({channel_names[c]: target_val_list})

    return targets_per_channel


def get_image_paths_for_planes_per_channel(img_dir, tag_Images):
    channel_plane_arr = get_target_per_channel_arrangement(tag_Images, target='plane')
    channel_paths = {}
    for channel in channel_plane_arr:
        plane_paths = []
        for plane in channel_plane_arr[channel]:
            plane_paths.append([img_dir + fn for fn in plane['file_name'].to_list()])
        channel_paths[channel] = plane_paths
    return channel_paths


def get_image_paths_for_fields_per_channel(img_dir, tag_Images):
    channel_field_arr = get_target_per_channel_arrangement(tag_Images, target='field')
    channel_paths = {}
    for channel in channel_field_arr:
        field_paths = []
        for field in channel_field_arr[channel]:
            field_paths.append([img_dir + fn for fn in field['file_name'].to_list()])
        channel_paths[channel] = field_paths
    return channel_paths
