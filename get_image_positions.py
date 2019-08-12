import xml.etree.ElementTree as ET
import pandas as pd


# ----------- Get relative position of images with respect to central image ----------
def median_position(val_list) -> int:
    """find central image"""
    return (len(val_list) // 2) + 1


def assign_rel_pos(array) -> list:
    """arrange images depending on how far they from central image"""
    if array[0][0] > 0:
        for i in range(0, len(array)):
            if array[i][2] > array[i - 1][2]:
                array[i][0] += array[i - 1][0]
            elif array[i][2] == array[i - 1][2]:
                array[i][0] = array[i - 1][0]
    elif array[0][0] < 0:
        for i in range(len(array) - 1, -1, -1):  # reverse order
            if array[i - 1][2] < array[i][2]:
                array[i - 1][0] += array[i][0]
            elif array[i - 1][2] == array[i][2]:
                array[i - 1][0] = array[i][0]

    return array


def create_relative_position(array, center_field) -> list:
    """find central image and sorts other values with respect to center"""

    plus = []   # values displaced in y+ or x+ from center
    minus = []  # values displaced in y- or x- from center
    same = []   # values that occupy same place as center (e.g. y0, x1)
    full_range = []
    for i in range(0, len(array)):
        if array[i] > center_field:
            plus.append([1, i, array[i]])
        elif array[i] < center_field:
            minus.append([-1, i, array[i]])
        else:
            same.append([0, i, array[i]])

    # sort by image coordinates from microscope
    plus.sort(key = lambda x: x[2])
    minus.sort(key = lambda x: x[2])

    # create relative coordinates for images
    plus_sorted = assign_rel_pos(plus)
    minus_sorted = assign_rel_pos(minus)

    # combine all coordinates into one list
    full_range.extend(minus_sorted)
    full_range.extend(same)
    full_range.extend(plus_sorted)
    return full_range


def get_positions_from_xml(path) -> (list, list):
    """read xml metadata and find image metadata (position, channel name) """
    with open(path, 'r', encoding='utf-8') as f:
        xml_file = f.read()
        f.close()

    xml_file = xml_file.replace('xmlns="http://www.perkinelmer.com/PEHH/HarmonyV5"', '')

    xml = ET.fromstring(xml_file)
    tag_Images = xml.find('Images')

    x_pos = []
    y_pos = []
    for img in tag_Images:
        if img.find('ChannelName').text == 'DAPI' and img.find('PlaneID').text == '1':
            x_pos.append(float(img.find('PositionX').text.lstrip('-')))
            y_pos.append(float(img.find('PositionY').text.lstrip('-')))
    return x_pos, y_pos


def get_image_postions(path):
    """specify path to read xml file
    function finds metadata about image location and computes
    relative location to central image."""
    # get microscope coordinates of images from xml file
    x_pos, y_pos = get_positions_from_xml(path)

    # find central field location in the list
    center_field_n = median_position(x_pos)  # x or y doesn't matter

    # find coordinates of central field
    center_field_x_pos = x_pos[center_field_n]
    center_field_y_pos = y_pos[center_field_n]

    # create relative coordinates for all images with respect to central field
    relative_x = create_relative_position(x_pos, center_field_x_pos)
    relative_y = create_relative_position(y_pos, center_field_y_pos)

    # combine x nad y coordinates
    full_range = []

    for i in relative_y:
        for j in relative_x:
            if i[1] == j[1]:  # if field id same
                full_range.append( [j[0], i[0], j[1]] )  # x position, y position, field id

    sorted_full_range = sorted(full_range, key = lambda x: (x[1], x[0]))

    # get range of images in y axis
    y_range = list(set([i[0] for i in relative_y]))
    y_range.sort()

    # arrange relative coordinates into the matrix representing rows and columns of full image
    matrix = []
    for i in y_range:
        temp_matrix = []
        for j in range(0, len(sorted_full_range)):
            if sorted_full_range[j][1] == i:
                temp_matrix.append(sorted_full_range[j])
        matrix.append(temp_matrix)

    return matrix


# ----------- Get full path of each image im xml ----------
def get_target_per_channel_arrangement(xml_path, target):
    """ target is either 'plane' or 'field' """

    xml = ET.parse(xml_path)
    root = xml.getroot()
    Images = root.find('Images')

    metadata_list = []
    for i in Images:
        # field id, plane id, channel name, file name
        metadata_list.append([int(i.find('FieldID').text), int(i.find('PlaneID').text), i.find('ChannelName').text, i.find('URL').text])

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
    c = 0
    for channel in channel_list:
        target_val_list = []
        for i in target_val:
            target_val_list.append(channel.loc[channel.loc[:, target + '_id'] == i, :])
        targets_per_channel.update({channel_names[c]: target_val_list})
        c += 1

    return targets_per_channel


def get_image_paths_for_planes_per_channel(img_dir, xml_path):
    channel_plane_arr = get_target_per_channel_arrangement(xml_path, target='plane')
    channel_paths = {}
    for channel in channel_plane_arr:
        plane_paths = []
        for plane in channel_plane_arr[channel]:
            plane_paths.append([img_dir + fn for fn in plane['file_name'].to_list()])
        channel_paths[channel] = plane_paths
    return channel_paths



def get_image_paths_for_fields_per_channel(img_dir, xml_path):
    channel_field_arr = get_target_per_channel_arrangement(xml_path, target='field')
    channel_paths = {}
    for channel in channel_field_arr:
        field_paths = []
        for field in channel_field_arr[channel]:
            field_paths.append([img_dir + fn for fn in field['file_name'].to_list()])
        channel_paths[channel] = field_paths
    return channel_paths
