import xml.etree.ElementTree as ET
import pandas as pd
import numpy as np
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

def load_xml_tag_Images(xml_path):
    with open(xml_path, 'r', encoding='utf-8') as f:
        xml_file = f.read()
        f.close()

    xml_file = xml_file.replace('xmlns="http://www.perkinelmer.com/PEHH/HarmonyV5"', '')

    xml = ET.fromstring(xml_file)
    tag_Images = xml.find('Images')
    return tag_Images


def get_positions_from_xml(tag_Images, main_channel) -> (list, list):
    """read xml metadata and find image metadata (position, channel name) """

    x_resol = '{:.20f}'.format(float(tag_Images[0].find('ImageResolutionX').text))
    y_resol = '{:.20f}'.format(float(tag_Images[0].find('ImageResolutionY').text))

    x_pos = []
    y_pos = []

    for img in tag_Images:
        if img.find('ChannelName').text == main_channel and img.find('PlaneID').text == '1':
            x_coord = '{:.9f}'.format(float(img.find('PositionX').text))  # limit precision to nm
            y_coord = '{:.9f}'.format(float(img.find('PositionY').text))

            x_pos.append(round(float(x_coord) / float(x_resol[:len(x_coord)-1]) ))  #convert to position in pixels by dividing on resolution in nm
            y_pos.append(round(float(y_coord) / float(y_resol[:len(y_coord)-1]) ))

    return x_pos, y_pos


def get_image_positions(tag_Images, main_channel):
    """specify path to read xml file
    function finds metadata about image location and computes
    relative location to central image."""
    # get microscope coordinates of images from xml file
    x_pos, y_pos = get_positions_from_xml(tag_Images, main_channel)

    # find central field location in the list
    center_field_n = median_position(x_pos)  # x or y doesn't matter

    # find coordinates of central field
    center_field_x_pos = x_pos[center_field_n]
    center_field_y_pos = y_pos[center_field_n]

    # create relative coordinates for all images with respect to central field
    relative_x = create_relative_position(x_pos, center_field_x_pos)
    relative_y = create_relative_position(y_pos, center_field_y_pos)

    # combine x nad y coordinates
    id_full_range = []

    for i in relative_y:
        for j in relative_x:
            if i[1] == j[1]:  # if field id same
                id_full_range.append( [j[0], i[0], j[1]] )  # x position, y position, field id

    x_full_range = []
    y_full_range = []

    for pos in id_full_range:
        x_full_range.append([pos[0], pos[1], abs(x_pos[pos[2]]) ])
        y_full_range.append([pos[0], pos[1], abs(y_pos[pos[2]]) ])

    # get range of images in y axis
    y_range = list(set([i[0] for i in relative_y]))
    y_range.sort()

    # get range of images in x axis
    x_range = list(set([i[0] for i in relative_x]))
    x_range.sort()

    # arrange relative coordinates into the matrix representing rows and columns of full image
    id_df = pd.DataFrame(columns=x_range, index=y_range)
    id_df.sort_index(ascending=False, inplace=True)
    x_df = id_df.copy()
    y_df = id_df.copy()

    for i in id_full_range:
        id_df.loc[i[1], i[0]] = i[2]

    for i in range(0, len(id_full_range)):
        id_df.loc[id_full_range[i][1], id_full_range[i][0]] = id_full_range[i][2]
        x_df.loc[x_full_range[i][1], x_full_range[i][0]] = x_full_range[i][2]
        y_df.loc[y_full_range[i][1], y_full_range[i][0]] = y_full_range[i][2]

    id_df.fillna('zeros', inplace=True)

    return id_df, x_df, y_df


def get_image_sizes(tag_Images, main_channel):
    id_df, x_df, y_df = get_image_positions(tag_Images, main_channel)
    nrows, ncols = id_df.shape
    # pd.options.display.width = 0
    x_size = pd.DataFrame(columns=x_df.columns, index=x_df.index)
    y_size = pd.DataFrame(columns=y_df.columns, index=y_df.index)

    # assuming that all images are of the same size, so default image size from first image in xml are taken
    x = int(tag_Images[0].find('ImageSizeX').text)
    y = int(tag_Images[0].find('ImageSizeY').text)

    j = 0
    for i in id_df.iloc[:, 0]:
        if i != 'zeros':
            x_size.iloc[j, 0] = x
        j += 1

    j = 0
    for i in id_df.iloc[0, :]:
        if i != 'zeros':
            y_size.iloc[0, j] = y
        j += 1

    y_size.iloc[0, :] = y_size.iloc[0, :].fillna(method='ffill', axis=0).fillna(method='bfill', axis=0).astype(np.int64)
    # axis=0 because series don't have axis 1
    x_size.iloc[:, 0] = x_size.iloc[:, 0].fillna(method='ffill', axis=0).fillna(method='bfill', axis=0).astype(np.int64)

    for n in range(1, ncols):
        x_size.iloc[:, n] = x_df.iloc[:, n - 1].subtract(x_df.iloc[:, n])

    for n in range(1, nrows):
        y_size.iloc[n, :] = y_df.iloc[n, :].subtract(y_df.iloc[n - 1, :])

    x_size = x_size.fillna(method='ffill', axis=0).fillna(method='bfill', axis=0).astype(np.int64)
    y_size = y_size.fillna(method='ffill', axis=1).fillna(method='bfill', axis=1).astype(np.int64)

    return id_df, x_size, y_size

# ----------- Get full path of each image im xml ----------


def get_target_per_channel_arrangement(tag_Images, target):
    """ target is either 'plane' or 'field' """

    metadata_list = []
    for i in tag_Images:
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

