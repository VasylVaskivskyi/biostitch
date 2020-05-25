import os.path as osp
import re
import xml.etree.ElementTree as ET
from itertools import chain

import numpy as np
import pandas as pd

from typing import List, Union, Tuple, Optional
from .my_types import Image, DF, XML


def load_necessary_xml_tags(xml_path: str) -> Tuple[XML, str, str]:
    """ xml tag Images contain information about image size, resolution, binning, position, wave length, objective information.
        xml tag Name - experiment name.
        xml tag MeasurementStartTime - image acquisition time.
    """
    with open(xml_path, 'r', encoding='utf-8') as f:
        xml_file = f.read()
        f.close()
    # remove this tag to avoid dealing with formatting like this {http://www.perkinelmer.com/PEHH/HarmonyV5}Image
    xml_file = re.sub(r'xmlns="http://www.perkinelmer.com/PEHH/HarmonyV\d"', '', xml_file)
    xml = ET.fromstring(xml_file)
    tag_Images = xml.find('Images')
    tag_Name = xml.find('Plates').find('Plate').find('Name').text.replace(' ', '_')
    tag_MeasurementStartTime = xml.find('Plates').find('Plate').find('MeasurementStartTime').text
    return tag_Images, tag_Name, tag_MeasurementStartTime


def get_positions_from_xml(tag_Images: XML, reference_channel: str, fovs: Union[None, List[int]]) -> Tuple[list, list, list]:
    """read xml metadata and find image metadata (position, channel name) """
    x_resol = round(float(tag_Images[0].find('ImageResolutionX').text), 23)
    y_resol = round(float(tag_Images[0].find('ImageResolutionY').text), 23)

    x_pos = []
    y_pos = []

    img_pos = []
    if fovs is not None:
        for img in tag_Images:
            if img.find('ChannelName').text == reference_channel and img.find('PlaneID').text == '1' and int(img.find('FieldID').text) in fovs:
                x_coord = round(float(img.find('PositionX').text), 10)  # limit precision to nm
                y_coord = round(float(img.find('PositionY').text), 10)

                # convert position to pixels by dividing on resolution in nm
                x_pos.append(round(x_coord / x_resol))
                y_pos.append(round(y_coord / y_resol))
                img_pos.append((round(x_coord / x_resol),
                                round(y_coord / y_resol),
                                int(img.find('FieldID').text) - 1))  # ids - 1, original data starts from 1
    else:
        for img in tag_Images:
            if img.find('ChannelName').text == reference_channel and img.find('PlaneID').text == '1':
                x_coord = round(float(img.find('PositionX').text), 10)  # limit precision to nm
                y_coord = round(float(img.find('PositionY').text), 10)

                # convert position to pixels by dividing on resolution in nm
                x_pos.append(round(x_coord / x_resol))
                y_pos.append(round(y_coord / y_resol))
                img_pos.append((round(x_coord / x_resol),
                                round(y_coord / y_resol),
                                int(img.find('FieldID').text) - 1))  # ids - 1, original data starts from 1

    return x_pos, y_pos, img_pos


def zero_center_coordinates(x_pos: list, y_pos: list, img_pos: list):
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


    return x_pos, y_pos, img_pos


def img_pos_to_size(img_pos_per_row: list, default_img_width: int, default_img_height: int):
    """ Create image sizes based on difference in coordinates """
    y_sizes = [row[0][1] for row in img_pos_per_row]
    y_sizes = np.diff(y_sizes).tolist()
    y_sizes.append(default_img_height)

    x_sizes_per_row = []
    width_per_row = []
    for row in img_pos_per_row:
        x_coords = [i[0] for i in row]
        img_ids = [i[2] for i in row]

        if len(row) == 1:
            row_x_sizes = [(x_coords[0], 'zeros'), (default_img_width, img_ids[0])]
        else:
            # start each row with zero padding and full width image
            row_x_sizes = [(x_coords[0], 'zeros'), (x_coords[1] - x_coords[0], img_ids[0])]
            # detect gaps between images
            for i in range(1, len(x_coords)):
                size = x_coords[i] - x_coords[i - 1]
                # if difference between two adjacent pictures is bigger than width of default picture,
                # then consider this part as a gap, subtract img size from it, and consider the rest as size of a gap
                if size > default_img_width:
                    image_size = default_img_width
                    gap_size = size - default_img_width
                    row_x_sizes.extend([(gap_size, 'zeros'), (image_size, img_ids[i])])
                else:
                    row_x_sizes.append((size, img_ids[i]))

        row_width = sum([i[0] for i in row_x_sizes])
        width_per_row.append(row_width)

        x_sizes_per_row.append(row_x_sizes)

    # add zero padding to the end of each row
    max_width = max(width_per_row)
    for i in range(0, len(width_per_row)):
        diff = max_width - width_per_row[i]
        x_sizes_per_row[i].append((diff, 'zeros'))

    # adding y_coordinate to tuple
    for row in range(0, len(x_sizes_per_row)):
        x_sizes_per_row[row] = [(i[0], y_sizes[row], i[1]) for i in x_sizes_per_row[row]]
    img_sizes_per_row = x_sizes_per_row

    return img_sizes_per_row



def get_image_positions_scan_manual(tag_Images: XML, reference_channel: str, fovs: Union[None, List[int]]) -> Tuple[list, list, list]:
    """ Specify path to read xml file.
        Computes position of each picture and zero padding.
    """
    # get microscope coordinates of images from xml file
    x_pos, y_pos, img_pos = get_positions_from_xml(tag_Images, reference_channel, fovs)

    if fovs is not None:
        fovs = [i - 1 for i in fovs]
        min_fovs = min(fovs)
        fovs = [i - min_fovs for i in fovs]
        img_pos = [(pos[0], pos[1], pos[2] - min_fovs) for pos in img_pos]

    # centering coordinates to 0,0
    x_pos, y_pos, img_pos = zero_center_coordinates(x_pos, y_pos, img_pos)

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

    return ids, x_pos, y_pos, row_list


def get_image_sizes_scan_manual(tag_Images: XML, reference_channel: str, fovs: Union[None, List[int]]) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    ids, x_pos, y_pos, row_list = get_image_positions_scan_manual(tag_Images, reference_channel, fovs)

    #ids = np.array(ids)
    #x_pos = np.array(x_pos)
    #y_pos = np.array(y_pos)

    # assuming that all images are of the same size, so default image size from first image in xml are taken
    default_img_width = int(tag_Images[0].find('ImageSizeX').text)
    default_img_height = int(tag_Images[0].find('ImageSizeY').text)


    # create image sizes based on difference in coordinates

    img_sizes_per_row = img_pos_to_size(row_list, default_img_width, default_img_height)

    ids = []
    x_size = []
    y_size = []
    for row in img_sizes_per_row:
        x_size.append([i[0] for i in row])
        y_size.append([i[1] for i in row])
        ids.append([i[2] for i in row])

    ids = np.array(ids)
    x_size = np.array(x_size)
    y_size = np.array(y_size)

    return ids, x_size, y_size


def get_image_sizes_scan_auto(tag_Images: XML, reference_channel: str, fovs: Union[None, List[int]]):
    """specify path to read xml file
    function finds metadata about image location, computes
    relative location to central image, and size in pixels of each image
    """
    # get microscope coordinates of images from xml file
    x_pos, y_pos, img_pos = get_positions_from_xml(tag_Images, reference_channel, fovs)

    # !IMPORTANT need to mutate fovs to start indexing from 0, and start from 0 for proper clustering
    if fovs is not None:
        fovs = [i - 1 for i in fovs]
        min_fovs = min(fovs)
        fovs = [i - min_fovs for i in fovs]
        img_pos = [(pos[0], pos[1], pos[2] - min_fovs) for pos in img_pos]

    default_img_width = int(tag_Images[0].find('ImageSizeX').text)
    default_img_height = int(tag_Images[0].find('ImageSizeY').text)

    # centering coordinates to 0,0
    x_pos, y_pos, img_pos = zero_center_coordinates(x_pos, y_pos, img_pos)

    from scipy.cluster.hierarchy import fclusterdata
    z = np.array(list(zip(x_pos, y_pos)))

    clusters = fclusterdata(z, t=default_img_height, criterion='distance', method='single')

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

    ids_in_clusters = [set(c) for c in c_ids]  # use set for faster inclusion testing
    y_pos_in_clusters = [sorted(set(c)) for c in c_ypos]

    y_sizes = []
    for cluster in y_pos_in_clusters:
        y_sizes.extend(list(np.diff(cluster)))
        y_sizes.append(default_img_height)

    y_range = []
    for cluster in y_pos_in_clusters:
        y_range.append(sorted(set(cluster)))

    # image coordinates arranged in rows by same y-coordinate
    row_list = []
    for i, cluster in enumerate(ids_in_clusters):
        this_cluster_img_pos = [j for j in img_pos if j[2] in cluster]
        this_cluster_y_range = y_range[i]
        for y in this_cluster_y_range:
            row = [j for j in this_cluster_img_pos if j[1] == y]
            if row == []:
                continue
            else:
                row = sorted(row, key=lambda x: x[0])  # sort by x coordinate
                row_list.append(row)

    # if using fovs check if rows with the same y position are not split by clustering
    if fovs is not None:
        y_pos_dict = {y: [] for y in list(chain.from_iterable(y_range))}
        for i, row in enumerate(row_list):
            row_y_pos = row[0][1]
            y_pos_dict[row_y_pos].append(i)

        rows_to_merge = []
        for k,v in y_pos_dict.items():
            if len(v) > 1:
               rows_to_merge.append(v)

        if rows_to_merge == []:
            pass
        else:
            new_rows = dict()
            rows_to_remove = []
            rows_to_replace = []

            for group in rows_to_merge:
                new_row = []
                for i in group:
                    new_row.extend(row_list[i])
                new_rows[group[0]] = sorted(new_row, key=lambda x: x[0])
                rows_to_remove.extend(group[1:])
                rows_to_replace.append(group[0])

            new_row_list = []
            for i in range(0, len(row_list)):
                if i in rows_to_replace:
                    new_row_list.append(new_rows[i])
                elif i not in rows_to_remove:
                    new_row_list.append(row_list[i])
            row_list = new_row_list

    # create image sizes based on difference in coordinates
    img_sizes_per_row = img_pos_to_size(row_list, default_img_width, default_img_height)

    ids = []
    x_size = []
    y_size = []
    for row in img_sizes_per_row:
        x_size.append([i[0] for i in row])
        y_size.append([i[1] for i in row])
        ids.append([i[2] for i in row])

    y_pos = list(chain.from_iterable(y_pos_in_clusters))

    return ids, x_size, y_size, ids_in_clusters, y_pos


# ----------- Get full path of each image im xml ----------


def get_path_for_each_plane_and_field_per_channel(tag_Images: XML, img_dir: str,
                                                  fovs: Union[None, List[int]]) -> Tuple[dict, dict]:
    """ target is either 'plane' or 'field' """

    metadata_list = []
    if fovs is not None:
        for i in tag_Images:
            # field id, plane id, channel name, file name
            if int(i.find('FieldID').text) in fovs:
                metadata_list.append(
                                    [int(i.find('FieldID').text), int(i.find('PlaneID').text),
                                     i.find('ChannelName').text, i.find('URL').text]
                                    )
    else:
        for i in tag_Images:
            # field id, plane id, channel name, file name
            metadata_list.append(
                                [int(i.find('FieldID').text), int(i.find('PlaneID').text),
                                 i.find('ChannelName').text, i.find('URL').text]
                                )

    metadata_table = pd.DataFrame(metadata_list, columns=['field_id', 'plane_id', 'channel_name', 'file_name'])
    # convert to full path names
    nrows = len(metadata_table['file_name'])
    metadata_table['file_name'] = list(map(osp.join, [img_dir] * nrows, metadata_table['file_name']))
    # get information about channels names and number of planes
    planes = list(metadata_table.loc[:, 'plane_id'].unique())
    fields = list(metadata_table.loc[:, 'field_id'].unique())
    channel_names = list(metadata_table.loc[:, 'channel_name'].unique())

    # organize images into list of channels
    channel_list = []
    for i in channel_names:
        channel_list.append(metadata_table.loc[metadata_table.loc[:, 'channel_name'] == i, :])

    # organize images into dictionary of a kind {channel_name: [list of images divided into planes]}
    # e.g. {'DAPI': [[plane1],[plane2]]}

    plane_per_channel = {}
    for c, channel in enumerate(channel_list):
        plane_list = []
        for i in planes:
            plane_list.append(channel.loc[channel.loc[:, 'plane_id'] == i, 'file_name'].to_list())
        plane_per_channel.update({channel_names[c]: plane_list})

    field_per_channel = {}
    for c, channel in enumerate(channel_list):
        field_list = []
        for i in fields:
            field_list.append(channel.loc[channel.loc[:, 'field_id'] == i, 'file_name'].to_list())
        field_per_channel.update({channel_names[c]: field_list})

    return plane_per_channel, field_per_channel
