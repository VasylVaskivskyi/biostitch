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

def load_necessary_xml_tags(xml_path):
    """ xml tag Images contain information about image size, resolution, binning, position, wave lenght, objective information.
        xml tag Name - experiment name.
        xml tag MeasurementStartTime - image acquisition time.
    """
    with open(xml_path, 'r', encoding='utf-8') as f:
        xml_file = f.read()
        f.close()
    # remove this tag to avoid dealing with formatting like this {http://www.perkinelmer.com/PEHH/HarmonyV5}Image
    xml_file = xml_file.replace('xmlns="http://www.perkinelmer.com/PEHH/HarmonyV5"', '')

    xml = ET.fromstring(xml_file)
    tag_Images = xml.find('Images')
    tag_Name = xml.find('Plates').find('Plate').find('Name').text.replace(' ','_')
    tag_MeasurementStartTime = xml.find('Plates').find('Plate').find('MeasurementStartTime').text
    return tag_Images, tag_Name, tag_MeasurementStartTime


def get_positions_from_xml(tag_Images, main_channel) -> (list, list):
    """read xml metadata and find image metadata (position, channel name) """
    
    magnification = tag_Images[0].find('ObjectiveMagnification').text
    binning = tag_Images[0].find('BinningX').text
    if magnification == '20' and binning == '1':
        correction = 0.9871
    elif magnification == '40' and binning == '2':
        correction = 0.9971
    elif magnification == '5' and binning == '2':
        correction = 1
    else:
        print('There is no correction parameter available for this magnification and binning setup. Results may be inacurate.')
        correction = 1
    
    x_resol = '{:.20f}'.format(float(tag_Images[0].find('ImageResolutionX').text))
    y_resol = '{:.20f}'.format(float(tag_Images[0].find('ImageResolutionY').text))

    x_pos = []
    y_pos = []

    for img in tag_Images:
        if img.find('ChannelName').text == main_channel and img.find('PlaneID').text == '1':
            x_coord = '{:.9f}'.format(float(img.find('PositionX').text))  # limit precision to nm
            y_coord = '{:.9f}'.format(float(img.find('PositionY').text))

            # convert position to pixels by dividing on resolution in nm
            x_pos.append(round(float(x_coord) / float(x_resol) / correction)) # x_resol[:cut_resol_x]
            y_pos.append(round(float(y_coord) / float(y_resol) / correction))

    images_to_ignore = None
    if 0 in x_pos:
        zero_id = x_pos.index(0)
        if y_pos[zero_id] == 0:
            images_to_ignore = zero_id

    return x_pos, y_pos, images_to_ignore


def get_image_positions(tag_Images, main_channel):
    """specify path to read xml file
    function finds metadata about image location, computes
    relative location to central image, and size in pixels of each image"""
    # get microscope coordinates of images from xml file
    x_pos, y_pos, images_to_ignore = get_positions_from_xml(tag_Images, main_channel)

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

    # remove id of image with (0,0) coordinates if there is any
    for i in range(0, len(id_full_range)):
        if id_full_range[i][2] == images_to_ignore:
            del id_full_range[i]

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

    for i in range(0, len(id_full_range)):
        id_df.loc[id_full_range[i][1], id_full_range[i][0]] = id_full_range[i][2]
        x_df.loc[x_full_range[i][1], x_full_range[i][0]] = x_full_range[i][2]
        y_df.loc[y_full_range[i][1], y_full_range[i][0]] = y_full_range[i][2]

    x_df = x_df.ffill(axis=0).bfill(axis=0)
    y_df = y_df.ffill(axis=1).bfill(axis=1)

    # remove columns and rows that have all na values
    na_only_cols = x_df.columns[x_df.isna().all(axis=0)]
    na_only_rows = y_df.index[y_df.isna().all(axis=1)]

    if na_only_cols.empty is False or na_only_rows.empty is False:
        id_df.drop(index=na_only_rows, columns=na_only_cols, inplace=True)
        x_df.drop(index=na_only_rows, columns=na_only_cols, inplace=True)
        y_df.drop(index=na_only_rows, columns=na_only_cols, inplace=True)

    id_df.fillna('zeros', inplace=True)

    return id_df, x_df, y_df


def get_image_sizes_manual(tag_Images, main_channel):
    id_df, x_df, y_df = get_image_positions(tag_Images, main_channel)
    nrows, ncols = id_df.shape
    # pd.options.display.width = 0
    x_size = pd.DataFrame(columns=x_df.columns, index=x_df.index)
    y_size = pd.DataFrame(columns=y_df.columns, index=y_df.index)

    # assuming that all images are of the same size, so default image size from first image in xml are taken
    x = int(tag_Images[0].find('ImageSizeX').text)
    y = int(tag_Images[0].find('ImageSizeY').text)

    for j, id in enumerate(id_df.iloc[:, 0]):
        if id != 'zeros':
            x_size.iloc[j, 0] = x

    for j, id in  enumerate(id_df.iloc[0, :]):
        if id != 'zeros':
            y_size.iloc[0, j] = y

    # set size of first column and first row to the size of single picture
    x_size.iloc[:, 0] = int(round(x_size.iloc[:, 0].mean()))
    y_size.iloc[0, :] = int(round(y_size.iloc[0, :].mean()))

    # fill nan with mean values of cols and rows
    for n in range(1, ncols):
        x_size.iloc[:, n] = abs(x_df.iloc[:, n - 1].subtract(x_df.iloc[:, n]))
        x_size.iloc[:, n] = int(round(x_size.iloc[:, n].mean()))

    for n in range(1, nrows):
        y_size.iloc[n, :] = abs(y_df.iloc[n, :].subtract(y_df.iloc[n - 1, :]))
        y_size.iloc[n, :] = int(round(y_size.iloc[n, :].mean()))
    """
    for n in range(0, ncols):
        y_size.iloc[1:, n] = int(round(y_size.iloc[1:, n].mean()))

    for n in range(0, nrows):
        x_size.iloc[n, 1:] = int(round(x_size.iloc[n, 1:].mean()))
    """
    return id_df, x_size, y_size


def get_image_sizes_auto(tag_Images, main_channel):
    """specify path to read xml file
    function finds metadata about image location, computes
    relative location to central image, and size in pixels of each image"""
    # get microscope coordinates of images from xml file
    x_pos, y_pos, img_pos = get_positions_from_xml(tag_Images, main_channel)
    default_img_width = int(tag_Images[0].find('ImageSizeX').text)
    default_img_height = int(tag_Images[0].find('ImageSizeY').text)

    # centering coordinates to 0,0
    # TODO check for cases where x and y are positive
    leftmost = min(x_pos)
    rightmost = abs(leftmost) + max(x_pos)
    top = max(y_pos)
    bottom = min(y_pos)
    if leftmost < 0:
        if top < 0:
            img_pos = [(pos[0] + abs(leftmost), abs(pos[1]) - abs(top), pos[2]) for pos in img_pos]
            x_pos = [i + abs(leftmost) for i in x_pos]
            y_pos = [(i*(-1) - abs(top)) for i in y_pos]
        else:
            img_pos = [(pos[0] + abs(leftmost), pos[1], pos[2]) for pos in img_pos]
            x_pos = [i + abs(leftmost) for i in x_pos]

    y_range = set(y_pos)
    y_range_sorted = sorted(y_range)
    y_sizes = [default_img_height]

    for i in range(0, len(y_range_sorted)-1):
        y_sizes.append(abs(y_range_sorted[i] - y_range_sorted[i+1]))

    z_score = []
    y_mean = np.mean(y_sizes)
    y_std = np.std(y_sizes) + 0.00001
    for i in y_sizes:
        z_score.append( abs(i - y_mean) / y_std )

    # image coordinates arranged in rows by same y-coordinate
    row_list = []
    for i in range(0, len(y_range_sorted)):
        row = [j for j in img_pos if j[1] == y_range_sorted[i]]
        row_list.append(row)

    # for each row, if z-score of y-size is > 1, then merge those row together
    rows_to_remove = []
    for i in range(1, len(row_list)):
        if z_score[i] > 1:
            prev_row = row_list[i - 1]
            cur_row = row_list[i]
            cur_row = [(val[0], prev_row[0][1], val[2]) for val in cur_row]
            row_list[i - 1].extend(cur_row)
            row_list[i - 1] = sorted(row_list[i - 1], key=lambda x: x[0])
            rows_to_remove.append(i)

    row_list = [row for i, row in enumerate(row_list) if i not in rows_to_remove]
    y_sizes = [y for i, y in enumerate(y_sizes) if i not in rows_to_remove]

    # create image sizes based on difference in coordinates
    img_sizes = []
    row_sizes = []
    for row in range(0, len(row_list)):
        img_coords = [row[0] for row in row_list[row]]
        img_ids = [int(row[2]) - 1 for row in row_list[row]]

        if img_coords[0] < img_coords[-1]:
            pass
        elif img_coords[0] > img_coords[-1]:
            img_coords.reverse()
            img_ids.reverse()

        img_size = [(img_coords[0], 'zeros'), (default_img_width, img_ids[0])]
        for i in range(1, len(img_coords)):
            size = img_coords[i] - img_coords[i - 1]
            if size > default_img_width:
                image_size = default_img_width
                space_size = size - default_img_width
                img_size.extend([(space_size, 'zeros'), (image_size, img_ids[i])])
            else:
                img_size.append((size, img_ids[i]))

        row_width = sum([i[0] for i in img_size])
        row_sizes.append(row_width)

        img_sizes.append(img_size)

    max_width = max(row_sizes)
    for i in range(0, len(row_sizes)):
        if row_sizes[i] < max_width:
            diff = max_width - row_sizes[i]
            img_sizes[i].append((diff, 'zeros'))

    # adding y_coordinate to tuple
    for row in range(0, len(img_sizes)):
        img_sizes[row] = [(i[0], y_sizes[row], i[1]) for i in img_sizes[row]]

    total_height = sum(y_sizes)
    total_width = max_width

    return img_sizes

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

