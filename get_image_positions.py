import xml.etree.ElementTree as ET


def median_position(val_list) -> int:
    return (len(val_list) // 2) + 1


def create_relative_position(array, center_field) -> list:
    """ function finds central image and sorts other values with respect to center"""

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

    plus.sort(key = lambda x: x[2])
    minus.sort(key = lambda x: x[2])
    plus_sorted = assign_rel_pos(plus)
    minus_sorted = assign_rel_pos(minus)

    full_range.extend(minus_sorted)
    full_range.extend(same)
    full_range.extend(plus_sorted)
    return full_range


def assign_rel_pos(array) -> list:
    # range images depending on how far they from center image
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


def get_positions_from_xml(path) -> (list, list):
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
            # location = 'X {0} Y {1}'.format(img.find('PositionX').text, img.find('PositionY').text)
            # dapi_loc_list.append(location)
            x_pos.append(float(img.find('PositionX').text.lstrip('-')))
            y_pos.append(float(img.find('PositionY').text.lstrip('-')))
    return x_pos, y_pos


def get_image_postions(path):
    x_pos, y_pos = get_positions_from_xml(path)

    center_field_n = median_position(x_pos)  # x or y doesn't matter

    center_field_x_pos = x_pos[center_field_n]
    center_field_y_pos = y_pos[center_field_n]

    relative_x = create_relative_position(x_pos, center_field_x_pos)
    relative_y = create_relative_position(y_pos, center_field_y_pos)

    full_range = []

    for i in relative_y:
        for j in relative_x:
            if i[1] == j[1]:
                full_range.append( [j[0],i[0],j[1]] )  # x postition, y position, filed id

    sorted_full_range = sorted(full_range, key = lambda x: (x[1], x[0]))

    y_range = list(set([i[0] for i in relative_y]))
    y_range.sort()

    matrix = []
    for i in y_range:
        temp_matrix = []
        for j in range(0, len(sorted_full_range)):
            if sorted_full_range[j][1] == i:
                temp_matrix.append(sorted_full_range[j])
        matrix.append(temp_matrix)

    return matrix
