import uuid


def get_channel_metadata(tag_Images, channel_ids):
    channel_meta = dict()
    for k, v in channel_ids.items():
        ch = tag_Images[v]
        ch_name = ch.find('ChannelName').text
        binning = '"{0}x{1}"'.format(ch.find('BinningX').text, ch.find('BinningY').text)
        acquisition_mode = ch.find('AcquisitionType').text

        if acquisition_mode == 'NipkowConfocal':
            acquisition_mode = 'SpinningDiskConfocal'

        value = ('<Channel Name="' + ch_name + '" ' +
                    'Fluor="' + ch_name + '" ' +
                    'AcquisitionMode="' + acquisition_mode + '" ' +
                    'IlluminationType="' + ch.find('IlluminationType').text + '" ' +
                    'ContrastMethod="' + ch.find('ChannelType').text + '" ' +
                    'ExcitationWavelength="' + ch.find('MainExcitationWavelength').text + '" ' +
                    'EmissionWavelength="' + ch.find('MainEmissionWavelength').text + '" ' +
                    'SamplesPerPixel="1">' +
                    '<DetectorSettings Binning=' + binning + ' ID="Detector:0:0" />'
                    )
        new_ch = {ch_name: value}
        channel_meta.update(new_ch)

    return channel_meta


def create_ome_metadata(tag_Name, dim_order, X, Y, C, Z, T, dtype, channels_meta, tag_Images, tag_MeasurementStartTime):
    imageid = str(uuid.uuid4())
    xml_start = '<?xml version="1.0" encoding="UTF-8"?>'
    header = '<OME xmlns="http://www.openmicroscopy.org/Schemas/OME/2016-06" xmlns:xsi="http://www.w3.org/2001/XMLSchema-instance" xmlns:str="http://exslt.org/strings" Creator="stitcher" UUID="urn:uuid:{0}" xsi:schemaLocation="http://www.openmicroscopy.org/Schemas/OME/2016-06 http://www.openmicroscopy.org/Schemas/OME/2016-06/ome.xsd">'.format(imageid)

    detector = tag_Images[0].find('CameraType').text
    NA = tag_Images[0].find('ObjectiveNA').text
    magnification = tag_Images[0].find('ObjectiveMagnification').text
    instrument = '<Instrument ID="Instrument:0"><Detector ID="Detector:0:0" Model="{0}" /><Objective ID="Objective:0"  LensNA="{1}"  NominalMagnification="{2}"/></Instrument>'.format(detector, NA, magnification)
    
    image = '<Image ID="Image:0" Name="{0}"><AcquisitionDate>{1}</AcquisitionDate>'.format(tag_Name + '.tif', tag_MeasurementStartTime)
    # multiply by million to convert from metre to um 
    physical_size_x = float(tag_Images[0].find('ImageResolutionX').text) * 1e6
    physical_size_y = float(tag_Images[0].find('ImageResolutionY').text) * 1e6
    pixels = '<Pixels DimensionOrder="{0}" ID="Pixels:0" SignificantBits="{7}" Interleaved="false" PhysicalSizeX="{8}" PhysicalSizeY="{9}" SizeC="{1}" SizeT="{2}" SizeX="{3}" SizeY="{4}" SizeZ="{5}" Type="{6}">'.format(dim_order, C, T, X, Y, Z, dtype, dtype.replace('uint' or 'int' or 'float', ''), physical_size_x, physical_size_y)

    channel = ''
    plane = ''
    IFD = 0
    for t in range(0,T):
        for i, c in enumerate(channels_meta):
            channel += '{0}</Channel>'.format(channels_meta[c])
            for p in range(0,Z):
                plane += '<TiffData FirstC="{0}" FirstT="{1}" FirstZ="{2}" IFD="{3}" PlaneCount="1"></TiffData>'.format(i, t, p, IFD)  
                IFD += 1

    footer = '</Pixels></Image></OME>'
    meta = xml_start + header + instrument + image + pixels + channel + plane + footer
    return meta
