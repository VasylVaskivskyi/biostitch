import uuid

def create_ome_metadata(img_name, dim_order, C, T, X, Y, Z, dtype, channels_info):
    imageid = str(uuid.uuid4())
    header = '<?xml version="1.0" encoding="UTF-8"?>\n<OME xmlns:xsi="http://www.w3.org/2001/XMLSchema-instance" Creator="stitcher" UUID="urn:uuid:{0}" xsi:schemaLocation="http://www.openmicroscopy.org/Schemas/OME/2016-06">\n'.format(imageid)
    image_name = '<Image ID="Image:0" Name="{0}">\n'.format(img_name)
    pixels = '<Pixels DimensionOrder="{0}" ID="Pixels:0" Interleaved="false" SizeC="{1}" SizeT="{2}" SizeX="{3}" SizeY="{4}" SizeZ="{5}" Type="{6}" >'.format(dim_order, C, T, X, Y, Z, dtype)

    channel = ''
    plane = ''
    for t in range(0,T):
        for i, c in enumerate(channels_info):
            channel += '<Channel ID="{0}" Name="{1}" SamplesPerPixel="1"><LightPath /></Channel>'.format(c['ID'], c['Name'])
            for p in range(0,Z):
                plane += '<TiffData FirstC="{0}" FirstT="{1}" FirstZ="{2}" IFD="{3}" PlaneCount="1"><UUID FileName="multi-channel-z-series.ome.tif">urn:uuid:{4}</UUID></TiffData>'.format(i, t, p, (t+1)*(i+1)*p, imageid)  # using t+1, i+1 because they start from 0

    #bindata = '<BinData BigEndian = "false" Compression = "none" Length = "0"/>'
    footer = '</Pixels></Image></OME>'
    xml = header + image_name + pixels + channel + plane + footer
    return xml
