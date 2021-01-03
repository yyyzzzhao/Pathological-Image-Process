import os
import multiresolutionimageinterface as mir
import scipy.io as sio


data_path = 'G:/DataBase/02_ZS_HCC_pathological/04-raw-data/01-differ-tissue'
all_files = [i[:-4] for i in os.listdir(data_path) if i[-4:] == '.xml']  # files with labelled
# print(len(all_files))  # 82 
output_path = 'G:/DataBase/02_ZS_HCC_pathological/02-data-block/01-bbox-patch'

# =========== color <-> type map ===========
label_map = {'#000000': 1,
            '#005500': 2,
            '#aa5500': 3,
            '#00aa00': 4,
            '#aaaa00': 5,
            '#ff0000': 6,
            '#ffff00': 7,
            '#ffaa7f': 8,
            '#00ffff': 9,
            '#ff55ff': 10}

def read_xml_get_tile(image_path, xml_path, output_path):
    print('Start ', str(image_path), str(xml_path))
    patient_id = os.path.basename(image_path)[:-4]
    patient_dir = os.path.join(output_path, patient_id)
    if not os.path.exists(patient_dir):
        os.mkdir(patient_dir)

    reader = mir.MultiResolutionImageReader()
    image = reader.open(str(image_path))

    # =========== read xml and write mask.tif ===========
    annotation_list = mir.AnnotationList()
    xml_repository = mir.XmlRepository(annotation_list)
    xml_repository.setSource(str(xml_path))
    xml_repository.load()

    annotations = annotation_list.getAnnotations()
    if len(annotations) == 0:
        print('empty annotation, id ', patient_id)
        return 0
    temp_map = {}
    for annotation in annotations:
        name = annotation.getName()
        color = annotation.getColor()
        temp_map[name] = label_map[color]

    annotations_mask = mir.AnnotationToMask()
    mask_output = os.path.join(patient_dir, 'mask.tif')
    annotations_mask.convert(annotation_list, mask_output,
                            image.getDimensions(),
                            image.getSpacing(), temp_map)
    reader2 = mir.MultiResolutionImageReader()
    mask = reader2.open(mask_output)

    # =========== make patch ===========
    for annotation in annotations:
        name = annotation.getName()
        color = annotation.getColor()
        x_coord, y_coord = [], []
        for coordinate in annotation.getCoordinates():
            x_coord.append(coordinate.getX())
            y_coord.append(coordinate.getY())
        x_max, x_min = max(x_coord), min(x_coord)
        y_max, y_min = max(y_coord), min(y_coord)
        rect_width = x_max - x_min
        rect_height = y_max - y_min

        image_tile = image.getUCharPatch(int(x_min), int(y_min), int(rect_width), int(rect_height), 0)
        mask_tile = mask.getUCharPatch(int(x_min), int(y_min), int(rect_width), int(rect_height), 0)

        tile_path = os.path.join(patient_dir, str(color))
        if not os.path.exists(tile_path):
            os.mkdir(tile_path)

        sio.savemat(os.path.join(tile_path, name+'.mat'), {'img': image_tile, 'mask': mask_tile})



image_paths = [os.path.join(data_path, i + '.tif') for i in all_files]
xml_paths = [os.path.join(data_path, i + '.xml') for i in all_files]

for i in range(len(image_paths)):
    image_path = image_paths[i]
    xml_path = xml_paths[i]
    read_xml_get_tile(image_path, xml_path, output_path)
