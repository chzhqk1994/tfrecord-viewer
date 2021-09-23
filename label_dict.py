def get_label_dict(dataset_name):
    label_dict = None
    if dataset_name == 'PASCAL':
        label_dict = {
            'back_ground': 0,
            'aeroplane': 1,
            'bicycle': 2,
            'bird': 3,
            'boat': 4,
            'bottle': 5,
            'bus': 6,
            'car': 7,
            'cat': 8,
            'chair': 9,
            'cow': 10,
            'diningtable': 11,
            'dog': 12,
            'horse': 13,
            'motorbike': 14,
            'person': 15,
            'pottedplant': 16,
            'sheep': 17,
            'sofa': 18,
            'train': 19,
            'tvmonitor': 20
        }

    elif dataset_name == 'COCO':
        pass

    elif dataset_name == 'ROOF':
        label_dict = {
            'back_ground': 0,
            'flatroof': 1,
            'solarpanel_slope': 2,
            'solarpanel_flat': 3,
            'parkinglot': 4,
            'facility': 5,
            'rooftop': 6,
            'heliport_r': 7,
            'heliport_h': 8
        }
    return label_dict
