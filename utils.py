import scipy.io as sio

def load_data(path, data_type, data_class):
    raw = sio.loadmat(path)
    if data_class == "train":
        if data_type == "data":
            cars_data = [(row[5][0][:],row[4][0][0],row[0][0][0],row[1][0][0],row[2][0][0],row[3][0][0]) for row in raw['annotations'][0]]
            return cars_data

        if data_type == "class":
            cars_classes = [(row[0]) for row in raw['class_names'][0]]
            return cars_classes
        
    if data_class == "test":
        if data_type == "data":
            cars_data = [(row[4][0][:],"empty",row[0][0][0],row[1][0][0],row[2][0][0],row[3][0][0]) for row in raw['annotations'][0]]
            return cars_data

    # if data_class == "all":
    # 	if data_type == "class":
    # 		cars_classes = [(row[0]) for row in raw['class_names'][0]]
    # 		return cars_classes