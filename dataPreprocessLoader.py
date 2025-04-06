import numpy as np
import os
import torch
from pycocotools.coco import COCO
from skimage import transform, io
from sklearn import preprocessing

##### DATA FILE PATHS #####
trainImgDir = "C:/repos/COCO/train2017"
valImgDir = "C:/repos/COCO/val2017"
trainAnnotations = "C:/repos/COCO/annotations/instances_train2017.json"
valAnnotations = "C:/repos/COCO/annotations/instances_val2017.json"

##### GETTING DATA LOADED PROPERLY #####
trainObj = COCO(trainAnnotations)
trainImgIds = trainObj.getImgIds()[:1000]
valObj = COCO(valAnnotations)
valImgIds = valObj.getImgIds()[:1000]

def extractLabels(obj, img_ids, img_dir):
    # right now this does single-label
    data = []
    for img_id in img_ids:
        ann_ids = obj.getAnnIds(imgIds=[img_id], iscrowd=None)
        if len(ann_ids) == 0:
            continue
    
        # GETS ANNOTATIONS - change for multilabel
        ann = obj.loadAnns([ann_ids[0]])[0]
        cat_id = ann['category_id']
        cat_info = obj.loadCats([cat_id])[0] 
        cat_name = cat_info['name']

        # RETRIEVES IMG FILE PATH
        img_info = obj.loadImgs([img_id])[0]
        file_name = img_info['file_name']
        full_path = os.path.join(img_dir, file_name)
        if os.path.isfile(full_path):
            # PAIRS IMG WITH LBL
            data.append((full_path, cat_name))
    return data


trainData = extractLabels(trainObj,trainImgIds, trainImgDir)
valData = extractLabels(valObj,valImgIds, valImgDir)

##### PREPROCESS AND FORMAT DATA #####
def transformImage(image_path, output_size=(224, 224)):
    # PREPROCESS IMG
    img = io.imread(image_path)
    img = transform.resize(img, output_size)
    if img.ndim == 2:
        img = np.stack([img] * 3, axis=-1)
    img = img.transpose(2,0,1)
    return img

def createDataloader(image_label_pairs, output_size=(224, 224)):
    X_list, y_list = [], []
    # RETRIEVE ACTUAL IMG
    for idx, (fpath, lbl) in enumerate(image_label_pairs):
        img = transformImage(fpath, output_size=output_size)
        X_list.append(img)
        y_list.append(lbl)
        if (idx + 1) % 100 == 0:
            print(f"Processed {idx+1}/{len(image_label_pairs)} images")
    
    # ENCODE LBLS
    y_encoded = preprocessing.LabelEncoder().fit_transform(np.array(y_list))
    X_tensor = torch.tensor(np.array(X_list, dtype=np.float32))
    y_tensor = torch.tensor(y_encoded, dtype=torch.long)
    dataset = torch.utils.data.TensorDataset(X_tensor, y_tensor)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=32, shuffle=True)
    return dataloader


trainDataloader = createDataloader(trainData)
valDataloader = createDataloader(valData)
print(f"Loaded {len(trainData)} training images.")
print(f"Loaded {len(valData)} validation images.")


