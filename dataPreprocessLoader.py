import numpy as np
import os
from pycocotools.coco import COCO

trainImgDir = "COCO/train2017"
valImgDir = "COCO/val2017"
trainAnnotations = "COCO/annotations/instances_train2017.json"
valAnnotations = "COCO/annotations/instances_val2017.json"

trainObj = COCO(trainAnnotations)
trainImgIds = trainObj.getImgIds()
valObj = COCO(valAnnotations)
valImgIds = valObj.getImgIds()

def extractLabels(obj, img_ids):
    # right now this does single-label
    data = []

    for img_id in img_ids:
        ann_ids = obj.getAnnIds(imgIds=[img_id], iscrowd=None)

        if len(ann_ids) == 0:
            continue
        
        ann = obj.loadAnns([ann_ids[0]])[0]
        cat_id = ann['category_id']
        cat_info = obj.loadCats([cat_id])[0] 
        cat_name = cat_info['name']

        img_info = obj.loadImgs([img_id])[0]
        file_name = img_info['file_name']
        full_path = os.path.join(valImgDir, file_name)

        if os.path.isfile(full_path):
            data.append((full_path, cat_name))
    return data

trainData = extractLabels(trainObj,trainImgIds)
valData = extractLabels(valObj,valImgIds)
