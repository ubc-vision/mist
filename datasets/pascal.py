import torch
import torch.utils.data
import numpy as np
import os
import skimage.io
import skimage.transform
import random
from tqdm import tqdm
import random
import sys

if sys.version_info[0] == 2:
    import xml.etree.cElementTree as ET
else:
    import xml.etree.ElementTree as ET


    
PASCAL_CLASSES = (
    "__background__",
    "aeroplane",
    "bicycle",
    "bird",
    "boat",
    "bottle",
    "bus",
    "car",
    "cat",
    "chair",
    "cow",
    "diningtable",
    "dog",
    "horse",
    "motorbike",
    "person",
    "pottedplant",
    "sheep",
    "sofa",
    "train",
    "tvmonitor",
)


class PascalVOCMetaData():
    def __init__(self, config, mode):
        self.root_dir = config.dataset_dir+'/'+config.dataset+'/'
        self.image_set = mode
        self._imgsetpath = os.path.join(self.root_dir, "ImageSets", "Main", "%s.txt")

        with open(self._imgsetpath % self.image_set) as f:
            self.ids = f.readlines()
        self.ids = [x.strip("\n") for x in self.ids]

        self.cls = PASCAL_CLASSES
    def get_num_class(self):
        return len(self.cls)
    def get_class_name(self, class_id):
        return self.cls[class_id]
    def get_image_name(self, image_id):
        return self.ids[image_id]




# modified based on 'maskrcnn-benchmark' 
# git@github.com:facebookresearch/maskrcnn-benchmark.git
class PascalVOC(torch.utils.data.Dataset):
    def __init__(self, config, mode):
        self.max_objects = config.k
        self.root_dir = config.dataset_dir+'/'+config.dataset+'/'
        self.image_set = mode
        self.keep_difficult = False
        self.num_kp = config.k
        self._annopath = os.path.join(self.root_dir, "Annotations", "%s.xml")
        self._imgpath = os.path.join(self.root_dir, "JPEGImages", "%s.jpg")
        self._imgsetpath = os.path.join(self.root_dir, "ImageSets", "Main", "%s.txt")
        self.image_size = config.image_size
        with open(self._imgsetpath % self.image_set) as f:
            self.ids = f.readlines()
        self.ids = [x.strip("\n") for x in self.ids]
        self.id_to_img_map = {k: v for k, v in enumerate(self.ids)}

        cls = PASCAL_CLASSES
        self.class_to_ind = dict(zip(cls, range(len(cls))))
        self.categories = dict(zip(range(len(cls)), cls))
        self.rand_horiz_flip = config.rand_horiz_flip
        self.rand_maskout = config.rand_maskout

    def __getitem__(self, index):
        img_id = self.ids[index]
        image = skimage.io.imread(self._imgpath%img_id)
        # resize image
        scale = min(image.shape[0], image.shape[1])/self.image_size
        size = [int(image.shape[0]/scale),int(image.shape[1]/scale)]
        image = skimage.transform.resize(image,size)

        anno = ET.parse(self._annopath % img_id).getroot()
        anno = self._preprocess_annotation(anno)
        bboxes = anno['boxes']
        bboxes = bboxes/scale

        keypoints = torch.zeros([self.num_kp,4],device=torch.device('cpu'))

        image = torch.from_numpy(image).permute(2, 0, 1).float()
        
        keypoints[:bboxes.shape[0],[0]] = (bboxes[:,[0]]+bboxes[:,[2]])/2
        keypoints[:bboxes.shape[0],[1]] = (bboxes[:,[1]]+bboxes[:,[3]])/2
        keypoints[:bboxes.shape[0],[2]] = bboxes[:,[2]]-bboxes[:,[0]]
        keypoints[:bboxes.shape[0],[3]] = bboxes[:,[3]]-bboxes[:,[1]]
        
        labels = torch.zeros([self.num_kp],device=torch.device('cpu'))
        labels[:anno['labels'].shape[0]] = anno['labels']

        image, keypoints, labels = self._transform(image,keypoints,labels)



        return image, keypoints, labels

    def _transform(self, image, keypoints, labels):

        if self.rand_horiz_flip and random.random()>0.5:
            image = image.flip(2)
            keypoints[:,0] = image.shape[2] - keypoints[:,0]
        if self.rand_maskout:
            _, H_image, W_image = image.shape
            S_image = min(H_image, W_image)
            H_mask = int((random.random()*0.45+0.15)*S_image)
            W_mask = int((random.random()*0.45+0.15)*S_image)
            y_mask = int(random.random()*(H_image-H_mask-2)) + int(H_mask/2) +1
            x_mask = int(random.random()*(W_image-W_mask-2)) + int(W_mask/2) +1
            avg_color = torch.mean(image[:,y_mask-int(H_mask/2):y_mask+int(H_mask/2),x_mask-int(W_mask/2):x_mask+int(W_mask/2)],(1,2),keepdim=True)
            image[:,y_mask-int(H_mask/2):y_mask+int(H_mask/2),x_mask-int(W_mask/2):x_mask+int(W_mask/2)] = avg_color

        return image, keypoints, labels


    def __len__(self):
        return len(self.ids)



    def _preprocess_annotation(self, target):
        boxes = []
        gt_classes = []
        difficult_boxes = []
        TO_REMOVE = 1
        num_objects = 0
        for obj in target.iter("object"):
            if num_objects == self.max_objects:
                break
            difficult = int(obj.find("difficult").text) == 1
            if not self.keep_difficult and difficult:
                continue
            num_objects = num_objects + 1
            name = obj.find("name").text.lower().strip()
            bb = obj.find("bndbox")
            # Make pixel indexes 0-based
            # Refer to "https://github.com/rbgirshick/py-faster-rcnn/blob/master/lib/datasets/pascal_voc.py#L208-L211"
            box = [
                bb.find("xmin").text,
                bb.find("ymin").text,
                bb.find("xmax").text,
                bb.find("ymax").text,
            ]
            bndbox = tuple(
                map(lambda x: x - TO_REMOVE, list(map(int, box)))
            )

            boxes.append(bndbox)
            gt_classes.append(self.class_to_ind[name])
            difficult_boxes.append(difficult)

        size = target.find("size")
        im_info = tuple(map(int, (size.find("height").text, size.find("width").text)))

        res = {
            "boxes": torch.tensor(boxes, dtype=torch.float32, device=torch.device('cpu')),
            "labels": torch.tensor(gt_classes, device=torch.device('cpu')),
            "difficult": torch.tensor(difficult_boxes, device=torch.device('cpu')),
            "im_info": im_info,
        }
        return res

    def get_img_info(self, index):
        img_id = self.ids[index]
        anno = ET.parse(self._annopath % img_id).getroot()
        size = anno.find("size")
        im_info = tuple(map(int, (size.find("height").text, size.find("width").text)))
        return {"height": im_info[0], "width": im_info[1]}


