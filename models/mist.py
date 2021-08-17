import torch
from models.detector import Detector
from models.classifier import ClassifierNetWork
from utils.torch_utils import xywh_to_xyxy, xyxy_to_xywh

class MIST():
    def __init__(self, config):
        # init detector
        self.detector = Detector(config).cuda()
        # init classifier
        feat_ch = self.detector.get_featuremap_channels()
        self.classifier = ClassifierNetWork(config,feat_ch).cuda()

    def save_state_dict(self, path):
        torch.save({'detector': self.detector.state_dict(), 
                    'classifier': self.classifier.state_dict()}, path)

    def load_state_dict(self, model_dict):
        self.detector.load_state_dict(model_dict['detector'])
        self.classifier.load_state_dict(model_dict['classifier'])

    def forward(self, image):
        # run detection network
        bbox_dt, detector_diag = self.detector.forward(image)
        # run task network
        labels, logits, _ = self.classifier.forward(detector_diag['featuremap'], bbox_dt.detach())
        # convert bbox to image scale
        bbox_img = bbox_dt * detector_diag['stride']

        return bbox_img, labels, logits



