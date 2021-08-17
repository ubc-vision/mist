import torch
import argparse
import numpy as np
import random
import os
import time
import math
from tqdm import tqdm

from utils.summary import CustomSummaryWriter
from datasets.datasets import get_dataset
from models.mist import MIST
from utils.config_utils import print_config, str2bool, get_mist_config, config_to_string
from utils.torch_utils import to_gpu, eval_accuracy, inverse_heatmap_gaussian, inverse_heatmap, xywh_to_xyxy
from utils.loss_functions import one_hot_classification_loss
from utils.utils import tensorboard_scheduler
from utils.viz_utils import save_bbox_images



class MISTTrainer():
    def __init__(self, config, mist):

        # init tensorboard scheduler
        self.scheduler = tensorboard_scheduler(config.summary_period,config.
                                               save_period,config.valid_period,
                                               config.cooldown_period)
        # copy config
        self.config = config

        # init iteration counter
        self.num_iter = 0
        self.best_val_f1 = 0
        
        # init summary writer
        self.summary = CustomSummaryWriter(config.log_dir + '/' + config.name)

        # create the solvers
        self.detector_solver = torch.optim.Adam(mist.detector.parameters(), 
                                                lr=config.lr_detector)
        self.task_solver = torch.optim.Adam(mist.classifier.parameters(), 
                                            lr=config.lr_task)

        # reume flag
        self.resumed = False

        # contains background class or not
        if config.dataset.startswith('mnist'):
            self.background_class=False
            self.val_metric = 'iou'
        else:
            self.background_class=True
            self.val_metric = 'center'

        # create validation path
        self.val_results_path = os.path.join(config.val_path, config.name)
        if not os.path.isdir(self.val_results_path):
            os.makedirs(self.val_results_path)
        if not os.path.isdir(self.config.model_dir):
            os.makedirs(self.config.model_dir)



    def resume(self, mist):
        try:
            print('Loading saved models ...')
            mist.load_state_dict(torch.load(os.path.join(
                self.config.model_dir,self.config.name+'_models')))
            print('Loading saved solvers ...')
            self.load_solvers()
            print('Previous state resumed, continue training')
            self.resumed = True
        except:
            print('Did not find saved model, fresh start')
            self.resumed = False
    
    def write_meta_data(self):
        # add hyper parameters to summary 
        self.summary.add_text('hyper paramter',config_to_string(self.config))
        # add comment to summary
        self.summary.add_text('comment',self.config.comment)

    def save_solvers(self):
        torch.save({ 
            'detector_solver': self.detector_solver.state_dict(),
            'task_solver': self.task_solver.state_dict(),
            'iteration': self.num_iter,
            'best_val_f1': self.best_val_f1
            }, os.path.join(self.config.model_dir, 
                            self.config.name + '_solvers'))         

    def load_solvers(self):
        checkpoint = torch.load(os.path.join(self.config.model_dir, 
                                             self.config.name + '_solvers'))
        if hasattr(self,'solver'):
            self.solver.load_state_dict(checkpoint['solver'])
        else:
            self.detector_solver.load_state_dict(checkpoint['detector_solver'])
            self.task_solver.load_state_dict(checkpoint['task_solver'])
        self.num_iter = checkpoint['iteration']
        self.best_val_f1 = checkpoint['best_val_f1']
        print('continue at {} iter, f1: {}'.format(self.num_iter,
                                                   self.best_val_f1))

    def train_kp(self, mist, featuremap, bbox, labels_gt):
        # only regress bbox location, fix width and height
        bbox_xy = bbox[:,:,:2].clone().detach().requires_grad_(True)
        bbox_wh = bbox[:,:,2:].clone().detach().requires_grad_(False)
        bbox = torch.cat([bbox_xy,bbox_wh],dim=-1)
        bbox_solver = torch.optim.SGD([bbox_xy], lr=self.config.lr_k)
        
        # training loop
        for k in range(self.config.k_iter):
            bbox_solver.zero_grad()
            _, logits, _ = mist.classifier.forward(featuremap, bbox)
            loss, _= one_hot_classification_loss(logits, labels_gt, 
                self.config.num_classes, self.config.loss_type)
            loss.backward() 
            bbox_solver.step()  

            bbox = torch.cat([bbox_xy,bbox_wh],dim=-1)
    
        kp_diag = {}
        kp_diag['bbox_grads'] = bbox_xy.grad
        return bbox.detach(),  kp_diag


    def train_task(self, mist,featuremap,bbox_opt, labels_gt):
        labels, logits, diag = mist.classifier.forward(featuremap, bbox_opt)
        # clear grad in solver
        self.task_solver.zero_grad()
        # calculate loss
        loss, loss_diag = one_hot_classification_loss(logits, labels_gt, 
            self.config.num_classes, self.config.loss_type)
        # back propagate
        loss.backward()
        # update weights
        self.task_solver.step()
        task_diag = {}
        task_diag['loss_per_sample'] = loss_diag['loss_per_sample']
        task_diag['loss_task'] = loss
        task_diag['classifier'] = diag
        task_diag['labels'] = labels
        return loss, task_diag

    def train_detector(self, heatmap, bbox, loss_per_sample):
        diag = {}

        B,C,H,W = heatmap.shape
        K = bbox.shape[1]

        if self.config.sub_pixel_kp:
            # get optimized offset temp gt
            offset_gt = bbox[:,:,:2] - torch.floor(bbox[:,:,:2])
            # cast bbox location to int
            bbox_int = torch.floor(bbox)
            # clamp bbox location within image
            bbox_int[:,:,0] = torch.clamp(bbox_int[:,:,0], min=0 ,max=W-1)
            bbox_int[:,:,1] = torch.clamp(bbox_int[:,:,1], min=0 ,max=H-1)
            # extract offset at bbox locations from heatmap
            idx = torch.cat([torch.arange(B).unsqueeze(-1).unsqueeze(-1).unsqueeze(-1).repeat(1,K,2,1).type(torch.long),
                             torch.arange(1,3).reshape(1,1,2,1).repeat(B,K,1,1),
                             bbox_int[:,:,[1,0]].type(torch.long).unsqueeze(-2).repeat(1,1,2,1)],dim=-1).permute(3,0,1,2)
            offset = heatmap[tuple(idx)]
            # offset loss
            loss_offset = 0.1*torch.nn.SmoothL1Loss()(offset,offset_gt)  
            diag['offset_gt'] = offset_gt
            diag['offset'] = offset
        else:
            loss_offset = 0
        diag['loss_offset'] = loss_offset

        
        # reconstruct heatmap
        if self.config.heatmap_reconstruct=='gaussian':
            bbox = torch.cat([torch.floor(bbox[:,:,:2]),bbox[:,:,2:]],dim=-1)
            target_heatmap = inverse_heatmap_gaussian(bbox, heatmap[:,[0],:,:].shape)
        elif self.config.heatmap_reconstruct=='single_point':
            target_heatmap = inverse_heatmap(bbox[:,:,:2], heatmap[:,[0],:,:].shape)
        else:
            raise RunTimeError('Unknown method ({}) for heatmap reconsturction'.format(self.heatmap_reconstruct))
        diag["target_heatmap"] = target_heatmap
        
        # reconstruction loss
        diff_per_sample = (heatmap[:,[0],:,:] - target_heatmap).pow(2).mean(dim=[1,2,3])
        sample_weight = torch.exp(-5.0 * loss_per_sample.detach())
        loss_heatmap = (sample_weight * diff_per_sample).mean()
        diag['loss_heatmap'] = loss_heatmap

        # sum of losses
        loss = loss_offset + loss_heatmap

        # back propogate
        self.detector_solver.zero_grad()
        loss.backward()
        self.detector_solver.step()
        
        return loss, diag


    def train(self, mist, dataset_tr, metadata_tr, dataset_va, metadata_va):
        # main training loop
        for epoch in tqdm(range(self.config.epochs)):
            for x in tqdm(dataset_tr, smoothing=0.1):
                # put data on gpu
                images, bbox_gt, labels_gt = to_gpu(x)

                # sanity check on data format
                if len(images.shape)!=4 or images.shape[1]!=3:
                    raise RunTimeError('Images does not have dimension (B,3,H,W)')

                # forward path on detector
                bbox_dt, detector_diag = mist.detector.forward(images)
                labels_dt, logits, _ = mist.classifier.forward(detector_diag['featuremap'], bbox_dt)

                # optimize bbox
                bbox_opt, train_kp_diag = self.train_kp(mist, detector_diag['featuremap'], bbox_dt, labels_gt)                  
                
                # optimize the task network
                loss_task,  train_task_diag= self.train_task(mist,detector_diag['featuremap'],bbox_opt, labels_gt)

                # optimize the detector
                loss_heatmap, train_detect_diag = self.train_detector( detector_diag['heatmap'], bbox_opt, train_task_diag['loss_per_sample'])   

                eval_flag, save_flag, valid_flag = self.scheduler.schedule()


                if eval_flag:
                    bbox_img = bbox_dt.detach() * detector_diag['stride']
                    labels_nms  = labels_dt.detach()
                    eval_batch = eval_accuracy (bbox_img, bbox_gt, labels_nms, labels_gt, self.config.num_classes, self.background_class)        

                    # loss
                    self.summary.add_scalar('loss offset', train_detect_diag['loss_offset'], self.num_iter)
                    self.summary.add_scalar('loss heatmap', train_detect_diag['loss_heatmap'], self.num_iter)
                    self.summary.add_scalar('loss heatmap + offset', loss_heatmap, self.num_iter)
                    self.summary.add_scalar('loss task', loss_task, self.num_iter)


                    # input image
                    self.summary.add_images('1 input', images, self.num_iter,resize=2)

                    # input image with predicted bbox  
                    self.summary.add_images('2 boxes', images, self.num_iter, 
                                       boxes_infer=xywh_to_xyxy(bbox_img), 
                                       boxes_gt=xywh_to_xyxy(bbox_gt), 
                                       labels=labels_nms,
                                       match=eval_batch['keypoint_match_detection'],
                                       resize=2)

                    # keypoints displacement
                    displacement = torch.cat([inverse_heatmap(bbox_dt.clone().detach(), [detector_diag['featuremap'].shape[0],1 ,detector_diag['featuremap'].shape[2], detector_diag['featuremap'].shape[3]]), 
                                              inverse_heatmap(bbox_opt.clone().detach(), [detector_diag['featuremap'].shape[0],1 ,detector_diag['featuremap'].shape[2], detector_diag['featuremap'].shape[3]]),
                                              detector_diag['featuremap'].mean(dim=1, keepdim=True)], 
                                              dim=1)
                    self.summary.add_images('3 displacement', displacement, self.num_iter, resize=2)          
                   
                    # patches for task network
                    self.summary.add_images('4 patches', train_task_diag['classifier']['patches'], self.num_iter)

                    # heatmap
                    self.summary.add_images('5 heatmap', detector_diag['heatmap'][:,[0],:,:], self.num_iter)

                    # target heatmap
                    self.summary.add_images('6 target_heatmap', torch.clamp(train_detect_diag['target_heatmap'],0,1), self.num_iter, resize=2)

                    # labels
                    self.summary.add_histogram('labels', labels_nms.view(-1), self.num_iter)
                    
                    # offset
                    if 'offset_gt' in train_detect_diag.keys() and 'offset' in train_detect_diag.keys():
                        self.summary.add_histogram('offset gt', train_detect_diag['offset_gt'].view(-1), self.num_iter)
                        self.summary.add_histogram('offset', train_detect_diag['offset'].view(-1), self.num_iter)

                    # accuracy
                    self.summary.add_scalar('Batch f1 center', eval_batch['f1_center'], self.num_iter)
                    self.summary.add_scalar('Batch f1 iou', eval_batch['f1_iou'], self.num_iter)
                    self.summary.add_scalar('Batch precision center', eval_batch['precision_center'], self.num_iter)
                    self.summary.add_scalar('Batch precision iou', eval_batch['precision_iou'], self.num_iter)
                    self.summary.add_scalar('Batch recall center', eval_batch['recall_center'], self.num_iter)
                    self.summary.add_scalar('Batch recall iou', eval_batch['recall_iou'], self.num_iter)
                    self.summary.add_scalar('Batch AP detection', eval_batch['acc_det'], self.num_iter)
                    self.summary.add_scalar('Batch AP classification', eval_batch['acc_class'], self.num_iter)
                    

                    self.summary.flush()

                if valid_flag and self.config.run_val:
                    eval_val = self.validate(mist, dataset_va,metadata_va)
                    if eval_val['num_detetcions'] == 0:
                        precision = 0
                    else:
                        precision = eval_val['tp_'+self.val_metric]/eval_val['num_detetcions']
                    recall = eval_val['tp_'+self.val_metric]/eval_val['num_objects']
                    if (precision + recall) == 0:
                        f1 = 0
                    else:
                        f1 = 2*precision*recall/(precision+recall)
                    print('validation set f1 {} score: {}'.format(self.val_metric, f1))
                    if self.best_val_f1 < f1:
                        mist.save_state_dict(os.path.join(self.config.model_dir, self.config.name + '_best_models'))
                        self.best_val_f1  = f1
                        print('savin best model f1 score: {}'.format(self.best_val_f1))

                if save_flag:
                    if self.config.save_weights:
                        # save network weights
                        print('saving wieghts ...')
                        mist.save_state_dict(os.path.join(self.config.model_dir, self.config.name + '_models'))
                        # save optimizer params
                        self.save_solvers()

                self.num_iter += 1



    def validate(self, mist, dataset,metadata_va, mode='val'):
        output = {}
        output['tp_center'] = 0
        output['tp_iou'] = 0
        output['num_objects'] = 0
        output['num_detetcions'] = 0

        # only run validation on 50 mini batches to save time
        max_iter = 50
        print('Running on validation set up to 50 mini batches')
        for i, x in tqdm(enumerate(dataset), smoothing=0.1):
            images, keypoints_gt, labels_gt = to_gpu(x)
            B, _, _, _ = images.shape
            bboxs, labels, _ = mist.forward(images.clone())
            eval_val = eval_accuracy(bboxs, keypoints_gt, labels, labels_gt, self.config.num_classes, self.background_class)
            for j,(image, bbox, label) in enumerate(zip(images,xywh_to_xyxy(bboxs),labels)):
                save_bbox_images(image,bbox,[metadata_va.get_class_name(_label) for _label in label], str(i*B+j),self.val_results_path,self.background_class)

            # accumulate results
            output['tp_center'] += eval_val['tp_center']
            output['tp_iou'] += eval_val['tp_iou']
            output['num_objects'] += eval_val['num_objects']
            output['num_detetcions'] += eval_val['num_detetcions']
            if i > max_iter:
                break
        return output




def main():

    config = get_mist_config()
    print_config(config)

    # set torch seed
    if config.set_seed:
        torch.manual_seed(0)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False   

    torch.set_default_tensor_type(torch.cuda.FloatTensor)
    torch.set_printoptions(profile="full")
    torch.set_printoptions(threshold=5000)
    torch.set_printoptions(precision=10)

    # init data loader
    dataset_tr, metadata_tr = get_dataset(config, mode='train')

    val_config = config
    val_config.rand_horiz_flip = False
    val_config.rand_maskout = False 
    dataset_va, metadata_va = get_dataset(val_config, mode='valid') 

    # init network
    mist = MIST(config)

    # init network trainer
    mist_trainer = MISTTrainer(config, mist)
    
    # resume model
    if config.resume:
        mist_trainer.resume(mist)

    # wirte meta data if first time run
    if not mist_trainer.resumed:
        mist_trainer.write_meta_data()

    # train model
    mist_trainer.train(mist, dataset_tr, metadata_tr, dataset_va, metadata_va)


if __name__ == '__main__':
    main()
    
