import torch
import torch.nn as nn
import torch.nn.functional as F
import pdb

class CrowdCounter(nn.Module):
    def __init__(self, gpus, model_name, cfg=None):
        super(CrowdCounter, self).__init__()        
        self.GPU_OK = torch.cuda.is_available()
        if model_name == 'MobileCount':
            from models.MobileCount import MobileCount as net
        elif model_name == 'MobileCountx1_25':
            from models.MobileCountx1_25 import MobileCount as net
        elif model_name == 'MobileCountx2':
            from models.MobileCountx2 import MobileCount as net
        self.cfg = cfg
        self.CCN = net()
        if self.GPU_OK:
            if len(gpus)>1:
                self.CCN = torch.nn.DataParallel(self.CCN, device_ids=gpus).cuda()
            else:
                self.CCN = self.CCN.cuda()
            self.loss_mse_fn = nn.MSELoss().cuda()
            #self.loss_mse_fn = nn.MSELoss(reduction='none').cuda()
        else:
            self.loss_mse_fn = nn.MSELoss()
            #self.loss_mse_fn = nn.MSELoss(reduction='none')
    
    
    
    def compute_lc_loss(self, output, target, sizes=(1, 2, 4)):
        criterion_L1 = torch.nn.L1Loss(reduction=self.cfg.L1_LOSS_REDUCTION)
        #criterion_L1 = torch.nn.L1Loss(reduction='none')
        if self.GPU_OK:
            criterion_L1 = criterion_L1.cuda()
        lc_loss = None
        for s in sizes:
            pool = torch.nn.AdaptiveAvgPool2d(s)
            if self.GPU_OK:
                pool = pool.cuda()
            est = pool(output.unsqueeze(0))
            gt = pool(target.unsqueeze(0))
            c = criterion_L1(est, gt)
            #print('c:', c.shape, c)
            if lc_loss:
                lc_loss += c / s**2
                #lc_loss += criterion_L1(est, gt) / s**2
            else:
                lc_loss = c / s**2
                #lc_loss = criterion_L1(est, gt) / s**2
            
        return lc_loss
    
    @property
    def loss(self):
        return self.loss_mse

    def f_loss(self):
        return self.loss_mse
    
    def forward(self, img, gt_map=None, sample_weight=None):                               
        density_map = self.CCN(img)    
        if gt_map is not None : 
            self.loss_mse = self.build_loss(density_map.squeeze(), gt_map.squeeze(), sample_weight)     
        return density_map
    
    def build_loss(self, density_map, gt_data, sample_weight=None):
        loss_mse = self.loss_mse_fn(density_map, gt_data)
        #print('loss_mse:', loss_mse.shape, loss_mse)
        self.lc_loss = 0
        if self.cfg.CUSTOM_LOSS:
            lc_loss = self.compute_lc_loss(density_map, gt_data, sizes=self.cfg.CUSTOM_LOSS_SIZES)
            self.lc_loss = lc_loss
            loss_mse = loss_mse + (self.cfg.CUSTOM_LOSS_LAMBDA * lc_loss)
        #if sample_weight is not None:
        #    print('sample_weight:', sample_weight.shape, sample_weight)
        #    print('loss_mse:', self.loss_mse)
        #    self.loss_mse = torch.mul(self.loss_mse, sample_weight)
        #    print('loss_mse:',self.loss_mse)
        return loss_mse

    def test_forward(self, img):                               
        density_map = self.CCN(img)                    
        return density_map

