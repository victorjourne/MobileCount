import pdb
import torch
import torch.nn as nn
import torch.nn.functional as F


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
        self.debug=False
        if self.GPU_OK:
            if len(gpus) > 1:
                self.CCN = torch.nn.DataParallel(self.CCN, device_ids=gpus).cuda()
            else:
                self.CCN = self.CCN.cuda()
            #self.loss_mse_fn = nn.MSELoss().cuda()
            self.loss_mse_fn = nn.MSELoss(reduction='none').cuda()
        else:
            #self.loss_mse_fn = nn.MSELoss()
            self.loss_mse_fn = nn.MSELoss(reduction='none')
    
    def compute_lc_loss(self, output, target, sizes=(1, 2, 4)):
        #criterion_L1 = torch.nn.L1Loss(reduction=self.cfg.L1_LOSS_REDUCTION)
        criterion_L1 = torch.nn.L1Loss(reduction='none')
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
            if self.debug:
                print('c(raw):', c.size())
            #Validation [1, 2, 2] ou [1, 4, 4] ou [1, 8, 8]
            #Train [1, 16, 2, 2] ou [1, 16, 4, 4] ou [1, 16, 8, 8]
            if len(list(c.squeeze().size()))==3:
                c_mean = torch.mean(c.squeeze(), dim=(1,2)) / s**2
            else:
                c_mean = torch.mean(c.squeeze()) / s**2
            if self.debug:
                print('c(mean):', c_mean.size(), c_mean)
            #Validation = constant
            #Train [16]
            if lc_loss is not None:
                lc_loss += c_mean 
                #lc_loss += criterion_L1(est, gt) / s**2
            else:
                lc_loss = c_mean
                #lc_loss = criterion_L1(est, gt) / s**2
        return lc_loss

    @property
    def loss(self):
        return self.loss_mse

    def f_loss(self):
        return self.loss_mse

    def forward(self, img, gt_map=None, sample_weight=None):
        density_map = self.CCN(img)
        if gt_map is not None:
            self.loss_mse = self.build_loss(density_map.squeeze(), gt_map.squeeze(), sample_weight)
        return density_map

    def build_loss(self, density_map, gt_data, sample_weight=None):
        loss_mse = self.loss_mse_fn(density_map, gt_data)
        if self.debug:
            print('loss_mse(raw):', loss_mse.size())
        #Validation [642, 1000]
        #Train [16, 576, 768]
        if len(list(loss_mse.size()))==3:
            loss_mse = torch.mean(loss_mse, dim=(1,2))
        else:
            loss_mse = torch.mean(loss_mse)
        if self.debug:
            print('loss_mse(mean):', loss_mse.size(), loss_mse)
        #Validation = constant
        #Train [16]
        self.lc_loss = 0
        if self.cfg.CUSTOM_LOSS:
            lc_loss = self.compute_lc_loss(density_map, gt_data, sizes=self.cfg.CUSTOM_LOSS_SIZES)
            if self.debug:
                print('lc_loss:', lc_loss.size(), lc_loss)
            #Validation = constant
            #Train [16]
            self.lc_loss = torch.sum(lc_loss) #just for display
            #loss_mse = loss_mse + (self.cfg.CUSTOM_LOSS_LAMBDA * lc_loss)
            lc_loss = torch.mul(lc_loss, self.cfg.CUSTOM_LOSS_LAMBDA)
            if self.debug:
                print('lc_loss(lambda):', lc_loss.size(), lc_loss)
            #Validation = constant
            #Train [16]
            loss_mse = loss_mse + lc_loss
        if self.debug:
            print('loss_mse(mean+lc):', loss_mse.shape, loss_mse)
        #Validation = constant
        #Train [16]
        if sample_weight is not None:
            if self.debug:
                print('sample_weight:', sample_weight.size(), sample_weight)
            a = torch.mul(loss_mse,sample_weight)
            if self.debug:
                print('a:', a.size(), a)            
            b = torch.sum(a)
            if self.debug:
                print('b:', b.size(), b)
            c = torch.sum(sample_weight)
            if self.debug:
                print('c:', c.size(), c)
            loss_mse = b/c
        else:
            loss_mse =  torch.mean(loss_mse)
        if self.debug:
            print('loss_mse(calculated):', loss_mse.shape, loss_mse)
        #Validation= constant
        #Train [16]
        #loss_mse =  torch.sum(loss_mse)
        #if self.debug:
        #    print('loss_mse(final mean):', loss_mse.shape, loss_mse)
        #Validation & train = constant
        return loss_mse

    def test_forward(self, img):
        density_map = self.CCN(img)
        return density_map
