import importlib
import torch
from collections import OrderedDict
from copy import deepcopy
from os import path as osp
from tqdm import tqdm
import logging

from basicsr.models.archs.EnhancedEFNet import EnhancedEFNet  # Import your model
from basicsr.models.base_model import BaseModel
from basicsr.utils import get_root_logger, imwrite, tensor2img, get_model_flops

loss_module = importlib.import_module('basicsr.models.losses')
metric_module = importlib.import_module('basicsr.metrics')
logger = logging.getLogger('basicsr')


class ImageEventRestorationModel(BaseModel):
    """Event-based image deblurring model for single-image restoration."""

    def __init__(self, opt):
        super(ImageEventRestorationModel, self).__init__(opt)

        # Define network
        self.net_g = EnhancedEFNet()
        self.net_g = self.model_to_device(self.net_g)
        self.print_network(self.net_g)

        # Compute FLOPs if enabled
        if self.opt.get('print_flops', False):
            input_dim = [(3, 256, 256), (6, 256, 256)]
            flops = get_model_flops(self.net_g, input_dim, False) / 10**9
            logger.info(f"FLOPs: {flops:.4f} G")

        # Load pretrained model if specified
        load_path = self.opt['path'].get('pretrain_network_g', None)
        if load_path:
            self.load_network(self.net_g, load_path, self.opt['path'].get('strict_load_g', True))

        if self.is_train:
            self.init_training_settings()

    def init_training_settings(self):
        self.net_g.train()
        train_opt = self.opt['train']

        # Define loss function
        self.cri_pix = None
        if train_opt.get('pixel_opt'):
            loss_type = train_opt['pixel_opt'].pop('type')
            self.cri_pix = getattr(loss_module, loss_type)(**train_opt['pixel_opt']).to(self.device)
        
        if self.cri_pix is None:
            raise ValueError("Loss function is not defined.")

        # Setup optimizer and scheduler
        self.setup_optimizers()
        self.setup_schedulers()

    def setup_optimizers(self):
        train_opt = self.opt['train']
        optim_params = [v for k, v in self.net_g.named_parameters() if v.requires_grad]
        optim_type = train_opt['optim_g'].pop('type')

        if optim_type == 'Adam':
            self.optimizer_g = torch.optim.Adam(optim_params, **train_opt['optim_g'])
        elif optim_type == 'AdamW':
            self.optimizer_g = torch.optim.AdamW(optim_params, **train_opt['optim_g'])
        else:
            raise NotImplementedError(f'Optimizer {optim_type} is not supported.')

        self.optimizers.append(self.optimizer_g)

    def feed_data(self, data):
        self.lq = data['frame'].to(self.device)
        self.voxel = data['voxel'].to(self.device)
        self.gt = data.get('frame_gt', None)

    def optimize_parameters(self, current_iter):
        self.optimizer_g.zero_grad()
        self.output = self.net_g(self.lq, self.voxel)

        loss_dict = OrderedDict()
        l_total = self.cri_pix(self.output, self.gt)
        loss_dict['l_pix'] = l_total

        l_total.backward()
        self.optimizer_g.step()
        self.log_dict = self.reduce_loss_dict(loss_dict)

    def test(self):
        self.net_g.eval()
        with torch.no_grad():
            self.output = self.net_g(self.lq, self.voxel)
        self.net_g.train()

    def nondist_validation(self, dataloader, current_iter, tb_logger, save_img, rgb2bgr, use_image):
        dataset_name = self.opt.get('name')
        with_metrics = self.opt['val'].get('metrics') is not None
        if with_metrics:
            self.metric_results = {metric: 0 for metric in self.opt['val']['metrics'].keys()}

        pbar = tqdm(total=len(dataloader), unit='image')
        for idx, val_data in enumerate(dataloader):
            self.feed_data(val_data)
            self.test()
            visuals = self.get_current_visuals()
            sr_img = tensor2img([visuals['result']], rgb2bgr=rgb2bgr)

            if save_img:
                save_img_path = osp.join(self.opt['path']['visualization'], dataset_name, f'{idx}.png')
                imwrite(sr_img, save_img_path)

            pbar.update(1)
        pbar.close()

    def get_current_visuals(self):
        return {'lq': self.lq.cpu(), 'result': self.output.cpu(), 'gt': self.gt.cpu() if self.gt is not None else None}

    def save(self, epoch, current_iter):
        self.save_network(self.net_g, 'net_g', current_iter)
        self.save_training_state(epoch, current_iter)
