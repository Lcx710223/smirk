import torch.utils.data
import torch.nn.functional as F
import torch.nn as nn
import numpy as np
from torchvision.utils import make_grid
import cv2
import os
import random
import copy
from src.utils.utils import batch_draw_keypoints, make_grid_from_opencv_images

class BaseTrainer(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config

    def logging(self, batch_idx, losses, phase):
        if self.config.train.log_losses_every > 0 and batch_idx % self.config.train.log_losses_every == 0:
            loss_str = ''
            for k, v in losses.items():
                loss_str += f'{k}: {v:.6f} '
            print(loss_str)

    def configure_optimizers(self, n_steps):
        self.n_steps = n_steps
        encoder_scale = .25

        if hasattr(self, 'encoder_optimizer'):
            for g in self.encoder_optimizer.param_groups:
                g['lr'] = encoder_scale * self.config.train.lr
        else:
            params = []
            if self.config.train.optimize_expression:
                params += list(self.smirk_encoder.expression_encoder.parameters()) 
            if self.config.train.optimize_shape:
                params += list(self.smirk_encoder.shape_encoder.parameters())
            if self.config.train.optimize_pose:
                params += list(self.smirk_encoder.pose_encoder.parameters())

            self.encoder_optimizer = torch.optim.Adam(params, lr= encoder_scale * self.config.train.lr)
                
        self.encoder_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.encoder_optimizer, T_max=n_steps,
            eta_min=0.01 * encoder_scale * self.config.train.lr
        )

        if self.config.arch.enable_fuse_generator:
            if hasattr(self, 'fuse_generator_optimizer'):
                for g in self.smirk_generator_optimizer.param_groups:
                    g['lr'] = self.config.train.lr
            else:
                self.smirk_generator_optimizer = torch.optim.Adam(
                    self.smirk_generator.parameters(),
                    lr=self.config.train.lr, betas=(0.5, 0.999)
                )
            
            self.smirk_generator_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                self.smirk_generator_optimizer, T_max=n_steps,
                eta_min=0.01 * self.config.train.lr
            )

    def load_random_template(self, num_expressions=50):
        random_key = random.choice(list(self.templates.keys()))
        templates = self.templates[random_key]
        random_index = random.randint(0, templates.shape[0]-1)
        return templates[random_index][:num_expressions]

    def setup_losses(self):
        from src.losses.VGGPerceptualLoss import VGGPerceptualLoss
        self.vgg_loss = VGGPerceptualLoss()
        self.vgg_loss.eval()
        for param in self.vgg_loss.parameters():
            param.requires_grad_(False)
        
        if self.config.train.loss_weights['emotion_loss'] > 0:
            from src.losses.ExpressionLoss import ExpressionLoss
            self.emotion_loss = ExpressionLoss()
            self.emotion_loss.eval()
            for param in self.emotion_loss.parameters():
                param.requires_grad_(False)

        if self.config.train.loss_weights['mica_loss'] > 0:
            from src.models.MICA.mica import MICA
            self.mica = MICA()
            self.mica.eval()
            for param in self.mica.parameters():
                param.requires_grad_(False)

    def scheduler_step(self):
        self.encoder_scheduler.step()
        if self.config.arch.enable_fuse_generator:
            self.smirk_generator_scheduler.step()

    def train(self):
        self.smirk_encoder.train()
        if self.config.arch.enable_fuse_generator:
            self.smirk_generator.train()
    
    def eval(self):
        self.smirk_encoder.eval()
        if self.config.arch.enable_fuse_generator:
            self.smirk_generator.eval()

    def optimizers_zero_grad(self):
        self.encoder_optimizer.zero_grad()
        if self.config.arch.enable_fuse_generator:
            self.smirk_generator_optimizer.zero_grad()

    def optimizers_step(self, step_encoder=True, step_fuse_generator=True):
        if step_encoder:
            self.encoder_optimizer.step()
        if step_fuse_generator and self.config.arch.enable_fuse_generator:
            self.smirk_generator_optimizer.step()

    def save_visualizations(self, outputs, save_path, show_landmarks=False):
        nrow = 1
        
        if 'img' in outputs and 'rendered_img' in outputs and 'masked_1st_path' in outputs:
            outputs['overlap_image'] = outputs['img'] * 0.7 + outputs['rendered_img'] * 0.3
            outputs['overlap_image_pixels'] = outputs['img'] * 0.7 +  0.3 * outputs['masked_1st_path']
        
        if show_landmarks:
            original_img_with_landmarks = batch_draw_keypoints(outputs['img'], outputs['landmarks_mp'], color=(0,255,0))
            original_img_with_landmarks = batch_draw_keypoints(original_img_with_landmarks, outputs['landmarks_mp_gt'], color=(0,0,255))
            original_img_with_landmarks = batch_draw_keypoints(original_img_with_landmarks, outputs['landmarks_fan'][:,:17], color=(255,0,255))
            original_img_with_landmarks = batch_draw_keypoints(original_img_with_landmarks, outputs['landmarks_fan_gt'][:,:17], color=(255,255,255))
            original_grid = make_grid_from_opencv_images(original_img_with_landmarks, nrow=nrow)
        else:
            original_img_with_landmarks = outputs['img']
            original_grid = make_grid(original_img_with_landmarks, nrow=nrow)

        image_keys = ['img_mica', 'rendered_img_base', 'rendered_img', 
                      'overlap_image', 'overlap_image_pixels',
                      'rendered_img_mica_zero', 'rendered_img_zero', 
                      'masked_1st_path', 'reconstructed_img', 'loss_img', 
                      '2nd_path']
        
        nrows = [1 if '2nd_path' not in key else 4 * self.config.train.Ke for key in image_keys]

        # unify sizes before concatenation
        target_h = original_grid.shape[1]
        target_w = original_grid.shape[2]

        grids = [original_grid]
        for key, nr in zip(image_keys, nrows):
            if key in outputs.keys():
                g = make_grid(outputs[key].detach().cpu(), nrow=nr)
                if g.shape[1] != target_h or g.shape[2] != target_w:
                    g = F.interpolate(g.unsqueeze(0), size=(target_h, target_w), mode='bilinear', align_corners=False).squeeze(0)
                grids.append(g)

        grid = torch.cat(grids, dim=2)
            
        grid = grid.permute(1,2,0).cpu().numpy()*255.0
        grid = np.clip(grid, 0, 255)
        grid = grid.astype(np.uint8)
        grid = cv2.cvtColor(grid, cv2.COLOR_RGB2BGR)

        cv2.imwrite(save_path, grid)

    def create_visualizations(self, batch, outputs):
        # 零姿态相机参数
        zero_pose_cam = torch.tensor([7, 0, 0]).unsqueeze(0).repeat(batch['img'].shape[0], 1).float().to(self.config.device)

        visualizations = {}

        # 输入图像
        if 'img' in batch:
            visualizations['img'] = batch['img']

        # 渲染图像
        if outputs is not None:
            if 'landmarks_mp' in outputs:
                visualizations['landmarks_mp'] = outputs['landmarks_mp']
            if 'landmarks_mp_gt' in outputs:
                visualizations['landmarks_mp_gt'] = outputs['landmarks_mp_gt']
            if 'landmarks_fan' in outputs:
                visualizations['landmarks_fan'] = outputs['landmarks_fan']
            if 'landmarks_fan_gt' in outputs:
                visualizations['landmarks_fan_gt'] = outputs['landmarks_fan_gt']


        # 基准渲染
        if hasattr(self, "base_encoder") and self.base_encoder is not None:
            base_output = self.base_encoder(batch['img'])
            flame_output_base = self.flame.forward(base_output)
            rendered_img_base = self.renderer.forward(flame_output_base['vertices'], base_output['cam'])['rendered_img']
            visualizations['rendered_img_base'] = rendered_img_base

        # 零姿态渲染
        if outputs is not None and 'encoder_output' in outputs:
            flame_output_zero = self.flame.forward(outputs['encoder_output'], zero_expression=True, zero_pose=True)
            rendered_img_zero = self.renderer.forward(flame_output_zero['vertices'].to(self.config.device), zero_pose_cam)['rendered_img']
            visualizations['rendered_img_zero'] = rendered_img_zero

        # 生成器相关可视化
        if self.config.arch.enable_fuse_generator and outputs is not None:
            if 'reconstructed_img' in outputs:
                visualizations['reconstructed_img'] = outputs['reconstructed_img']
            if 'masked_1st_path' in outputs:
                visualizations['masked_1st_path'] = outputs['masked_1st_path']
            if 'loss_img' in outputs:
                visualizations['loss_img'] = outputs['loss_img']

        # detach/cpu 安全处理
        for key, value in list(visualizations.items()):
            if isinstance(value, torch.Tensor):
                visualizations[key] = value.detach().cpu()

        # MICA 可视化
        if self.config.train.loss_weights.get('mica_loss', 0) > 0 and hasattr(self, "mica"):
            mica_output_shape = self.mica(batch['img_mica'])
            mica_output = copy.deepcopy(base_output)
            mica_output['shape_params'] = mica_output_shape['shape_params']

            if self.config.arch.num_shape < 300:
                mica_output['shape_params'] = mica_output['shape_params'][:, :self.config.arch.num_shape]

            flame_output_mica = self.flame.forward(mica_output, zero_expression=True, zero_pose=True)
            rendered_img_mica_zero = self.renderer.forward(flame_output_mica['vertices'], zero_pose_cam)['rendered_img']
            visualizations['rendered_img_mica_zero'] = rendered_img_mica_zero

            # 补全 img_mica
            if 'img_mica' in batch:
                visualizations['img_mica'] = batch['img_mica'].detach().cpu()

        return visualizations


    def set_freeze_status(self, config, batch_idx, epoch_idx):
        # 默认不冻结
        self.config.train.freeze_encoder_in_second_path = False
        self.config.train.freeze_generator_in_second_path = False

        # 使用 batch_idx 控制交替冻结策略
        decision_idx_second_path = batch_idx
        self.config.train.freeze_encoder_in_second_path = (decision_idx_second_path % 2 == 0)
        self.config.train.freeze_generator_in_second_path = (decision_idx_second_path % 2 == 1)


    def save_model(self, state_dict, save_path):
        # 只保存 smirk_encoder 和 smirk_generator 的参数
        new_state_dict = {}
        for key in list(state_dict.keys()):
            if key.startswith('smirk_encoder') or key.startswith('smirk_generator'):
                new_state_dict[key] = state_dict[key]
        torch.save(new_state_dict, save_path)
