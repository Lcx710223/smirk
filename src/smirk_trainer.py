import torch
import torch.utils.data
import torch.nn.functional as F
from src.FLAME.FLAME import FLAME
from src.renderer.renderer import Renderer
from src.smirk_encoder import SmirkEncoder
from src.smirk_generator import SmirkGenerator
from src.base_trainer import BaseTrainer
import numpy as np
import src.utils.utils as utils
import src.utils.masking as masking_utils
import copy

class SmirkTrainer(BaseTrainer):
    def __init__(self, config):
        super().__init__(config)   # 继承 BaseTrainer，获得通用方法

        self.config = config
        # 确保属性存在，train.py 会在创建后通过 try_create_base_encoder(trainer) 填充
        self.base_encoder = None

        if self.config.arch.enable_fuse_generator:
            self.smirk_generator = SmirkGenerator(in_channels=6, out_channels=3, init_features=32, res_blocks=5)

        self.smirk_encoder = SmirkEncoder(n_exp=self.config.arch.num_expression, n_shape=self.config.arch.num_shape)
        self.flame = FLAME(n_exp=self.config.arch.num_expression, n_shape=self.config.arch.num_shape)
        self.renderer = Renderer(render_full_head=False)

        # 初始化损失函数等通用组件
        self.setup_losses()

        self.templates = utils.load_templates()

        # --------- setup flame masks for sampling --------- #
        self.face_probabilities = masking_utils.load_probabilities_per_FLAME_triangle()

    # ---------------- first path ---------------- #
    def step1(self, batch):
        # 基本输入
        img = batch['img']
        B, C, H, W = img.shape

        # 编码器前向
        encoder_output = self.smirk_encoder(img)

        # 基准编码器输出（用于正则）
        use_base_reg = getattr(self.config.train, "use_base_model_for_regularization", False)
        if use_base_reg and (self.base_encoder is not None):
            with torch.no_grad():
                base_output = self.base_encoder(img)
        else:
            # 使用零张量占位，确保键完整
            device = self.config.device
            base_output = {
                'expression_params': torch.zeros(B, self.config.arch.num_expression, device=device),
                'shape_params': torch.zeros(B, self.config.arch.num_shape, device=device),
                'jaw_params': torch.zeros(B, 3, device=device),
            }

        # FLAME + 渲染
        flame_output = self.flame.forward(encoder_output)
        renderer_output = self.renderer.forward(
            flame_output['vertices'],
            encoder_output['cam'],
            landmarks_fan=flame_output['landmarks_fan'],
            landmarks_mp=flame_output['landmarks_mp']
        )
        rendered_img = renderer_output['rendered_img']
        flame_output.update(renderer_output)

        # ---------------- losses ---------------- #
        losses = {}

        # landmark losses
        valid_landmarks = batch['flag_landmarks_fan']
        losses['landmark_loss_fan'] = 0 if torch.sum(valid_landmarks) == 0 else F.mse_loss(
            flame_output['landmarks_fan'][valid_landmarks, :17],
            batch['landmarks_fan'][valid_landmarks, :17]
        )
        losses['landmark_loss_mp'] = F.mse_loss(flame_output['landmarks_mp'], batch['landmarks_mp'])

        # regularization losses
        losses['expression_regularization'] = torch.mean(
            (encoder_output['expression_params'] - base_output['expression_params']) ** 2
        )
        losses['shape_regularization'] = torch.mean(
            (encoder_output['shape_params'] - base_output['shape_params']) ** 2
        )
        losses['jaw_regularization'] = torch.mean(
            (encoder_output['jaw_params'] - base_output['jaw_params']) ** 2
        )

        # 生成器路径（可选）
        loss_img = None
        reconstructed_img = None
        masked_img = None

        if self.config.arch.enable_fuse_generator:
            masks = batch['mask']
            # 根据渲染结果得到有效区域掩码
            rendered_mask = 1 - (rendered_img == 0).all(dim=1, keepdim=True).float()
            tmask_ratio = getattr(self.config.train, "mask_ratio", 0.0)

            npoints, _ = masking_utils.mesh_based_mask_uniform_faces(
                flame_output['transformed_vertices'],
                flame_faces=self.flame.faces_tensor,
                face_probabilities=self.face_probabilities,
                mask_ratio=tmask_ratio
            )

            # transfer + dilation
            extra_points = masking_utils.transfer_pixels(img, npoints, npoints)
            masked_img = masking_utils.masking(
                img, masks, extra_points,
                getattr(self.config.train, "mask_dilation_radius", 0),
                rendered_mask=rendered_mask
            )

            # 重建图像
            reconstructed_img = self.smirk_generator(torch.cat([rendered_img, masked_img], dim=1))
            reconstruction_loss = F.l1_loss(reconstructed_img, img, reduction='none')
            loss_img = reconstruction_loss.mean(dim=1, keepdim=True)
            losses['reconstruction_loss'] = reconstruction_loss.mean()
            losses['perceptual_vgg_loss'] = self.vgg_loss(reconstructed_img, img)

            # emotion loss（按权重启用）
            if self.config.train.loss_weights.get('emotion_loss', 0) > 0:
                # 冻结生成器做前向，随后恢复
                for param in self.smirk_generator.parameters():
                    param.requires_grad_(False)
                self.smirk_generator.eval()
                reconstructed_img_p = self.smirk_generator(torch.cat([rendered_img, masked_img], dim=1))
                for param in self.smirk_generator.parameters():
                    param.requires_grad_(True)
                self.smirk_generator.train()
                losses['emotion_loss'] = self.emotion_loss(
                    reconstructed_img_p, img, metric='l2', use_mean=False
                ).mean()
            else:
                losses['emotion_loss'] = torch.tensor(0.0, device=img.device)
        else:
            # 未启用生成器路径时，相关损失为 0
            losses['reconstruction_loss'] = torch.tensor(0.0, device=img.device)
            losses['perceptual_vgg_loss'] = torch.tensor(0.0, device=img.device)
            losses['emotion_loss'] = torch.tensor(0.0, device=img.device)

        # 组合加权损失
        shape_losses = losses['shape_regularization'] * self.config.train.loss_weights['shape_regularization'] + \
                       losses.get('mica_loss', torch.tensor(0.0, device=img.device)) * self.config.train.loss_weights['mica_loss']
        expression_losses = losses['expression_regularization'] * self.config.train.loss_weights['expression_regularization'] + \
                            losses['jaw_regularization'] * self.config.train.loss_weights['jaw_regularization']
        landmark_losses = losses['landmark_loss_fan'] * self.config.train.loss_weights['landmark_loss'] + \
                          losses['landmark_loss_mp'] * self.config.train.loss_weights['landmark_loss']
        fuse_generator_losses = losses['perceptual_vgg_loss'] * self.config.train.loss_weights['perceptual_vgg_loss'] + \
                                losses['reconstruction_loss'] * self.config.train.loss_weights['reconstruction_loss'] + \
                                losses['emotion_loss'] * self.config.train.loss_weights['emotion_loss']

        loss_first_path = (
            (shape_losses if self.config.train.optimize_shape else 0) +
            (expression_losses if self.config.train.optimize_expression else 0) +
            (landmark_losses) +
            (fuse_generator_losses if self.config.arch.enable_fuse_generator else 0)
        )

        # MICA 形状损失（按权重启用）
        if self.config.train.loss_weights.get('mica_loss', 0) > 0:
            losses['mica_loss'] = self.mica.calculate_mica_shape_loss(encoder_output['shape_params'], batch['img_mica'])
        else:
            losses['mica_loss'] = torch.tensor(0.0, device=img.device)

        # 将标量化便于日志记录
        for key, value in list(losses.items()):
            if isinstance(value, torch.Tensor):
                losses[key] = value.item()

        # 构造可视化/输出字段（尽量完整）
        outputs = {
            'img': img.detach().cpu(),
            'rendered_img': rendered_img.detach().cpu(),
            'vertices': flame_output['vertices'].detach().cpu(),
            'landmarks_fan_gt': batch['landmarks_fan'].detach().cpu(),
            'landmarks_fan': flame_output['landmarks_fan'].detach().cpu(),
            'landmarks_mp': flame_output['landmarks_mp'].detach().cpu(),
            'landmarks_mp_gt': batch['landmarks_mp'].detach().cpu(),
            'encoder_output': encoder_output  # 不 detach，外部可能继续使用
        }

        if self.config.arch.enable_fuse_generator:
            outputs['loss_img'] = loss_img.detach().cpu() if loss_img is not None else None
            outputs['reconstructed_img'] = reconstructed_img.detach().cpu() if reconstructed_img is not None else None
            outputs['masked_1st_path'] = masked_img.detach().cpu() if masked_img is not None else None

        return outputs, losses, loss_first_path, encoder_output

    # ---------------- second path ---------------- #
    def step2(self, encoder_output, batch, batch_idx, phase='train'):
        # 保持原有逻辑（此处留空或后续补充）
        pass

    def freeze_encoder(self):
        utils.freeze_module(self.smirk_encoder.pose_encoder, 'pose encoder')
        utils.freeze_module(self.smirk_encoder.shape_encoder, 'shape encoder')
        utils.freeze_module(self.smirk_encoder.expression_encoder, 'expression encoder')

    def unfreeze_encoder(self):
        if self.config.train.optimize_pose:
            utils.unfreeze_module(self.smirk_encoder.pose_encoder, 'pose encoder')
        if self.config.train.optimize_shape:
            utils.unfreeze_module(self.smirk_encoder.shape_encoder, 'shape encoder')
        if self.config.train.optimize_expression:
            utils.unfreeze_module(self.smirk_encoder.expression_encoder, 'expression encoder')

    def step(self, batch, batch_idx, phase='train'):
        # 训练/验证模式切换
        if phase == 'train':
            self.train()
            torch.set_grad_enabled(True)
        else:
            self.eval()
            torch.set_grad_enabled(False)

        # 主路径
        outputs1, losses1, loss_first_path, encoder_output = self.step1(batch)

        # 反向传播与优化
        if phase == 'train':
            self.optimizers_zero_grad()
            loss_first_path.backward()
            self.optimizers_step(step_encoder=True, step_fuse_generator=self.config.arch.enable_fuse_generator)

        # 可选第二路径（循环一致损失）
        if (self.config.train.loss_weights.get('cycle_loss', 0) > 0) and (phase == 'train'):
            if self.config.train.freeze_encoder_in_second_path:
                self.freeze_encoder()
            if self.config.train.freeze_generator_in_second_path and self.config.arch.enable_fuse_generator:
                utils.freeze_module(self.smirk_generator, 'smirk generator')
            # 如果你后续实现了 step2，这里可调用：
            # outputs2, losses2, loss_second_path = self.step2(encoder_output, batch, batch_idx, phase)
            # self.optimizers_zero_grad()
            # loss_second_path.backward()
            # self.optimizers_step(step_encoder=True, step_fuse_generator=self.config.arch.enable_fuse_generator)
            # 合并或返回需要可视化的路径输出

        # 确保返回的是用于可视化的字典（非 None）
        return outputs1
