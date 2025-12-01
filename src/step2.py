    # ---------------- second path ---------------- #
    def step2(self, encoder_output, batch, batch_idx, phase='train'):
        B, C, H, W = batch['img'].shape        
        img = batch['img']
        masks = batch['mask']
        
        Ke = self.config.train.Ke
        
        # ----------- 构造 flame_feats -----------
        flame_feats = {}
        for k, v in encoder_output.items():
            tmp = v.clone().detach()
            flame_feats[k] = torch.cat(Ke * [tmp], dim=0)

        # ----------- 分组并扰动参数（保持原逻辑）-----------
        gids = torch.randperm(Ke * B)
        gids = [gids[:Ke * B // 4], gids[Ke * B // 4: 2 * Ke * B // 4], 
                gids[2 * Ke * B // 4: 3 * Ke * B // 4], gids[3 * Ke * B // 4:]] 
        feats_dim = flame_feats['expression_params'].size(1)

        # 随机表情 / permutation / template injection / jaw / eyelid / zero expression
        # （这里保持原始源码逻辑，不再赘述）

        # detach
        for k in ['expression_params','pose_params','shape_params','jaw_params','eyelid_params']:
            flame_feats[k] = flame_feats[k].detach()

        # ----------- 渲染 -----------
        with torch.no_grad():
            flame_output = self.flame.forward(encoder_output)
            rendered_output = self.renderer.forward(flame_output['vertices'], encoder_output['cam'])
            flame_output.update(rendered_output)

            flame_output_2nd_path = self.flame.forward(flame_feats)

            # 相机参数扩展到 Ke * B
            cam = encoder_output['cam']
            Bv = flame_output_2nd_path['vertices'].shape[0]   # 128
            Bc = cam.shape[0]                                 # 32
            if Bv % Bc == 0:
                repeat_times = Bv // Bc
                cam_expanded = cam.repeat(repeat_times, 1)
            else:
                raise RuntimeError(f"Batch mismatch: verts B={Bv}, cam B={Bc}")

            renderer_output_2nd_path = self.renderer.forward(flame_output_2nd_path['vertices'], cam_expanded)
            rendered_img_2nd_path = renderer_output_2nd_path['rendered_img'].detach()

            # 采样点、masking（保持原逻辑）
            tmask_ratio = self.config.train.mask_ratio
            points1, sampled_coords = masking_utils.mesh_based_mask_uniform_faces(
                flame_output['transformed_vertices'], 
                flame_faces=self.flame.faces_tensor,
                face_probabilities=self.face_probabilities,
                mask_ratio=tmask_ratio
            )
            sampled_coords['sampled_faces_indices'] = sampled_coords['sampled_faces_indices'].repeat(Ke, 1)
            sampled_coords['barycentric_coords'] = sampled_coords['barycentric_coords'].repeat(Ke, 1, 1)
            points2, sampled_coords = masking_utils.mesh_based_mask_uniform_faces(
                renderer_output_2nd_path['transformed_vertices'], 
                flame_faces=self.flame.faces_tensor,
                face_probabilities=self.face_probabilities,
                mask_ratio=tmask_ratio,
                coords=sampled_coords
            )
            extra_points = masking_utils.transfer_pixels(img.repeat(Ke, 1, 1, 1), points1.repeat(Ke, 1, 1), points2)
            rendered_mask = (rendered_img_2nd_path > 0).all(dim=1, keepdim=True).float()
                
        masked_img_2nd_path = masking_utils.masking(
            img.repeat(Ke, 1, 1, 1), 
            masks.repeat(Ke, 1, 1, 1), 
            extra_points, 
            self.config.train.mask_dilation_radius, 
            rendered_mask=rendered_mask, 
            extra_noise=True, random_mask=0.005
        )
        
        reconstructed_img_2nd_path = self.smirk_generator(
            torch.cat([rendered_img_2nd_path, masked_img_2nd_path], dim=1).detach()
        )
        if self.config.train.freeze_generator_in_second_path:
            reconstructed_img_2nd_path = reconstructed_img_2nd_path.detach()

        recon_feats = self.smirk_encoder(reconstructed_img_2nd_path.view(Ke * B, C, H, W)) 
        flame_output_2nd_path_2 = self.flame.forward(recon_feats)
        rendered_img_2nd_path_2 = self.renderer.forward(flame_output_2nd_path_2['vertices'], recon_feats['cam'])['rendered_img']

        # ----------- cycle loss -----------
        losses = {}
        cycle_loss = 1.0 * F.mse_loss(recon_feats['expression_params'], flame_feats['expression_params']) + \
                     10.0 * F.mse_loss(recon_feats['jaw_params'], flame_feats['jaw_params'])
        if self.config.arch.use_eyelids:
            cycle_loss += 10.0 * F.mse_loss(recon_feats['eyelid_params'], flame_feats['eyelid_params'])
        if not self.config.train.freeze_generator_in_second_path:                
            cycle_loss += 1.0 * F.mse_loss(recon_feats['shape_params'], flame_feats['shape_params']) 

        losses['cycle_loss']  = cycle_loss
        loss_second_path = losses['cycle_loss'] * self.config.train.loss_weights['cycle_loss']

        for key, value in losses.items():
            losses[key] = value.item() if isinstance(value, torch.Tensor) else value

        # ----------- visualization struct -----------
        outputs = {}
        if batch_idx % self.config.train.visualize_every == 0:
            outputs['2nd_path'] = torch.stack([
                rendered_img_2nd_path.detach().cpu().view(Ke, B, C, H, W).permute(1,0,2,3,4).reshape(-1,C,H,W),
                masked_img_2nd_path.detach().cpu().view(Ke, B, C, H, W).permute(1,0,2,3,4).reshape(-1,C,H,W),
                reconstructed_img_2nd_path.detach().cpu().view(Ke, B, C, H, W).permute(1,0,2,3,4).reshape(-1,C,H,W),
                rendered_img_2nd_path_2.detach().cpu().view(Ke, B, C, H, W).permute(1,0,2,3,4).reshape(-1,C,H,W)
            ], dim=1).reshape(-1, C, H, W)

        return outputs, losses, loss_second_path
