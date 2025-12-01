    # ---------------- second path ---------------- #
    def step2(self, encoder_output, batch, batch_idx, phase='train'):
        B, C, H, W = batch['img'].shape        
        img = batch['img']
        masks = batch['mask']
        
        # number of multiple versions for the second path
        Ke = self.config.train.Ke
        
        # start from the same encoder output and add noise to expression params
        flame_feats = {}
        for k, v in encoder_output.items():
            tmp = v.clone().detach()
            flame_feats[k] = torch.cat(Ke * [tmp], dim=0)

        # split Ke * B into 4 random groups        
        gids = torch.randperm(Ke * B)
        gids = [gids[:Ke * B // 4], gids[Ke * B // 4: 2 * Ke * B // 4], 
                gids[2 * Ke * B // 4: 3 * Ke * B // 4], gids[3 * Ke * B // 4:]] 

        feats_dim = flame_feats['expression_params'].size(1)        

        # ---------------- random expression ---------------- #
        param_mask = torch.bernoulli(torch.ones((len(gids[0]), feats_dim)) * 0.5).to(self.config.device)
        new_expressions = (torch.randn((len(gids[0]), feats_dim)).to(self.config.device)) * \
                          (1 + 2 * torch.rand((len(gids[0]), 1)).to(self.config.device)) * param_mask + \
                          flame_feats['expression_params'][gids[0]]
        flame_feats['expression_params'][gids[0]] = torch.clamp(new_expressions, -4.0, 4.0) + \
                                                    (0 + 0.2 * torch.rand((len(gids[0]), 1)).to(self.config.device)) * \
                                                    torch.randn((len(gids[0]), feats_dim)).to(self.config.device)
        
        # ---------------- permutation of expression ---------------- #
        flame_feats['expression_params'][gids[1]] = (0.25 + 1.25 * torch.rand((len(gids[1]), 1)).to(self.config.device)) * \
                                                    flame_feats['expression_params'][gids[1]][torch.randperm(len(gids[1]))] + \
                                                    (0 + 0.2 * torch.rand((len(gids[1]), 1)).to(self.config.device)) * \
                                                    torch.randn((len(gids[1]), feats_dim)).to(self.config.device)
        
        # ---------------- template injection ---------------- #
        for i in range(len(gids[2])):
            expression = self.load_random_template(num_expressions=self.config.arch.num_expression)
            flame_feats['expression_params'][gids[2][i],:self.config.arch.num_expression] = \
                (0.25 + 1.25 * torch.rand((1, 1)).to(self.config.device)) * torch.Tensor(expression).to(self.config.device)
        flame_feats['expression_params'][gids[2]] += (0 + 0.2 * torch.rand((len(gids[2]), 1)).to(self.config.device)) * \
                                                     torch.randn((len(gids[2]), feats_dim)).to(self.config.device)

        # ---------------- tweak jaw ---------------- #
        scale_mask = torch.Tensor([1, .1, .1]).to(self.config.device).view(1, 3) * \
                     torch.bernoulli(torch.ones(Ke * B) * 0.5).to(self.config.device).view(-1, 1)
        flame_feats['jaw_params'] = flame_feats['jaw_params'] + \
                                    torch.randn(flame_feats['jaw_params'].size()).to(self.config.device) * 0.2 * scale_mask
        flame_feats['jaw_params'][..., 0] = torch.clamp(flame_feats['jaw_params'][..., 0], 0.0, 0.5)
        
        # ---------------- tweak eyelids ---------------- #
        if self.config.arch.use_eyelids:
            flame_feats['eyelid_params'] += (-1 + 2 * torch.rand(size=flame_feats['eyelid_params'].size()).to(self.config.device)) * 0.25
            flame_feats['eyelid_params'] = torch.clamp(flame_feats['eyelid_params'], 0.0, 1.0)

        # ---------------- zero expression ---------------- #
        flame_feats['expression_params'][gids[3]] *= 0.0
        flame_feats['expression_params'][gids[3]] += (0 + 0.2 * torch.rand((len(gids[3]), 1)).to(self.config.device)) * \
                                                     torch.randn((len(gids[3]), flame_feats['expression_params'].size(1))).to(self.config.device)
        flame_feats['jaw_params'][gids[3]] *= 0.0
        flame_feats['eyelid_params'][gids[3]] = torch.rand(size=flame_feats['eyelid_params'][gids[3]].size()).to(self.config.device)        

        # detach
        for k in ['expression_params','pose_params','shape_params','jaw_params','eyelid_params']:
            flame_feats[k] = flame_feats[k].detach()

        # ---------------- render ---------------- #
        with torch.no_grad():
            flame_output = self.flame.forward(encoder_output)
            rendered_output = self.renderer.forward(flame_output['vertices'], encoder_output['cam'])
            flame_output.update(rendered_output)

            # render the tweaked face
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
