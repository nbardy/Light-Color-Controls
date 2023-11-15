class LayerNorm(nn.LayerNorm):
    """Subclass torch's LayerNorm to handle fp16."""

    def forward(self, x: torch.Tensor):
        orig_type = x.dtype
        ret = super().forward(x.type(torch.float32))
        return ret.type(orig_type)


class QuickGELU(nn.Module):

    def forward(self, x: torch.Tensor):
        return x * torch.sigmoid(1.702 * x)


class ResidualAttentionBlock(nn.Module):

    def __init__(self, d_model: int, n_head: int, attn_mask: torch.Tensor = None):
        super().__init__()

        self.attn = nn.MultiheadAttention(d_model, n_head)
        self.ln_1 = LayerNorm(d_model)
        self.mlp = nn.Sequential(
            OrderedDict([("c_fc", nn.Linear(d_model, d_model * 4)), ("gelu", QuickGELU()),
                         ("c_proj", nn.Linear(d_model * 4, d_model))]))
        self.ln_2 = LayerNorm(d_model)
        self.attn_mask = attn_mask

    def attention(self, x: torch.Tensor):
        self.attn_mask = self.attn_mask.to(dtype=x.dtype, device=x.device) if self.attn_mask is not None else None
        return self.attn(x, x, x, need_weights=False, attn_mask=self.attn_mask)[0]

    def forward(self, x: torch.Tensor):
        x = x + self.attention(self.ln_1(x))
        x = x + self.mlp(self.ln_2(x))
        return x

class CoAdapterFuser(nn.Module):
    def __init__(self, unet_channels=[320, 640, 1280, 1280], width=768, num_head=8, n_layes=3):
        super(CoAdapterFuser, self).__init__()
        scale = width ** 0.5
        # 16, maybe large enough for the number of adapters?
        self.task_embedding = nn.Parameter(scale * torch.randn(16, width))
        self.positional_embedding = nn.Parameter(scale * torch.randn(len(unet_channels), width))
        self.spatial_feat_mapping = nn.ModuleList()
        for ch in unet_channels:
            self.spatial_feat_mapping.append(nn.Sequential(
                nn.SiLU(),
                nn.Linear(ch, width),
            ))
        self.transformer_layes = nn.Sequential(*[ResidualAttentionBlock(width, num_head) for _ in range(n_layes)])
        self.ln_post = LayerNorm(width)
        self.ln_pre = LayerNorm(width)
        self.spatial_ch_projs = nn.ModuleList()
        for ch in unet_channels:
            self.spatial_ch_projs.append(zero_module(nn.Linear(width, ch)))
        self.seq_proj = nn.Parameter(torch.zeros(width, width))

    def forward(self, features):
        if len(features) == 0:
            return None, None
        inputs = []
        for cond_name in features.keys():
            task_idx = getattr(ExtraCondition, cond_name).value
            if not isinstance(features[cond_name], list):
                inputs.append(features[cond_name] + self.task_embedding[task_idx])
                continue

            feat_seq = []
            for idx, feature_map in enumerate(features[cond_name]):
                feature_vec = torch.mean(feature_map, dim=(2, 3))
                feature_vec = self.spatial_feat_mapping[idx](feature_vec)
                feat_seq.append(feature_vec)
            feat_seq = torch.stack(feat_seq, dim=1)  # Nx4xC
            feat_seq = feat_seq + self.task_embedding[task_idx]
            feat_seq = feat_seq + self.positional_embedding
            inputs.append(feat_seq)

        x = torch.cat(inputs, dim=1)  # NxLxC
        x = self.ln_pre(x)
        x = x.permute(1, 0, 2)  # NLD -> LND
        x = self.transformer_layes(x)
        x = x.permute(1, 0, 2)  # LND -> NLD
        x = self.ln_post(x)

        ret_feat_map = None
        ret_feat_seq = None
        cur_seq_idx = 0
        for cond_name in features.keys():
            if not isinstance(features[cond_name], list):
                length = features[cond_name].size(1)
                transformed_feature = features[cond_name] * ((x[:, cur_seq_idx:cur_seq_idx+length] @ self.seq_proj) + 1)
                if ret_feat_seq is None:
                    ret_feat_seq = transformed_feature
                else:
                    ret_feat_seq = torch.cat([ret_feat_seq, transformed_feature], dim=1)
                cur_seq_idx += length
                continue

            length = len(features[cond_name])
            transformed_feature_list = []
            for idx in range(length):
                alpha = self.spatial_ch_projs[idx](x[:, cur_seq_idx+idx])
                alpha = alpha.unsqueeze(-1).unsqueeze(-1) + 1
                transformed_feature_list.append(features[cond_name][idx] * alpha)
            if ret_feat_map is None:
                ret_feat_map = transformed_feature_list
            else:
                ret_feat_map = list(map(lambda x, y: x + y, ret_feat_map, transformed_feature_list))
            cur_seq_idx += length

        assert cur_seq_idx == x.size(1)

        return ret_feat_map, ret_feat_seq
