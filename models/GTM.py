import torch
from torch import nn
from GTM.layers.Transformer_EncDec import Decoder,DecoderLayer
from GTM.layers.SelfAttention_Family import FullAttention, AttentionLayer
from GTM.layers.Embed import PatchEmbedding
from einops import rearrange

class Transpose(nn.Module):
    def __init__(self, *dims, contiguous=False): 
        super().__init__()
        self.dims, self.contiguous = dims, contiguous
    def forward(self, x):
        if self.contiguous: return x.transpose(*self.dims).contiguous()
        else: return x.transpose(*self.dims)


class FlattenHead(nn.Module):
    def __init__(self, n_vars, nf, target_window, head_dropout=0):
        super().__init__()
        self.n_vars = n_vars
        self.flatten = nn.Flatten(start_dim=-2)
        self.linear = nn.Linear(nf, target_window)
        self.dropout = nn.Dropout(head_dropout)

    def forward(self, x):  # x: [bs x nvars x d_model x patch_num]
        x = self.flatten(x)
        x = self.linear(x)
        x = self.dropout(x)
        return x


class Model(nn.Module):
    """
    Paper link: https://arxiv.org/pdf/2211.14730.pdf
    """

    def __init__(self, configs):
        """
        patch_len: int, patch len for patch_embedding
        stride: int, stride for patch_embedding
        """
        super().__init__()
        self.task_name = configs.task_name
        self.seq_len = configs.seq_len
        self.pred_len = configs.pred_len
        self.patch_len = configs.patch_len
        self.stride = configs.stride
        padding = 0
        # patching and embedding
        self.patch_embedding = PatchEmbedding(
            configs,configs.d_model, self.patch_len, self.stride, padding, configs.dropout)
        self.flatten_pred = nn.Flatten(start_dim=-2)
        self.device = configs.device
        self.decoder = Decoder(
            [
                DecoderLayer(
                    configs,
                    AttentionLayer(
                        FullAttention(False, configs.factor, attention_dropout=configs.dropout, output_attention=False),
                        configs.d_model, configs.n_heads),
                    configs.d_model,
                    configs.d_ff,
                    dropout=configs.dropout,
                    activation=configs.activation,
                )
                for l in range(configs.d_layers)
            ],
            norm_layer=torch.nn.LayerNorm(configs.d_model),
        )

        # Prediction Head
        self.head_nf = configs.d_model * (self.seq_len//self.patch_len)
        if self.task_name == 'long_term_forecast' or self.task_name == 'short_term_forecast' or self.task_name == 'anomaly_detection':
            self.head = nn.Linear(configs.d_model, self.patch_len)
        if self.task_name == 'imputation':
            self.head = nn.Linear(configs.d_model, configs.enc_in)
        elif 'pre_train' in self.task_name:
            self.head = nn.Linear(configs.d_model, self.patch_len)

    def forecast(self, x,time_gra):
        B, L, C = x.size()
        means = x.mean(1, keepdim=True).detach()
        x = x - means
        stdev = torch.sqrt(
            torch.var(x, dim=1, keepdim=True, unbiased=False) + 1e-5)
        x /= stdev
        # Normalization from Non-stationary Transformer
        # do patching and embedding
        x = x.permute(0, 2, 1)
        if self.pred_len>self.seq_len:
            pred_mask_len = ((self.pred_len-self.seq_len)// self.patch_len+1) * self.patch_len if self.pred_len % self.patch_len != 0 else self.pred_len - self.seq_len
        # 检查 pred_mask_len 是否大于 0
            if pred_mask_len > 0:
                pred_mask = torch.zeros(B, C, pred_mask_len, device=x.device)  # 直接在目标设备上创建
                x = torch.concat((x, pred_mask), dim=2)
        # u: [bs * nvars x patch_num x d_model]
        dec_in,n_vars= self.patch_embedding(x)

        # Encoder
        # z: [bs * nvars x patch_num x d_model]
        dec_out = self.decoder(dec_in, dec_in,time_gra, x_mask=None, cross_mask=None)

        dec_out = torch.reshape(
            dec_out, (B, C, dec_out.shape[-2], dec_out.shape[-1]))
        # Decoder
        dec_out = self.head(dec_out)
        dec_out = self.flatten_pred(dec_out)
        dec_out = dec_out.permute(0, 2, 1)
        dec_out = dec_out * \
                  (stdev.repeat(1, dec_out.size(1),1))
        dec_out = dec_out + \
                  (means.repeat(1,dec_out.size(1),1))
        return dec_out[:,-self.pred_len:,:]

    def imputation(self, x_enc, time_gra,mask):
        # Normalization from Non-stationary Transformer
        B, L, C = x_enc.size()
        means = torch.sum(x_enc, dim=1) / torch.sum(mask == 1, dim=1)
        means = means.unsqueeze(1).detach()
        x_enc = x_enc - means
        x_enc = x_enc.masked_fill(mask == 0, 0)
        stdev = torch.sqrt(torch.sum(x_enc * x_enc, dim=1) /
                           torch.sum(mask == 1, dim=1) + 1e-5)
        stdev = stdev.unsqueeze(1).detach()
        x_enc /= stdev
        # Normalization from Non-stationary Transformer
        # do patching and embedding
        dec_in, n_vars = self.patch_embedding(x_enc)

        # Encoder
        # z: [bs * nvars x patch_num x d_model]
        dec_out = self.decoder(dec_in, dec_in, time_gra, x_mask=None, cross_mask=None)
        # Decoder
        dec_out = self.head(dec_out)
        dec_out = dec_out * \
                  (stdev.repeat(1, dec_out.size(1), 1))
        dec_out = dec_out + \
                  (means.repeat(1, dec_out.size(1), 1))
        return dec_out

    def anomaly_detection(self, x_enc,time_gra):
        B, L, C = x_enc.size()
        means = x_enc.mean(1, keepdim=True).detach()
        x = x_enc - means
        stdev = torch.sqrt(
            torch.var(x, dim=1, keepdim=True, unbiased=False) + 1e-5)
        x /= stdev
        x = x.permute(0, 2, 1)
        # Normalization from Non-stationary Transformer
        # do patching and embedding
        dec_in, n_vars = self.patch_embedding(x)

        # Encoder
        dec_out = self.decoder(dec_in, dec_in, time_gra, x_mask=None, cross_mask=None)

        # Decoder
        dec_out = self.head(dec_out)
        dec_out = torch.reshape(
            dec_out, (B, C, dec_out.shape[-2], dec_out.shape[-1]))

        dec_out = self.flatten_pred(dec_out)
        dec_out = dec_out.permute(0, 2, 1)
        dec_out = dec_out * \
                  (stdev.repeat(1, dec_out.size(1), 1))
        dec_out = dec_out + \
                  (means.repeat(1, dec_out.size(1), 1))
        return dec_out

    def pre_train(self, x,time_gra):
        # Normalization from Non-stationary Transformer
        B,L,C = x.size()
        means = x.mean(1, keepdim=True).detach()
        x = x - means
        stdev = torch.sqrt(
            torch.var(x, dim=1, keepdim=True, unbiased=False) + 1e-5)
        x /= stdev

        # do patching and embedding
        x = x.permute(0, 2, 1)
        # u: [bs * nvars x patch_num x d_model]
        dec_in, x_patch,atten_mask,mask_id,end_token_indices,pos_1,pos_2 = self.patch_embedding(x,mask=True)

        # Encoder
        # z: [bs * nvars x patch_num x d_model]
        dec_out = self.decoder(dec_in, dec_in, time_gra,x_mask=atten_mask, cross_mask=None)
        # z: [bs x nvars x patch_num x d_model]
        dec_out = torch.reshape(
            dec_out, (B, C, dec_out.shape[-2], dec_out.shape[-1]))
        # Decoder
        dec_out = self.head(dec_out)  # z: [bs x nvars x target_window]

        #将end_token位置掩掉，不进行损失计算
        mask_index = torch.where(end_token_indices, False, mask_id)
        mask_index = torch.reshape(mask_index, (B, C, dec_out.shape[-2]))
        mask_index =mask_index.unsqueeze(-1).expand(-1, -1, -1, self.patch_len)
        dec_out = dec_out*mask_index
        x_patch = torch.reshape(
            x_patch, (B, C, x_patch.shape[-2], x_patch.shape[-1]))
        x_patch = x_patch*mask_index
        x_patch = x_patch.permute(0,2,3,1)
        dec_out = dec_out.permute(0,2,3,1)
        x_patch = x_patch * \
                  (stdev.unsqueeze(1).repeat(1,dec_in.shape[1],self.patch_len,1))
        x_patch = x_patch + \
                  (means.unsqueeze(1).repeat(1,dec_in.shape[1],self.patch_len,1))
        # De-Normalization from Non-stationary Transformer
        dec_out = dec_out * \
                  (stdev.unsqueeze(1).repeat(1,dec_in.shape[1],self.patch_len,1))
        dec_out = dec_out + \
                  (means.unsqueeze(1).repeat(1,dec_in.shape[1],self.patch_len,1))

        return dec_out,x_patch

    def forward(self, x_enc, time_gra,mask=None):
        if self.task_name == 'long_term_forecast' or self.task_name == 'short_term_forecast':
            #掩码预测
            dec_out = self.forecast(x_enc,time_gra)
            return dec_out  # [B, N]
        if self.task_name == 'imputation':
            dec_out = self.imputation(
                x_enc, time_gra,mask)
            return dec_out  # [B, L, D]
        if self.task_name == 'anomaly_detection':
            dec_out = self.anomaly_detection(x_enc,time_gra)
            return dec_out  # [B, L, D]
        if 'pre_train' in self.task_name:
            dec_out,x = self.pre_train(x_enc,time_gra)
            return dec_out,x  # [B, N]
