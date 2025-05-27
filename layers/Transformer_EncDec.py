import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init

class ConvLayer(nn.Module):
    def __init__(self, c_in):
        super(ConvLayer, self).__init__()
        self.downConv = nn.Conv1d(in_channels=c_in,
                                  out_channels=c_in,
                                  kernel_size=3,
                                  padding=2,
                                  padding_mode='circular')
        self.norm = nn.BatchNorm1d(c_in)
        self.activation = nn.ELU()
        self.maxPool = nn.MaxPool1d(kernel_size=3, stride=2, padding=1)

    def forward(self, x):
        x = self.downConv(x.permute(0, 2, 1))
        x = self.norm(x)
        x = self.activation(x)
        x = self.maxPool(x)
        x = x.transpose(1, 2)
        return x


class EncoderLayer(nn.Module):
    def __init__(self, attention, d_model, d_ff=None, dropout=0.1, activation="relu"):
        super(EncoderLayer, self).__init__()
        d_ff = d_ff or 4 * d_model
        self.attention = attention
        self.conv1 = nn.Conv1d(in_channels=d_model, out_channels=d_ff, kernel_size=1)
        self.conv2 = nn.Conv1d(in_channels=d_ff, out_channels=d_model, kernel_size=1)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        self.activation = F.relu if activation == "relu" else F.gelu

    def forward(self, x, attn_mask=None):
        new_x, attn = self.attention(
            x, x, x,
            attn_mask=attn_mask
        )
        x = x + self.dropout(new_x)

        y = x = self.norm1(x)
        y = self.dropout(self.activation(self.conv1(y.transpose(-1, 1))))
        y = self.dropout(self.conv2(y).transpose(-1, 1))

        return self.norm2(x + y), attn


class Encoder(nn.Module):
    def __init__(self, attn_layers, conv_layers=None, norm_layer=None):
        super(Encoder, self).__init__()
        self.attn_layers = nn.ModuleList(attn_layers)
        self.conv_layers = nn.ModuleList(conv_layers) if conv_layers is not None else None
        self.norm = norm_layer

    def forward(self, x, attn_mask=None):
        # x [B, L, D]
        attns = []
        if self.conv_layers is not None:
            for attn_layer, conv_layer in zip(self.attn_layers, self.conv_layers):
                x, attn = attn_layer(x, attn_mask=attn_mask)
                x = conv_layer(x)
                attns.append(attn)
            x, attn = self.attn_layers[-1](x)
            attns.append(attn)
        else:
            for attn_layer in self.attn_layers:
                x, attn = attn_layer(x, attn_mask=attn_mask)
                attns.append(attn)

        if self.norm is not None:
            x = self.norm(x)

        return x, attns


class Lora_linear(nn.Module):
    def __init__(self, alpha, r, d_model):
        super().__init__()
        self.alpha = alpha
        self.r = r
        self.A_real = nn.Parameter(torch.randn(r, d_model), requires_grad=True)  
        self.A_imag = nn.Parameter(torch.randn(r, d_model), requires_grad=True)  
        self.B_real = nn.Parameter(torch.zeros(d_model, r), requires_grad=True)  
        self.B_imag = nn.Parameter(torch.zeros(d_model, r), requires_grad=True)  
        # self.real = nn.Parameter(torch.zeros(d_model, d_model), requires_grad=True)
        # self.imag = nn.Parameter(torch.zeros(d_model, d_model), requires_grad=True)
    def forward(self, x):
        A = self.A_real + 1j * self.A_imag  
        B = self.B_real + 1j * self.B_imag  
        # complex_parm = self.real+1j*self.imag
        result = x @ (B @ A) * (self.alpha / self.r)
        # result = x@complex_parm
        return result
class AdaptiveFourierNeuralOperator(nn.Module):
    def __init__(self,args, dim):
        super().__init__()
        self.args = args
        self.hidden_size = dim
        self.scale = 0.02
        # self.linear_amp = nn.Linear(self.hidden_size, self.hidden_size)
        # self.linear_phi = nn.Linear(self.hidden_size, self.hidden_size)
        self.w1 = torch.nn.Parameter(self.scale * torch.randn(2, self.hidden_size//2+1, self.hidden_size//2+1))
        self.b1 = torch.nn.Parameter(self.scale * torch.randn(2,  self.hidden_size//2+1))
        self.w2 = torch.nn.Parameter(self.scale * torch.randn(2,  self.hidden_size//2+1, self.hidden_size//2+1))
        self.b2 = torch.nn.Parameter(self.scale * torch.randn(2, self.hidden_size//2+1))
        # self.feature_linear_1 = torch.nn.Linear(self.hidden_size//2+1,self.hidden_size//2+1).to(torch.cfloat)
        # self.feature_linear_2 = torch.nn.Linear(self.hidden_size // 2 + 1, self.hidden_size // 2 + 1).to(torch.cfloat)
        self.gra_linear_1 = nn.ModuleList([
            Lora_linear(4, 1, dim // 2 + 1), 
            Lora_linear(4, 1, dim // 2 + 1),  
            Lora_linear(4, 1, dim // 2 + 1),  
            Lora_linear(4, 1, dim // 2 + 1),  
            Lora_linear(4, 1, dim // 2 + 1) 
        ])
        self.gra_linear_2 = nn.ModuleList([
            Lora_linear(4, 1, dim // 2 + 1), 
            Lora_linear(4, 1, dim // 2 + 1), 
            Lora_linear(4, 1, dim // 2 + 1), 
            Lora_linear(4, 1, dim // 2 + 1), 
            Lora_linear(4, 1, dim // 2 + 1)
        ])
        self.gra_embedding = nn.Linear(5,self.hidden_size//2+1)
        self.gra_feature =  torch.nn.Parameter(torch.randn(5,self.hidden_size//2+1))
        self.relu = nn.ReLU()

    def multiply(self, input, weights):
        return torch.einsum('...bd,dk->...bk', input, weights)

    def replace_null(self,gra_list):
        null_indices = []
        for i in range(len(gra_list)):
            if sum(gra_list[i]) == 0:
                null_indices.append(i)
        return gra_list, null_indices
    def forward(self, x, time_gra,spatial_size=None):
        B, N, C = x.shape
        x = torch.fft.rfft(x,dim=2,norm='ortho')
        # if time_gra == 'null':
        #     x = self.feature_linear_1(x)
        #     x = self.feature_linear_2(x)
        #     x = torch.fft.irfft(x, dim=2, norm="ortho")
        #     return x
        time_gra = torch.stack(time_gra)
        time_gra = time_gra.permute(1,0).to(x.device).to(torch.float)
        time_gra,null_index = self.replace_null(time_gra)
        # time_gra = torch.tensor(time_gra).to(x.device)
        gra = self.gra_embedding(time_gra)
        gra_atten = self.gra_attention(gra,self.gra_feature)
        gra_atten = gra_atten.permute(1,0)
        # amp = torch.abs(x)
        # phi = torch.angle(x)
        # amp = self.linear_amp(amp)
        # phi = self.linear_phi(phi)
        # frequencies = torch.zeros(x.shape,dtype=torch.complex64)
        # frequencies = amp * torch.exp(1j * phi)
        # x_time = torch.fft.irfft(frequencies,dim=1)
        gra_atten = gra_atten.unsqueeze(-1).repeat(1,1,x.size(-1))
        if self.args.task_name == 'imputation':
            gra_atten = gra_atten.unsqueeze(-2).repeat(1,1,x.size(1),1)
        else:
            gra_atten = gra_atten.unsqueeze(-2).repeat(1, self.args.enc_in, x.size(1), 1)
        moe_out_1 = self.gra_linear_1[0](gra_atten[0]*x)
        for i in range(1,len(self.gra_linear_1)):
            moe_out_1+=self.gra_linear_1[i](gra_atten[i]*x)
        zero_tensor = torch.zeros_like(moe_out_1).to(x.device)
        for index in null_index:
            moe_out_1[index] = zero_tensor[index]

        x_real_1 = F.relu(self.multiply(x.real, self.w1[0]) - self.multiply(x.imag, self.w1[1]) + self.b1[0])
        x_imag_1 = F.relu(self.multiply(x.real, self.w1[1]) + self.multiply(x.imag, self.w1[0]) + self.b1[1])
        x = torch.stack([x_real_1, x_imag_1], dim=-1).float()
        x = torch.view_as_complex(x)
        x = x + moe_out_1
        moe_out_2 = self.gra_linear_2[0](gra_atten[0]*x)
        for i in range(1,len(self.gra_linear_2)):
            moe_out_2+=self.gra_linear_2[i](gra_atten[i]*x)
        zero_tensor = torch.zeros_like(moe_out_2).to(x.device)
        for index in null_index:
            moe_out_2[index] = zero_tensor[index]

        x_real_2 = self.multiply(x.real, self.w2[0]) - self.multiply(x.imag, self.w2[1]) + self.b2[0]
        x_imag_2 = self.multiply(x.real, self.w2[1]) + self.multiply(x.imag, self.w2[0]) + self.b2[1]
        x = torch.stack([x_real_2, x_imag_2], dim=-1).float()
        x = torch.view_as_complex(x)
        x = x + moe_out_2
        # x_real_2 = self.multiply(x_real_1, self.w2[0]) - self.multiply(x_imag_1, self.w2[1]) + self.b2[0]
        # x_imag_2 = self.multiply(x_real_1, self.w2[1]) + self.multiply(x_imag_1, self.w2[0]) + self.b2[1]


        # x = F.softshrink(x, lambd=self.softshrink) if self.softshrink else x

        # x = x.reshape(B, x.shape[1], x.shape[2], self.hidden_size)
        x = torch.fft.irfft(x, dim=2, norm="ortho")
        # x = x.squeeze(-2)
        return x

    def gra_attention(self,input,gra_feature):
        attention_scores = torch.einsum('ik,jk->ij', input, gra_feature)
        attention_weights = F.softmax( attention_scores, dim=-1)
        return attention_weights

class DecoderLayer(nn.Module):
    def __init__(self,args, cross_attention, d_model, d_ff=None,
                 dropout=0.1, activation="relu"):
        super(DecoderLayer, self).__init__()
        d_ff = d_ff or 4 * d_model
        self.cross_attention = cross_attention
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        self.activation = F.relu if activation == "relu" else F.gelu
        self.filter = AdaptiveFourierNeuralOperator(args, d_model)
    def forward(self, x, cross, time_gra, x_mask=None, cross_mask=None):
        x = x + self.dropout(self.cross_attention(
            x, cross, cross,
            attn_mask=x_mask
        )[0])
        y = self.norm2(x)
        y = self.filter(y,time_gra)
        return self.norm3(x + y)
        # return y


class Decoder(nn.Module):
    def __init__(self, layers, norm_layer=None, projection=None):
        super(Decoder, self).__init__()
        self.layers = nn.ModuleList(layers)
        self.norm = norm_layer
        self.projection = projection

    def forward(self, x,cross,time_gra, x_mask=None, cross_mask=None):
        for layer in self.layers:
            x = layer(x, cross,time_gra, x_mask=x_mask, cross_mask=cross_mask)

        if self.norm is not None:
            x = self.norm(x)

        if self.projection is not None:
            x = self.projection(x)
        return x
