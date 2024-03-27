import torch
from torch import einsum
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
import math
import warnings

def cg(A, b, x0, tolerance, max_cgiter=50, M=None):
    r = b - A(x0)
    if M is None:
        p = r
    else:
        p = M(r)
    k = 0

    # Iterative solution
    for iter in range(max_cgiter):
        Ap = A(p)
        alpha = torch.dot(r, r) / torch.dot(p, Ap)
        x = x0 + alpha * p
        r_new = r - alpha * Ap
        beta = torch.dot(r_new, r_new) / torch.dot(r, r)
        p = r_new + beta * p

        # Update variables
        x0 = x
        r = r_new
        k += 1

        if torch.norm(r) <= tolerance:
            break

    return x


def _no_grad_trunc_normal_(tensor, mean, std, a, b):
    def norm_cdf(x):
        return (1. + math.erf(x / math.sqrt(2.))) / 2.

    if (mean < a - 2 * std) or (mean > b + 2 * std):
        warnings.warn("mean is more than 2 std from [a, b] in nn.init.trunc_normal_. "
                      "The distribution of values may be incorrect.",
                      stacklevel=2)
    with torch.no_grad():
        l = norm_cdf((a - mean) / std)
        u = norm_cdf((b - mean) / std)
        tensor.uniform_(2 * l - 1, 2 * u - 1)
        tensor.erfinv_()
        tensor.mul_(std * math.sqrt(2.))
        tensor.add_(mean)
        tensor.clamp_(min=a, max=b)
        return tensor

def trunc_normal_(tensor, mean=0., std=1., a=-2., b=2.):
    # type: (Tensor, float, float, float, float) -> Tensor
    return _no_grad_trunc_normal_(tensor, mean, std, a, b)
class Attention(nn.Module):
    def __init__(self, inner_dim, seq_l, num_heads=8, qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0., sr_ratio=1):
        super().__init__()
        self.inner_dim = inner_dim
        self.num_heads = num_heads
        head_dim = inner_dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5
        self.positional_encoding2 = nn.Parameter(torch.Tensor(1, num_heads, seq_l, seq_l))
        trunc_normal_(self.positional_encoding2)

    def forward(self, q,k,v,):
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h=self.num_heads), (q, k, v))
        q = q * self.scale
        sim = einsum('b h i d, b h j d -> b h i j', q, k)
        sim = sim + self.positional_encoding2
        attn = sim.softmax(dim=-1)
        out = einsum('b h i j, b h j d -> b h i d', attn, v)
        out = rearrange(out, 'b h n d -> b  n (h d)', h=self.num_heads)
        return out

class InterAttention(Attention):
    def forward(self, q, k, v,):
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h=self.num_heads), (q, k, v))
        q = q.transpose(2, 3)
        k = k.transpose(2, 3)

        cosine_similarity = torch.einsum('b h d n,b h d n->b h n', q, k) / (
                torch.norm(q, dim=2) * torch.norm(k, dim=2))
        cosine_similarity = cosine_similarity.unsqueeze(-1)
        out = cosine_similarity * v
        out = rearrange(out, 'b h n d -> b  n (h d)', h=self.num_heads)
        return out

class InnerAttentionColor(Attention):
    def forward(self, q, k, v,):
        q, k, v = map(lambda t: rearrange(t, 'b (h d) n  -> b h n d', h=self.num_heads), (q, k, v))
        q = q * self.scale
        sim = einsum('b h i d, b h j d -> b h i j', q, k)
        sim = sim + self.positional_encoding2
        attn = sim.softmax(dim=-1)
        out = einsum('b h i j, b h j d -> b h i d', attn, v)
        out = rearrange(out, 'b h n d -> b (h d) n ', h=self.num_heads)
        return out

class In2MA(nn.Module):
    def __init__(
            self,
            num_heads,
            window_size,
            spa_dim,
            spa_inner_dim,
            shift=False,
            spa_local=True
    ):
        super().__init__()

        self.window_size = window_size
        self.shift = shift
        self.num_heads = num_heads
        self.spa_local = spa_local

        self.pan_to_q = nn.Linear(spa_dim // 2, spa_inner_dim // 2, bias=False)
        self.pan_to_k = nn.Linear(spa_dim // 2, spa_inner_dim // 2, bias=False)
        self.x_to_v1 = nn.Linear(spa_dim, spa_inner_dim, bias=False)
        self.x_to_v2 = nn.Linear(spa_dim, spa_inner_dim, bias=False)
        self.x_to_k2 = nn.Linear(spa_dim, spa_inner_dim // 2, bias=False)

        self.inner_out = nn.Linear(spa_inner_dim, spa_dim, bias=False)
        self.inter_out = nn.Linear(spa_inner_dim, spa_dim, bias=False)

        self.x_to_q1_color = nn.Linear(spa_dim, spa_inner_dim//2, bias=False)
        self.x_to_k1_color = nn.Linear(spa_dim, spa_inner_dim//2, bias=False)

        if self.spa_local:
            # MHA-S
            self.inner_atten = Attention(inner_dim=spa_inner_dim, seq_l=self.window_size[0] * self.window_size[1],
                                         num_heads=num_heads)
            # MHA-C
            self.inner_atten_color = InnerAttentionColor(inner_dim=self.window_size[0] * self.window_size[1],
                                                         seq_l=spa_inner_dim // 2,
                                                         num_heads=num_heads)
            # CRW
            self.inter_atten = InterAttention(inner_dim=spa_inner_dim, seq_l=self.window_size[0] * self.window_size[1],
                                              num_heads=num_heads)


        else:
            h, w = 256 // num_heads, 256 // num_heads
            h_num, w_num = (h // self.window_size[0]), (w // self.window_size[1])
            # MHA-S
            self.inner_atten = Attention(inner_dim=spa_inner_dim, seq_l=h_num*w_num,
                                         num_heads=num_heads)
            # MHA-C
            self.inner_atten_color = InnerAttentionColor(inner_dim=self.window_size[0] * self.window_size[1],
                                                         seq_l=spa_inner_dim // 2,
                                                         num_heads=num_heads)
            # CRW
            self.inter_atten = InterAttention(inner_dim=spa_inner_dim, seq_l=h_num * w_num,
                                              num_heads=num_heads)

    def forward(self,x,pan_feature):
        """
        x: [b,c,h,w]
        pan_feature:[b,c//2,h,w]
        """
        b,c1,h,w = x.shape


        w_size = self.window_size
        h_num, w_num = (h // w_size[0]), (w // w_size[1])


        if self.spa_local:
            # prepare_token
            pan_fea = rearrange(pan_feature, "b c (h_num hl) (w_num wl) -> (b h_num w_num) (hl wl) c",h_num=h_num,w_num=w_num)
            x_fea = rearrange(x, "b c (h_num hl) (w_num wl) -> (b h_num w_num) (hl wl) c",h_num=h_num,w_num=w_num)
            # atten
            x_fea = self.In2Atten(pan_fea, x_fea)
            out = rearrange(
                x_fea, "(b h_num w_num) (hl wl) c -> b c (h_num hl) (w_num wl)",
                h_num=h_num,
                w_num=w_num,
                hl=w_size[0],
                wl=w_size[1],
            )
        else:
            # prepare_token
            pan_fea = rearrange(pan_feature, "b c (h_num hl) (w_num wl) -> (b hl wl) (h_num w_num) c", h_num=h_num,
                                  w_num=w_num)
            x_fea = rearrange(x, "b c (h_num hl) (w_num wl) -> (b hl wl) (h_num w_num) c", h_num=h_num, w_num=w_num)
            # Apply Attention
            x_fea = self.In2Atten(pan_fea, x_fea)
            out = rearrange(
                x_fea, "(b hl wl) (h_num w_num) c -> b c (h_num hl) (w_num wl)",
                h_num=h_num,
                w_num=w_num,
                hl=w_size[0],
                wl=w_size[1],
            )
        return out

    def In2Atten(self, pan, x):

        pan_q = self.pan_to_q(pan)
        pan_k = self.pan_to_k(pan)
        x_v1 = self.x_to_v1(x)


        x_v1_pan = x_v1[:, :, :x_v1.shape[-1] // 2]
        x_v1_color = x_v1[:, :, x_v1.shape[-1] // 2:]
        out_pan = self.inner_atten(pan_q, pan_k, x_v1_pan)
        # MHA-C
        x_q1_color = self.x_to_q1_color(x)
        x_k1_color = self.x_to_k1_color(x)
        out_color = self.inner_atten_color(x_q1_color, x_k1_color, x_v1_color)
        # MHA-S
        out = self.inner_out(torch.cat([out_pan,out_color],dim=-1))
        x_v2 = self.x_to_v2(out)
        x_k2 = self.x_to_k2(out)
        # CRW
        out = self.inter_atten(pan_q, x_k2, x_v2)
        out = self.inter_out(out)
        return out
class BatchNorm(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.fn = fn
        self.norm = nn.BatchNorm2d(dim)
    def forward(self,x,*args, **kwargs):
        x = self.norm(x)
        return self.fn(x, *args, **kwargs)


class GELU(nn.Module):
    def forward(self, x):
        return F.gelu(x)

class FeedForward(nn.Module):
    def __init__(self, dim, mult=4):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(dim, dim * mult, 1, 1, bias=False),
            GELU(),
            nn.Conv2d(dim * mult, dim * mult, 3, 1, 1, bias=False, groups=dim * mult),
            GELU(),
            nn.Conv2d(dim * mult, dim, 1, 1, bias=False),
        )

    def forward(self, x):
        """
        x: [b c h w]
        return out: [b c h w]
        """
        out = self.net(x)
        return out

class In2AB(nn.Module):
    def __init__(self,dim,h,w,
            window_size=(8, 8),
            dim_head=64,
            heads=8,
            num_blocks=2,):
        super().__init__()
        self.blocks = nn.ModuleList([])
        spa_dim = dim
        for _ in range(num_blocks):
            self.blocks.append(nn.ModuleList([
                BatchNorm(
                    dim,
                    In2MA(
                        spa_dim=spa_dim,
                        spa_inner_dim=dim_head*heads,
                        num_heads=heads,
                        window_size=window_size,
                        spa_local=(heads==1)
                    )
                ),
                BatchNorm(dim, FeedForward(dim=dim))
            ]))

    def forward(self, x,pan_feature):
        """
        x: [b,c,h,w]
        x: [b,c//2,h,w]
        return out: [b,c,h,w]
        """
        for (attn, ff) in self.blocks:
            x = attn(x,pan_feature) + x
            x = ff(x) + x
        return x

class PANFeatureExTractor(nn.Module):
    def __init__(self, in_channel, out_channels=[10, 20, 40]):
        super().__init__()
        self.extractors = nn.ModuleList([])
        self.extractors.append(
            nn.Sequential(
                CBR(nn.Conv2d(in_channels=in_channel, out_channels=out_channels[0],kernel_size=7,stride=1,padding=3)),
                CBR(nn.Conv2d(in_channels=out_channels[0], out_channels=out_channels[0],kernel_size=3,stride=1,padding=1)),
            )
        )
        for i in range(2):
            self.extractors.append(
                nn.Sequential(
                    nn.MaxPool2d(2, stride=2),
                    CBR(nn.Conv2d(in_channels=out_channels[i + 0], out_channels=out_channels[i + 1],kernel_size=3,stride=1,padding=1)),
                    CBR(nn.Conv2d(in_channels=out_channels[i + 1], out_channels=out_channels[i + 1],kernel_size=3,stride=1,padding=1)),
                )
            )

    def forward(self, x):
        Qs = []
        for extractor in self.extractors:
            x = extractor(x)
            Qs.append(x)
        return Qs
class In2SET(nn.Module):
    def __init__(self, in_dim=28, out_dim=28, dim=28,H=None,W=None, num_blocks=[1,1,1]):
        super(In2SET, self).__init__()
        self.dim = dim
        self.scales = len(num_blocks)



        # Input projection
        self.embedding = nn.Conv2d(in_dim, self.dim, 3, 1, 1, bias=False)

        # Encoder
        self.encoder_layers = nn.ModuleList([])
        dim_scale = dim
        h = H
        w = W

        for i in range(self.scales-1):
            self.encoder_layers.append(nn.ModuleList([
                In2AB(dim=dim_scale, h=h, w=w, num_blocks=num_blocks[i], dim_head=dim, heads=dim_scale // dim),
                nn.Conv2d(dim_scale, dim_scale * 2, 4, 2, 1, bias=False),
            ]))
            dim_scale *= 2
            h = h // 2
            w = w // 2

        # Bottleneck
        self.bottleneck = In2AB(dim=dim_scale, h=h, w=w, dim_head=dim, heads=dim_scale // dim, num_blocks=num_blocks[-1])

        # Decoder
        self.decoder_layers = nn.ModuleList([])
        for i in range(self.scales-1):
            self.decoder_layers.append(nn.ModuleList([
                nn.ConvTranspose2d(dim_scale, dim_scale // 2, stride=2, kernel_size=2, padding=0, output_padding=0),
                nn.Conv2d(dim_scale, dim_scale // 2, 1, 1, bias=False),
                In2AB(dim=dim_scale // 2, h=h, w=w, num_blocks=num_blocks[self.scales - 2 - i], dim_head=dim,
                      heads=(dim_scale // 2) // dim),
            ]))
            dim_scale //= 2
            h = h * 2
            w = w * 2

        # Output projection
        self.mapping = nn.Conv2d(self.dim, out_dim, 3, 1, 1, bias=False)

        #### activation function
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def forward(self, x,pan_feas):
        """
        x: [b,c,h,w]
        return out:[b,c,h,w]
        """


        b, c, h_inp, w_inp = x.shape
        hb, wb = 16, 16
        pad_h = (hb - h_inp % hb) % hb
        pad_w = (wb - w_inp % wb) % wb
        x = F.pad(x, [0, pad_w, 0, pad_h], mode='reflect')


        # Embedding
        fea = self.embedding(x)
        x = x[:,:28,:,:]

        # pan_feas = self.pan_extractors(pan)
        # Encoder
        fea_encoder = []
        for i,(SPSAB, FeaDownSample) in enumerate(self.encoder_layers):
            pan_fea = pan_feas[i]
            fea = SPSAB(fea,pan_fea)
            fea_encoder.append(fea)
            fea = FeaDownSample(fea)

        # Bottleneck
        pan_fea = pan_feas[-1]
        fea = self.bottleneck(fea,pan_fea)

        pan_feas = pan_feas[::-1][1:]
        # Decoder
        for i, (FeaUpSample, Fution, SPSAB) in enumerate(self.decoder_layers):
            fea = FeaUpSample(fea)
            fea = Fution(torch.cat([fea, fea_encoder[self.scales-2-i]], dim=1))
            pan_fea = pan_feas[i]
            fea = SPSAB(fea, pan_fea)

        # Mapping
        out = self.mapping(fea) + x
        return out[:, :, :h_inp, :w_inp]

def A(x,Phi):
    temp = x*Phi
    y = torch.sum(temp,1)
    return y

def At(y,Phi):
    temp = torch.unsqueeze(y, 1).repeat(1,Phi.shape[1],1,1)
    x = temp*Phi
    return x

def shift_3d(inputs,step=2):
    [bs, nC, row, col] = inputs.shape
    for i in range(nC):
        inputs[:,i,:,:] = torch.roll(inputs[:,i,:,:], shifts=step*i, dims=2)
    return inputs

def shift_back_3d(inputs,step=2):
    [bs, nC, row, col] = inputs.shape
    for i in range(nC):
        inputs[:,i,:,:] = torch.roll(inputs[:,i,:,:], shifts=(-1)*step*i, dims=2)
    return inputs

class HyPaNet(nn.Module):
    def __init__(self, in_nc=29, out_nc=8, channel=64):
        super(HyPaNet, self).__init__()
        self.fution = nn.Conv2d(in_nc, channel, 1, 1, 0, bias=True)
        self.down_sample = nn.Conv2d(channel, channel, 3, 2, 1, bias=True)
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.mlp = nn.Sequential(
                nn.Conv2d(channel, channel, 1, padding=0, bias=True),
                nn.ReLU(inplace=True),
                nn.Conv2d(channel, channel, 1, padding=0, bias=True),
                nn.ReLU(inplace=True),
                nn.Conv2d(channel, out_nc, 1, padding=0, bias=True),
                nn.Softplus())
        self.relu = nn.ReLU(inplace=True)
        self.out_nc = out_nc

    def forward(self, x):
        x = self.down_sample(self.relu(self.fution(x)))
        x = self.avg_pool(x)
        x = self.mlp(x) + 1e-6
        return x[:,:self.out_nc//2,:,:], x[:,self.out_nc//2:,:,:]
def Afun(x, AT, A, miu, sizeX):
    """
    :param x:
    :param AT:
    :param A:
    :param Dt:
    :param D:
    :param ro:
    :param sizeX:
    :return:
    """
    x = x.reshape(sizeX)
    y = AT(A(x))
    y = y + miu
    y = y.reshape(-1)
    return y
def CBR(conv:nn.Conv2d):
    """
    convolution
    BatchNorm
    ReLU
    """
    return nn.Sequential(
        conv,
        nn.BatchNorm2d(conv.out_channels),
        nn.ReLU(inplace=True),
    )

# global define
A = None
At = None
shift = None
shift_back = None

class PGU(nn.Module):
    "pan guided unrolling"
    def __init__(self, num_iterations,physical_operator):
        super(PGU, self).__init__()
        # define physical operator
        global A,At,shift,shift_back
        A, At, shift, shift_back = physical_operator

        # define newwork
        self.para_estimator = HyPaNet(in_nc=28, out_nc=num_iterations*2)
        self.fution = nn.Conv2d(56 + 1 + 28, 28, 1, padding=0, bias=True)
        self.num_iterations = num_iterations
        self.denoisers = nn.ModuleList([])
        for _ in range(num_iterations):
            self.denoisers.append(
                In2SET(in_dim=29, out_dim=28, dim=28, H=256, W=256, num_blocks=[1, 1, 1]),
            )
        self.pan_extractors = PANFeatureExTractor(
            in_channel=1,
            out_channels=[c//2 for c in [28, 28 * 2, 28 * 2 * 2]]
        )

    def initial(self, y,y2, Phi,A2):
        """
        :param y: [b,256,310]
        :param Phi: [b,28,256,310]
        :return: z0: [b,28,256,310]; alpha: [b, num_iterations]; beta: [b, num_iterations]
        """
        nC, step = 28, 2
        y = y / nC * 2
        bs,row,col = y.shape
        y_shift = torch.zeros(bs, nC, row, col).cuda().float()
        for i in range(nC):
            y_shift[:, i, :, step * i:step * i + col - (nC - 1) * step] = y[:, :, step * i:step * i + col - (nC - 1) * step]
        z = self.fution(torch.cat([y_shift, Phi, y2, A2], dim=1))
        alpha, beta = self.para_estimator(z)
        return z, alpha, beta

    def forward(self, y, forward_info=None):
        """
        :param y: [b,256,310]
        :param forward_info: ([b,28,256,310],[b,28,1,1])
        :return: z_crop: [b,28,256,256]
        """
        Phi,cameraSpectralResponse_cuda = forward_info
        y1, y2 = y[:, :, :310], y[:, :, 310:]

        y2 = y2.unsqueeze(1)
        y2_pad = F.pad(y2, [0, 310 - 256, 0, 0], mode='constant', value=0)
        # y2_pad = y2_pad.unsqueeze(1)
        A2 = torch.ones_like(Phi)
        A2[:, :, :, 256:] = 0
        A2 = A2 * cameraSpectralResponse_cuda

        # y = y1
        z0, alphas, betas = self.initial(y1, y2_pad, Phi, A2)

        z = shift_back(z0)
        sizeX = z.shape
        atb = At(y)

        b, c, h_inp, w_inp = y2.shape
        hb, wb = 16, 16
        pad_h = (hb - h_inp % hb) % hb
        pad_w = (wb - w_inp % wb) % wb
        pan = F.pad(y2, [0, pad_w, 0, pad_h], mode='reflect')

        pan_feas = self.pan_extractors(pan)
        for i in range(self.num_iterations):
            alpha, beta = alphas[:,i:i+1,:,:], betas[:,i:i+1,:,:]

            # data item
            self.pcgA = lambda j: Afun(j, At, A, alpha, sizeX)
            aux = atb + alpha * z
            x = z.reshape(-1)
            x = cg(self.pcgA, aux.reshape(-1), x, 1e-4, 5)
            x = x.reshape(sizeX)
            # prior item
            beta_repeat = beta.repeat(1,1,x.shape[2], x.shape[3])
            z = self.denoisers[i](torch.cat([x,beta_repeat], dim=1),pan_feas)

        return z[:, :, :, 0:256]


