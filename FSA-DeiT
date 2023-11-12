class Attention(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=False, attn_drop=0., proj_drop=0., depth_index=0):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim ** -0.5
        self.depth_index = depth_index

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

        self.W_Ne = nn.Parameter(torch.ones(num_heads, 1, 1))
        # self.W_s = nn.Parameter(torch.ones(num_heads, 1, 1))

    def forward(self, x):
        B, N, C = x.shape

        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(0)   # make torchscript happy (cannot use tensor as tuple)

        q = q - q.mean(dim=-1, keepdim=True)
        k = k - k.mean(dim=-1, keepdim=True)
        attn = (F.normalize(q, dim=-1) @ k.transpose(-2, -1))
        attn_P = attn.softmax(dim=-1)
        attn_N = (-attn).softmax(dim=-1)
        attn_all = (attn_P + torch.clamp(self.W_Ne , 0, 9) * attn_N) * attn
        x = (attn_all @ v).transpose(1, 2).reshape(B, N, C)

        x = self.proj(x)
        x = self.proj_drop(x)
        return x
