import torch
import math
from torch import nn
## shape:(batch, seq_len, head, head_dim)
query = torch.randn(10, 128, 8, 128)
key = torch.randn(10, 128, 2, 128)
value = torch.randn(10, 128, 2, 128)

# groups = query_head // key_head
groups = query.shape[-2] // key.shape[-2]

# key/value的扩展， 以便于矩阵乘法

# 定义输入x， n_rep是需要重复的次数，在这里一般是组数
def repeat_kv(hidden_states:torch.Tensor, n_rep: int) -> torch.Tensor:
    batch, num_key_value_heads, slen, head_dim = hidden_states.shape
    if n_rep == 1:
        return hidden_states
    hidden_states = hidden_states[:,:,None,:,:].expand(batch, num_key_value_heads,n_rep, slen,head_dim)
    return hidden_states.reshape(batch, num_key_value_heads * n_rep, slen, head_dim)

#(bs, head, seq_len, head_dim)
query = query.transpose(1, 2)
key = repeat_kv(key.transpose(1, 2), 4)
value = repeat_kv(value.transpose(1, 2), 4)
scores = torch.matmul(query, key.transpose(2, 3)) / math.sqrt(query.shape[-1])
scores = torch.nn.functional.softmax(scores, dim=-1)

out = torch.matmul(scores, value)
#上一步转置了，还得转回去
out = out.transpose(1, 2)


class Qwen2RotaryEmbedding(nn.Module):
    def __init__(self, dim, max_position_embeddings=2048, base=10000, device=None):
        super().__init__()
        self.dim = dim
        self.max_position_embeddings = max_position_embeddings
        self.base = base
        
        # 旋转角
        inv_freq = 1.0 / (self.base ** (torch.arange(0,self.dim, 2, dtype=torch.int64).float().to(device)/self.dim))
        self.register_buffer("inv_freq", inv_freq, persistent=False)

        self._set_cos_sin_cache(
            seq_len=max_position_embeddings, device = self.inv_freq.device, dtype=torch.get_default_dtype()
        )
        print('end')

    def _set_cos_sin_cache(self, seq_len, device, dtype):
        self.max_seq_len_cached = seq_len
        t = torch.arange(self.max_seq_len_cached, device=device, dtype=torch.int64).type_as(self.inv_freq)

        freqs = torch.outer(t,self.inv_freq)

        emb = torch.cat((freqs, freqs), dim=-1)
        # 生成角度信息 利用注册机制生成self.cos_cacheed sin_cached
        self.register_buffer("cos_cached", emb.cos().to(dtype), persistent=False)
        self.register_buffer("sin_cached", emb.sin().to(dtype), persistent=False)
    
    def forward(self, x, seq_len=None):
        if seq_len > self.max_seq_cached:
            self._set_cos_sin_cached(seq_len = seq_len, device=x.device, dtype=x.dtype)
        return(
            self.cos_cached[:seq_len].to(dtype=x.dtype),
            self.sin_cached[:seq_len].to(dtype=x.dtype),
        )

    def rotate_half(x):
        x1 = x[..., :x.shape[-1] // 2]
        x2 = x[..., x.shape[-1] // 2:]
        return torch.cat((-x2,x1),dim=-1)
    
    def apply_rotate_pos_emb(q,k,cos,sin,position_ids, unsqueeze_dim=1):
        cos = cos[position_ids].unsqueeze(unsqueeze_dim)
        sin = sin[position_ids].unsqueeze(unsqueeze_dim)
        q_embed = (q * cos) + (rotate_half(q) * sin)
        k_embed = (k * cos) + (rotate_half(k) * sin)
        return q_embed, k_embed

rope = Qwen2RotaryEmbedding(dim=64, max_position_embeddings=1024 )
# rope.apply_rotate_pos_emb(query, key, rope.cos_cache,)
