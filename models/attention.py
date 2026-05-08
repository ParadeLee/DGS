import torch
import torch.nn as nn
import torch.nn.functional as F
import math

def get_attn_fn(attn_fn_name):
    name = attn_fn_name.lower()
    options = {
        'sdlora': Attention_sdLoRA,
        'glora': Attention_GLoRA,
        'hlora': Attention_HLoRA,
        'elora': Attention_LoRA,
        'duallora': Attention_DualLoRA,
    }
    return options[name]


class Attention(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0.):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        # NOTE scale factor was wrong in my original version, can set manually to be compat with prev weights
        self.scale = qk_scale or head_dim ** -0.5
        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)
        self.attn_gradients = None
        self.attention_map = None

        self.cur_matrix = torch.zeros(dim ,dim)
        self.n_cur_matrix = 0
        
    def save_attn_gradients(self, attn_gradients):
        self.attn_gradients = attn_gradients
        
    def get_attn_gradients(self):
        return self.attn_gradients
    
    def save_attention_map(self, attention_map):
        self.attention_map = attention_map
        
    def get_attention_map(self):
        return self.attention_map
    
    def forward(self, x, register_hook=False, prompt=None,get_cur_feat=False):

        if get_cur_feat:
            # average self-attention
            self.cur_matrix = (self.cur_matrix*self.n_cur_matrix + torch.bmm(x.detach().permute(0, 2, 1), x.detach()).sum(dim=0).cpu())/(self.n_cur_matrix + x.shape[0]*x.shape[1])
            self.n_cur_matrix += x.shape[0]*x.shape[1]
    
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]   # make torchscript happy (cannot use tensor as tuple)


        if prompt is not None:
            prompt = prompt.permute(1, 0, 3, 2, 4).contiguous() # 2, B, num_heads, prompt_length, C // num_heads
            key_prefix = prompt[0] # B, num_heads, prompt_length, embed_dim // num_heads
            value_prefix = prompt[1] # B, num_heads, prompt_length, embed_dim // num_heads

            expected_shape = (B, self.num_heads, C // self.num_heads)
            
            assert (key_prefix.shape[0], key_prefix.shape[1], key_prefix.shape[3]) == expected_shape, f'key_prefix.shape: {key_prefix.shape} not match k.shape: {k.shape}'
            assert (value_prefix.shape[0], value_prefix.shape[1], value_prefix.shape[3]) == expected_shape, f'value_prefix.shape: {value_prefix.shape} not match v.shape: {v.shape}'

            k = torch.cat([key_prefix, k], dim=2)
            v = torch.cat([value_prefix, v], dim=2)

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)
                
        if register_hook:
            self.save_attention_map(attn)
            attn.register_hook(self.save_attn_gradients)        

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class Adapter(nn.Module):
    def __init__(self, config=None, d_model=None, bottleneck=None, dropout=0.0, init_option="lora",
                 adapter_scalar="1.0", adapter_layernorm_option="in"):
        super().__init__()
        self.n_embd = config.d_model if d_model is None else d_model
        self.down_size = config.attn_bn if bottleneck is None else bottleneck

        # _before
        self.adapter_layernorm_option = adapter_layernorm_option

        self.adapter_layer_norm_before = None
        # layer_norm
        if adapter_layernorm_option == "in" or adapter_layernorm_option == "out":
            self.adapter_layer_norm_before = nn.LayerNorm(self.n_embd)

        if adapter_scalar == "learnable_scalar":
            self.scale = nn.Parameter(torch.ones(1))
        else:
            self.scale = float(adapter_scalar)
        self.down_proj = nn.Linear(self.n_embd, self.down_size)
        self.non_linear_func = nn.ReLU()
        self.up_proj = nn.Linear(self.down_size, self.n_embd)
        self.dropout = dropout
        if init_option == "bert":
            raise NotImplementedError
        elif init_option == "lora":
            with torch.no_grad():
                nn.init.kaiming_uniform_(self.down_proj.weight, a=math.sqrt(5))
                nn.init.zeros_(self.up_proj.weight)
                nn.init.zeros_(self.down_proj.bias)
                nn.init.zeros_(self.up_proj.bias)

    def forward(self, x, add_residual=True, residual=None):
        residual = x if residual is None else residual
        B, N, C = x.shape
        if self.adapter_layernorm_option == 'in':
            x = self.adapter_layer_norm_before(x)
        down = self.down_proj(x)
        down = self.non_linear_func(down)
        down = nn.functional.dropout(down, p=self.dropout, training=self.training)
        up = self.up_proj(down)

        up = up * self.scale

        if self.adapter_layernorm_option == 'out':
            up = self.adapter_layer_norm_before(up)

        if add_residual:
            output = up + residual
        else:
            output = up

        return output


class Attention_LoRA(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0., r=64, n_tasks=10):
        super().__init__()
        self.num_heads = num_heads
        self.dim = dim
        head_dim = dim // num_heads
        # NOTE scale factor was wrong in my original version, can set manually to be compat with prev weights
        self.scale = qk_scale or head_dim ** -0.5
        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)
        self.attn_gradients = None
        self.attention_map = None
        self.rank = r

        self.lora_A_q = nn.ModuleList([nn.Linear(dim, r, bias=False) for _ in range(n_tasks)])
        self.lora_B_q = nn.ModuleList([nn.Linear(r, dim, bias=False) for _ in range(n_tasks)])
        self.lora_A_v = nn.ModuleList([nn.Linear(dim, r, bias=False) for _ in range(n_tasks)])
        self.lora_B_v = nn.ModuleList([nn.Linear(r, dim, bias=False) for _ in range(n_tasks)])
        self.rank = r
        self.random_init = True

        self.matrix = torch.zeros(dim ,dim)
        self.n_matrix = 0
        self.cur_matrix = torch.zeros(dim ,dim)
        self.n_cur_matrix = 0

    
    def init_param(self):
        for t in range(len(self.lora_A_q)):
            # self.lora_A_q[t] = self.init_lora(self.lora_A_q[t])
            # self.lora_A_v[t] = self.init_lora(self.lora_A_v[t])
            nn.init.kaiming_uniform_(self.lora_A_q[t].weight, a=math.sqrt(5))
            nn.init.kaiming_uniform_(self.lora_A_v[t].weight, a=math.sqrt(5))
            kr = 1.0 / self.dim
            nn.init.zeros_(self.lora_B_q[t].weight)
            nn.init.zeros_(self.lora_B_v[t].weight)
            nn.init.sparse_(self.lora_B_q[t].weight, sparsity=(1 - kr))
            nn.init.sparse_(self.lora_B_v[t].weight, sparsity=(1 - kr))
    
    def init_lora(self, lora, is_b=True):
        if self.random_init:
            random_matrix = torch.rand(self.dim, self.rank)
            q, r = torch.linalg.qr(random_matrix)
            with torch.no_grad():
                lora.weight.copy_(q.T)
            scaling_factor = 1.
            lora.weight.data *= scaling_factor
        else:
            nn.init.kaiming_uniform_(lora.weight, a=math.sqrt(5))
        return lora


    def save_attn_gradients(self, attn_gradients):
        self.attn_gradients = attn_gradients
        
    def get_attn_gradients(self):
        return self.attn_gradients
    
    def save_attention_map(self, attention_map):
        self.attention_map = attention_map
        
    def get_attention_map(self):
        return self.attention_map
    
    def forward(self, x, task, register_hook=False, get_feat=False, get_cur_feat=False):
        # x size [128,197,768]
        if get_feat:
            self.matrix = (self.matrix*self.n_matrix + torch.bmm(x.detach().permute(0, 2, 1), x.detach()).sum(dim=0).cpu())/(self.n_matrix + x.shape[0]*x.shape[1])
            self.n_matrix += x.shape[0]*x.shape[1]
        if get_cur_feat:
            # average self-attention
            self.cur_matrix = (self.cur_matrix*self.n_cur_matrix + torch.bmm(x.detach().permute(0, 2, 1), x.detach()).sum(dim=0).cpu())/(self.n_cur_matrix + x.shape[0]*x.shape[1])
            self.n_cur_matrix += x.shape[0]*x.shape[1]

        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]   # make torchscript happy (cannot use tensor as tuple)

        # insert lora
        weight_q = torch.stack([torch.mm(self.lora_B_q[t].weight, self.lora_A_q[t].weight) for t in range(task+1)], dim=0).sum(dim=0)
        weight_v = torch.stack([torch.mm(self.lora_B_v[t].weight, self.lora_A_v[t].weight) for t in range(task+1)], dim=0).sum(dim=0)
        q = q + F.linear(x, weight_q).reshape(B, N, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)
        v = v + F.linear(x, weight_v).reshape(B, N, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)
                
        if register_hook:
            self.save_attention_map(attn)
            attn.register_hook(self.save_attn_gradients)        

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x

    def get_matrix(self, task):
        matrix_k = torch.mm(self.lora_B_k[task].weight, self.lora_A_k[task].weight)
        matrix_v = torch.mm(self.lora_B_v[task].weight, self.lora_A_v[task].weight)
        return matrix_k, matrix_v
    
    def get_pre_matrix(self, task):
        with torch.no_grad():
            weight_k = torch.stack([torch.mm(self.lora_B_k[t].weight, self.lora_A_k[t].weight) for t in range(task)], dim=0).sum(dim=0)
            weight_v = torch.stack([torch.mm(self.lora_B_v[t].weight, self.lora_A_v[t].weight) for t in range(task)], dim=0).sum(dim=0)
        return weight_k, weight_v
    

class Attention_GLoRA(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0., r=64, n_tasks=10):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        # NOTE scale factor was wrong in my original version, can set manually to be compat with prev weights
        self.scale = qk_scale or head_dim ** -0.5
        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)
        self.attn_gradients = None
        self.attention_map = None
        self.rank = r

        self.glora_A_k = nn.Linear(dim, r, bias=False)
        self.glora_B_k = nn.Linear(r, dim, bias=False)
        self.glora_A_v = nn.Linear(dim, r, bias=False)
        self.glora_B_v = nn.Linear(r, dim, bias=False)
        self.rank = r

        self.matrix = torch.zeros(dim ,dim)
        self.n_matrix = 0
        self.cur_matrix = torch.zeros(dim ,dim)
        self.n_cur_matrix = 0

    def init_param(self):
        nn.init.kaiming_uniform_(self.glora_A_k.weight, a=math.sqrt(5))
        nn.init.kaiming_uniform_(self.glora_A_v.weight, a=math.sqrt(5))
        nn.init.zeros_(self.glora_B_k.weight)
        nn.init.zeros_(self.glora_B_v.weight)

    def save_attn_gradients(self, attn_gradients):
        self.attn_gradients = attn_gradients
        
    def get_attn_gradients(self):
        return self.attn_gradients
    
    def save_attention_map(self, attention_map):
        self.attention_map = attention_map
        
    def get_attention_map(self):
        return self.attention_map
    
    def forward(self, x, task, register_hook=False, get_feat=False,get_cur_feat=False):
        # x size [128,197,768]
        if get_feat:
            self.matrix = (self.matrix*self.n_matrix + torch.bmm(x.detach().permute(0, 2, 1), x.detach()).sum(dim=0).cpu())/(self.n_matrix + x.shape[0]*x.shape[1])
            self.n_matrix += x.shape[0]*x.shape[1]
        if get_cur_feat:
            # average self-attention
            self.cur_matrix = (self.cur_matrix*self.n_cur_matrix + torch.bmm(x.detach().permute(0, 2, 1), x.detach()).sum(dim=0).cpu())/(self.n_cur_matrix + x.shape[0]*x.shape[1])
            self.n_cur_matrix += x.shape[0]*x.shape[1]

        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]   # make torchscript happy (cannot use tensor as tuple)

        # insert lora
        weight_k = torch.mm(self.glora_B_k.weight, self.glora_A_k.weight)
        weight_v = torch.mm(self.glora_B_v.weight, self.glora_A_v.weight)
        k = k + F.linear(x, weight_k).reshape(B, N, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)
        v = v + F.linear(x, weight_v).reshape(B, N, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)
                
        if register_hook:
            self.save_attention_map(attn)
            attn.register_hook(self.save_attn_gradients)        

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class Attention_HLoRA(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0., r=64, n_tasks=10):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        # NOTE scale factor was wrong in my original version, can set manually to be compat with prev weights
        self.scale = qk_scale or head_dim ** -0.5
        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)
        self.attn_gradients = None
        self.attention_map = None
        self.rank = r

        self.scaling_factor = nn.Parameter(torch.Tensor([0.8]))
        self.glora_A_k = nn.Linear(dim, r, bias=False)
        self.elora_B_k = nn.ModuleList([nn.Linear(r, dim, bias=False) for _ in range(n_tasks)])
        self.glora_A_v = nn.Linear(dim, r, bias=False)
        self.elora_B_v = nn.ModuleList([nn.Linear(r, dim, bias=False) for _ in range(n_tasks)])

        self.matrix = torch.zeros(dim ,dim)
        self.n_matrix = 0
        self.cur_matrix = torch.zeros(dim ,dim)
        self.n_cur_matrix = 0

    def init_param(self):
        nn.init.kaiming_uniform_(self.glora_A_k.weight, a=math.sqrt(5))
        nn.init.kaiming_uniform_(self.glora_A_v.weight, a=math.sqrt(5))
        for t in range(len(self.elora_B_k)):
            nn.init.zeros_(self.elora_B_k[t].weight)
            nn.init.zeros_(self.elora_B_v[t].weight)

    def save_attn_gradients(self, attn_gradients):
        self.attn_gradients = attn_gradients
        
    def get_attn_gradients(self):
        return self.attn_gradients
    
    def save_attention_map(self, attention_map):
        self.attention_map = attention_map
        
    def get_attention_map(self):
        return self.attention_map
    
    def forward(self, x, task, register_hook=False, get_feat=False,get_cur_feat=False):
        # x size [128,197,768]
        if get_feat:
            self.matrix = (self.matrix*self.n_matrix + torch.bmm(x.detach().permute(0, 2, 1), x.detach()).sum(dim=0).cpu())/(self.n_matrix + x.shape[0]*x.shape[1])
            self.n_matrix += x.shape[0]*x.shape[1]
        if get_cur_feat:
            # average self-attention
            self.cur_matrix = (self.cur_matrix*self.n_cur_matrix + torch.bmm(x.detach().permute(0, 2, 1), x.detach()).sum(dim=0).cpu())/(self.n_cur_matrix + x.shape[0]*x.shape[1])
            self.n_cur_matrix += x.shape[0]*x.shape[1]

        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]   # make torchscript happy (cannot use tensor as tuple)

        # insert lora
        # W + A(sum B)h
        weight_k = torch.mm(torch.stack([self.elora_B_k[t].weight for t in range(task+1)], dim=0).sum(dim=0), self.glora_A_k.weight)
        weight_v = torch.mm(torch.stack([self.elora_B_v[t].weight for t in range(task+1)], dim=0).sum(dim=0), self.glora_A_v.weight)
        k = k + F.linear(x, weight_k).reshape(B, N, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)
        v = v + F.linear(x, weight_v).reshape(B, N, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)
                
        if register_hook:
            self.save_attention_map(attn)
            attn.register_hook(self.save_attn_gradients)        

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class Attention_DualLoRA(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0., r=64, n_tasks=1):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        # NOTE scale factor was wrong in my original version, can set manually to be compat with prev weights
        self.scale = qk_scale or head_dim ** -0.5
        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)
        self.attn_gradients = None
        self.attention_map = None

        self.lora_A_k = nn.Linear(dim, r, bias=False)
        self.lora_B_k = nn.Linear(r, dim, bias=False)
        self.lora_A_v = nn.Linear(dim, r, bias=False)
        self.lora_B_v = nn.Linear(r, dim, bias=False)
        self.rank = r

        self.kmatrix_weight = None
        self.vmatrix_weight = None
    
    def init_param(self):
        nn.init.kaiming_uniform_(self.lora_A_k.weight, a=math.sqrt(5))
        nn.init.kaiming_uniform_(self.lora_A_v.weight, a=math.sqrt(5))
        nn.init.zeros_(self.lora_B_k.weight)
        nn.init.zeros_(self.lora_B_v.weight)

    def init_Lora_param(self):
        nn.init.kaiming_uniform_(self.lora_A_k.weight, a=math.sqrt(5))
        nn.init.kaiming_uniform_(self.lora_A_v.weight, a=math.sqrt(5))
        nn.init.zeros_(self.lora_B_k.weight)
        nn.init.zeros_(self.lora_B_v.weight)

    def save_attn_gradients(self, attn_gradients):
        self.attn_gradients = attn_gradients
        
    def get_attn_gradients(self):
        return self.attn_gradients
    
    def save_attention_map(self, attention_map):
        self.attention_map = attention_map
        
    def get_attention_map(self):
        return self.attention_map
    
    def forward(self, x, istrain=False, register_hook=False, get_feat=False,get_cur_feat=False):
        # x size [128,197,768]
        if get_feat:
            self.matrix = (self.matrix*self.n_matrix + torch.bmm(x.detach().permute(0, 2, 1), x.detach()).sum(dim=0).cpu())/(self.n_matrix + x.shape[0]*x.shape[1])
            self.n_matrix += x.shape[0]*x.shape[1]
        if get_cur_feat:
            # average self-attention
            self.cur_matrix = (self.cur_matrix*self.n_cur_matrix + torch.bmm(x.detach().permute(0, 2, 1), x.detach()).sum(dim=0).cpu())/(self.n_cur_matrix + x.shape[0]*x.shape[1])
            self.n_cur_matrix += x.shape[0]*x.shape[1]

        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]   # make torchscript happy (cannot use tensor as tuple)

        if self.kmatrix_weight is None:
            weight_k = torch.stack([torch.mm(self.lora_B_k.weight, self.lora_A_k.weight)], dim=0).sum(dim=0)
            weight_v = torch.stack([torch.mm(self.lora_B_v.weight, self.lora_A_v.weight)], dim=0).sum(dim=0)
        # insert lora
        else:
            if istrain:
                weight_k = torch.stack([self.kmatrix_weight, torch.mm(self.lora_B_k.weight, self.lora_A_k.weight)], dim=0).sum(dim=0)
                weight_v = torch.stack([self.vmatrix_weight, torch.mm(self.lora_B_v.weight, self.lora_A_v.weight)], dim=0).sum(dim=0)
            else:
                weight_k = self.kmatrix_weight
                weight_v = self.vmatrix_weight
        k = k + F.linear(x, weight_k).reshape(B, N, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)
        v = v + F.linear(x, weight_v).reshape(B, N, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)
                
        if register_hook:
            self.save_attention_map(attn)
            attn.register_hook(self.save_attn_gradients)        

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x

    @torch.no_grad()
    def get_lora_matrix(self):
        matrix_k = torch.mm(self.lora_B_k.weight, self.lora_A_k.weight)
        matrix_v = torch.mm(self.lora_B_v.weight, self.lora_A_v.weight)
        return matrix_k, matrix_v


class Attention_sdLoRA(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0., r=64, n_tasks=10):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        # NOTE scale factor was wrong in my original version, can set manually to be compat with prev weights
        self.scale = qk_scale or head_dim ** -0.5
        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)
        self.attn_gradients = None
        self.attention_map = None
        self.rank = r

        self.scaling_factor = nn.ParameterList([ nn.Parameter(torch.Tensor([1.0])) for _ in range(n_tasks) ])
        self.lora_A_q = nn.ModuleList([nn.Linear(dim, r, bias=False) for _ in range(n_tasks)])
        self.lora_B_q = nn.ModuleList([nn.Linear(r, dim, bias=False) for _ in range(n_tasks)])
        self.lora_A_v = nn.ModuleList([nn.Linear(dim, r, bias=False) for _ in range(n_tasks)])
        self.lora_B_v = nn.ModuleList([nn.Linear(r, dim, bias=False) for _ in range(n_tasks)])

    def init_param(self):
        for t in range(len(self.lora_A_q)):
            nn.init.kaiming_uniform_(self.lora_A_q[t].weight, a=math.sqrt(5))
            nn.init.kaiming_uniform_(self.lora_A_v[t].weight, a=math.sqrt(5))
            nn.init.zeros_(self.lora_B_q[t].weight)
            nn.init.zeros_(self.lora_B_v[t].weight)

    def forward(self, x, task, register_hook=False, get_feat=False, get_cur_feat=False):

        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]  # easy to use
        delta_q, delta_v = 0, 0
        for t in range(task):
            if t == 0:
                delta_q = self.scaling_factor[t] * \
                    (self.lora_B_q[t](self.lora_A_q[t](x)) / (torch.norm(self.lora_A_q[t].weight) * torch.norm(self.lora_B_q[t].weight)))
                delta_v = self.scaling_factor[t] * \
                    (self.lora_B_v[t](self.lora_A_v[t](x)) / (torch.norm(self.lora_A_v[t].weight) * torch.norm(self.lora_B_v[t].weight)))
            else:
                delta_q += self.scaling_factor[t] * \
                    (self.lora_B_q[t](self.lora_A_q[t](x)) / (torch.norm(self.lora_A_q[t].weight) * torch.norm(self.lora_B_q[t].weight)))
                delta_v += self.scaling_factor[t] * \
                    (self.lora_B_v[t](self.lora_A_v[t](x)) / (torch.norm(self.lora_A_v[t].weight) * torch.norm(self.lora_B_v[t].weight)))
        delta_q += self.scaling_factor[task] * ( self.lora_B_q[task](self.lora_A_q[task](x) ))
        delta_v += self.scaling_factor[task] * ( self.lora_B_v[task](self.lora_A_v[task](x) ))
        delta_q = delta_q.reshape(B, N, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)
        delta_v = delta_v.reshape(B, N, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)
        q = q + delta_q
        v = v + delta_v

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)

        return x


class Attention_deLoRA(nn.Module):

    def __init__(self, dim, num_heads=8, qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0., r=64, n_tasks=10):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.dim = dim
        # NOTE scale factor was wrong in my original version, can set manually to be compat with prev weights
        self.scale = qk_scale or head_dim ** -0.5
        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)
        self.attn_gradients = None
        self.attention_map = None
        self.rank = r

        self.lora_A_q = nn.ModuleList([nn.Linear(dim, r, bias=False) for _ in range(n_tasks)])
        self.lora_B_q = nn.ModuleList([nn.Linear(r, dim, bias=False) for _ in range(n_tasks)])
        self.lora_A_v = nn.ModuleList([nn.Linear(dim, r, bias=False) for _ in range(n_tasks)])
        self.lora_B_v = nn.ModuleList([nn.Linear(r, dim, bias=False) for _ in range(n_tasks)])

        self.matrix = torch.zeros(dim, dim)
        self.n_matrix = 0
        self.cur_matrix = torch.zeros(dim, dim)
        self.n_cur_matrix = 0

    def init_param(self):
        for t in range(len(self.lora_A_q)):
            nn.init.kaiming_uniform_(self.lora_A_q[t].weight, a=math.sqrt(5))
            nn.init.kaiming_uniform_(self.lora_A_v[t].weight, a=math.sqrt(5))
            kr = 1.0 / self.dim
            nn.init.zeros_(self.lora_B_q[t].weight)
            nn.init.zeros_(self.lora_B_v[t].weight)
            nn.init.sparse_(self.lora_B_q[t].weight, sparsity=(1 - kr))
            nn.init.sparse_(self.lora_B_v[t].weight, sparsity=(1 - kr))

    def forward(self, x, task, register_hook=False, get_feat=False, get_cur_feat=False):

        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]  # easy to use
        x0 = x.clone().detach()
        if get_cur_feat:
            rate = 1
             # 存储上一层lora层输出的处理后特征x，即每个一层lora层的输入特征，经过处理
            # x0是不可求导版本
            self.cur_matrix = (self.cur_matrix * self.n_cur_matrix + torch.bmm(x0.detach().permute(0, 2, 1), x0.detach()).sum(dim=0).cpu()) / \
                (rate * (self.n_cur_matrix + x0.shape[0] * x0.shape[1]))
            self.n_cur_matrix += x0.shape[0] * x0.shape[1]

            weight_q_old = torch.stack(
                [torch.mm(self.lora_B_q[t].weight, self.lora_A_q[t].weight) for t in range(task)],
                dim=0).sum(dim=0)
            weight_v_old = torch.stack(
                [torch.mm(self.lora_B_v[t].weight, self.lora_A_v[t].weight) for t in range(task)],
                dim=0).sum(dim=0)
            q = q - F.linear(x, weight_q_old).reshape(B, N, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)
            v = v - F.linear(x, weight_v_old).reshape(B, N, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3) 

        else:
            if task > -0.5:
                weight_q = torch.stack(
                    [torch.mm(self.lora_B_q[t].weight, self.lora_A_q[t].weight) for t in range(task + 1)],
                    dim=0).sum(dim=0)
                weight_v = torch.stack(
                    [torch.mm(self.lora_B_v[t].weight, self.lora_A_v[t].weight) for t in range(task + 1)],
                    dim=0).sum(dim=0)
                q = q + F.linear(x, weight_q).reshape(B, N, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)
                v = v + F.linear(x, weight_v).reshape(B, N, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)

        return x


class Attention_newLoRA(nn.Module):

    def __init__(self, dim, num_heads=8, qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0., r=64, n_tasks=10):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.dim = dim
        # NOTE scale factor was wrong in my original version, can set manually to be compat with prev weights
        self.scale = qk_scale or head_dim ** -0.5
        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)
        self.attn_gradients = None
        self.attention_map = None
        self.rank = r
        self.random_init = True

        self.lora_A_q = nn.ModuleList([nn.Linear(dim, r, bias=False) for _ in range(n_tasks)])
        self.lora_B_q = nn.ModuleList([nn.Linear(r, dim, bias=False) for _ in range(n_tasks)])
        self.lora_A_v = nn.ModuleList([nn.Linear(dim, r, bias=False) for _ in range(n_tasks)])
        self.lora_B_v = nn.ModuleList([nn.Linear(r, dim, bias=False) for _ in range(n_tasks)])

        self.matrix = torch.zeros(dim, dim)
        self.n_matrix = 0
        self.cur_matrix = torch.zeros(dim, dim)
        self.n_cur_matrix = 0
    
    def init_param(self):
        for t in range(len(self.lora_A_q)):
            # self.lora_A_q[t] = self.init_lora(self.lora_A_q[t])
            # self.lora_A_v[t] = self.init_lora(self.lora_A_v[t])
            nn.init.kaiming_uniform_(self.lora_A_q[t].weight, a=math.sqrt(5))
            nn.init.kaiming_uniform_(self.lora_A_v[t].weight, a=math.sqrt(5))
            kr = 1.0 / self.dim
            nn.init.zeros_(self.lora_B_q[t].weight)
            nn.init.zeros_(self.lora_B_v[t].weight)
            nn.init.sparse_(self.lora_B_q[t].weight, sparsity=(1 - kr))
            nn.init.sparse_(self.lora_B_v[t].weight, sparsity=(1 - kr))
    
    def forward(self, x, task, register_hook=False, get_feat=False, get_cur_feat=False):

        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]  # easy to use
        x0 = x.clone().detach()
        if get_cur_feat:
            rate = 1
            # 存储上一层lora层输出的处理后特征x，即每个一层lora层的输入特征，经过处理
            # x0是不可求导版本
            self.cur_matrix = (self.cur_matrix * self.n_cur_matrix + torch.bmm(x0.detach().permute(0, 2, 1), x0.detach()).sum(dim=0).cpu()) / \
                (rate * (self.n_cur_matrix + x0.shape[0] * x0.shape[1]))
            self.n_cur_matrix += x0.shape[0] * x0.shape[1]
            d_out, d_in = self.lora_B_q[0].weight.shape[0], self.lora_A_q[0].weight.shape[1]
            if task > 0.5:
                weight_q_old = torch.stack(
                    [torch.mm(self.lora_B_q[0].weight, self.lora_A_q[0].weight)],
                    dim=0).sum(dim=0)
                weight_v_old = torch.stack(
                    [torch.mm(self.lora_B_v[0].weight, self.lora_A_v[0].weight)],
                    dim=0).sum(dim=0)
            weight_q_null = torch.zeros(d_out, d_in, device=x.device)
            weight_v_null = torch.zeros(d_out, d_in, device=x.device)
            q = q - F.linear(x, weight_q_null).reshape(B, N, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)
            v = v - F.linear(x, weight_v_null).reshape(B, N, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3) 

        else:
            if task > -0.5:
                weight_q = torch.stack(
                    [torch.mm(self.lora_B_q[t].weight, self.lora_A_q[t].weight) for t in range(task + 1)],
                    dim=0).sum(dim=0)
                weight_v = torch.stack(
                    [torch.mm(self.lora_B_v[t].weight, self.lora_A_v[t].weight) for t in range(task + 1)],
                    dim=0).sum(dim=0)
                q = q + F.linear(x, weight_q).reshape(B, N, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)
                v = v + F.linear(x, weight_v).reshape(B, N, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)

        return x
    


class LoRAConv2d(nn.Module):
    def __init__(self, conv: nn.Conv2d, r: int = 4, lora_alpha: float = 1.0):
        super().__init__()
        self.conv = conv
        self.r = r
        self.lora_alpha = lora_alpha
        self.scaling = lora_alpha / r

        # Freeze original conv
        for param in self.conv.parameters():
            param.requires_grad = False

        dim = conv.in_channels
        in_channels = conv.in_channels
        out_channels = conv.out_channels
        kernel_h, kernel_w = conv.kernel_size
        K = kernel_h * kernel_w

        # LoRA matrices: A (r x Cin*K), B (Cout x r)
        self.lora_A = nn.Parameter(torch.zeros(r, in_channels * K))
        self.lora_B = nn.Parameter(torch.zeros(out_channels, r))
        self.cur_matrix = torch.zeros(dim, dim)
        self.n_cur_matrix = 0
        self.get_cur_feat = False

        # Initialize
        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
        nn.init.zeros_(self.lora_B)

    def forward(self, x):
        # Original output
        out = self.conv(x)
        x0 = x.clone().detach()
        if self.get_cur_feat:
            B, C, H, W = x0.shape
            x_flat = x0.permute(0, 2, 3, 1).reshape(B * H * W, C)
            gram = x_flat.t() @ x_flat  # [C, C]
            with torch.no_grad():
                old_count = self.n_cur_matrix
                new_count = B * H * W + old_count
                self.cur_matrix = (self.cur_matrix * old_count + gram.cpu()) / new_count
                self.n_cur_matrix = new_count

        # Compute LoRA delta
        batch, _, h_out, w_out = out.shape
        kh, kw = self.conv.kernel_size
        ph, pw = self.conv.padding
        sh, sw = self.conv.stride

        # Unfold input to [B, Cin*K, L]
        x_unfold = F.unfold(x, kernel_size=(kh, kw), padding=(ph, pw), stride=(sh, sw))  # [B, Cin*K, H*W]
        lora_delta = self.lora_B @ (self.lora_A @ x_unfold)  # [B, Cout, H*W]
        lora_delta = lora_delta.view(batch, -1, h_out, w_out)

        return out + lora_delta * self.scaling
