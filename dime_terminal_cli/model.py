"""Causal Transformer: pre-norm, RMSNorm, RoPE, GQA, SwiGLU and KV cache."""
from dataclasses import dataclass
import math
from pathlib import Path
import torch
from torch import nn
from torch.nn import functional as F
from .config import ModelConfig

@dataclass
class CausalLMOutput:
    logits:torch.Tensor
    loss:torch.Tensor|None=None
    loss_sum:torch.Tensor|None=None
    target_count:int=0
    past_key_values:tuple|None=None

class RMSNorm(nn.Module):
    def __init__(self,width,eps=1e-5):
        super().__init__()
        self.weight=nn.Parameter(torch.ones(width))
        self.eps=eps

    def forward(self,x):
        value=x.float()*torch.rsqrt(x.float().square().mean(-1,keepdim=True)+self.eps)
        return value.to(x.dtype)*self.weight.to(x.dtype)

class RotaryEmbedding(nn.Module):
    def __init__(self,head_dim,theta=10000.0):
        super().__init__()
        self.head_dim=head_dim;self.theta=theta
        self.register_buffer("_device_anchor",torch.empty(0),persistent=False)

    @torch.no_grad()
    def forward(self,length,start_position=0):
        # Frequencies are rebuilt in FP32 even after model.half()/bfloat16().
        device=self._device_anchor.device
        position=torch.arange(start_position,start_position+length,device=device,dtype=torch.float32)
        frequency=self.theta**(-torch.arange(0,self.head_dim,2,device=device,dtype=torch.float32)/self.head_dim)
        angle=torch.outer(position,frequency)
        return angle.cos()[None,None,:,:],angle.sin()[None,None,:,:]

def apply_rope(x,cos,sin):
    first,second=x.float().chunk(2,dim=-1)
    return torch.cat((first*cos-second*sin,second*cos+first*sin),dim=-1).to(x.dtype)

class GroupedQueryAttention(nn.Module):
    def __init__(self,config):
        super().__init__()
        self.num_heads=config.num_attention_heads
        self.num_kv_heads=config.num_key_value_heads
        self.head_dim=config.head_dim
        self.dropout=config.dropout
        d=config.hidden_size;kv=self.num_kv_heads*self.head_dim
        self.q_proj=nn.Linear(d,d,bias=False)
        self.k_proj=nn.Linear(d,kv,bias=False)
        self.v_proj=nn.Linear(d,kv,bias=False)
        self.o_proj=nn.Linear(d,d,bias=False)

    def forward(self,x,cos,sin,attention_mask=None,past=None,use_cache=False):
        batch,length,width=x.shape
        q=self.q_proj(x).view(batch,length,self.num_heads,self.head_dim).transpose(1,2)
        k=self.k_proj(x).view(batch,length,self.num_kv_heads,self.head_dim).transpose(1,2)
        v=self.v_proj(x).view(batch,length,self.num_kv_heads,self.head_dim).transpose(1,2)
        q,k=apply_rope(q,cos,sin),apply_rope(k,cos,sin)
        past_length=0
        if past is not None:
            past_length=past[0].shape[2]
            if past[0].dtype!=k.dtype or past[1].dtype!=v.dtype:
                raise ValueError("Cache precision differs from current inference precision")
            k=torch.cat((past[0],k),dim=2)
            v=torch.cat((past[1],v),dim=2)
        total_length=k.shape[2]
        mask=None;causal=False
        if attention_mask is None and past_length==0:
            causal=True
        elif attention_mask is None and length==1:
            # Every cached key precedes the new query. is_causal=True would
            # align a 1 x (past+1) mask to the wrong corner.
            causal=False
        else:
            query_positions=torch.arange(past_length,past_length+length,device=x.device)
            key_positions=torch.arange(total_length,device=x.device)
            mask=(key_positions[None,:]<=query_positions[:,None])[None,None,:,:]
            if attention_mask is not None:
                mask=mask & attention_mask[:,None,None,:].bool()
        dropout=self.dropout if self.training else 0.0
        if x.device.type=="cuda":
            attended=F.scaled_dot_product_attention(q,k,v,attn_mask=mask,
                dropout_p=dropout,is_causal=causal,enable_gqa=True)
        else:
            groups=self.num_heads//self.num_kv_heads
            attended=F.scaled_dot_product_attention(
                q,k.repeat_interleave(groups,dim=1),v.repeat_interleave(groups,dim=1),
                attn_mask=mask,dropout_p=dropout,is_causal=causal)
        attended=attended.transpose(1,2).contiguous().view(batch,length,width)
        return self.o_proj(attended),(k,v) if use_cache else None

class SwiGLU(nn.Module):
    def __init__(self,config):
        super().__init__()
        d=config.hidden_size;f=config.intermediate_size
        self.gate_proj=nn.Linear(d,f,bias=False)
        self.up_proj=nn.Linear(d,f,bias=False)
        self.down_proj=nn.Linear(f,d,bias=False)

    def forward(self,x):
        return self.down_proj(F.silu(self.gate_proj(x))*self.up_proj(x))

class DecoderBlock(nn.Module):
    def __init__(self,config):
        super().__init__()
        self.attention_norm=RMSNorm(config.hidden_size,config.rms_norm_eps)
        self.attention=GroupedQueryAttention(config)
        self.feedforward_norm=RMSNorm(config.hidden_size,config.rms_norm_eps)
        self.feedforward=SwiGLU(config)
        self.dropout=nn.Dropout(config.dropout)

    def forward(self,x,cos,sin,attention_mask=None,past=None,use_cache=False):
        update,cache=self.attention(self.attention_norm(x),cos,sin,attention_mask,past,use_cache)
        x=x+self.dropout(update)
        return x+self.dropout(self.feedforward(self.feedforward_norm(x))),cache

class TerminalLlama(nn.Module):
    def __init__(self,config):
        super().__init__()
        self.config=config
        self.token_embedding=nn.Embedding(config.vocab_size,config.hidden_size)
        self.rope=RotaryEmbedding(config.head_dim,config.rope_theta)
        self.layers=nn.ModuleList(DecoderBlock(config) for _ in range(config.num_layers))
        self.final_norm=RMSNorm(config.hidden_size,config.rms_norm_eps)
        self.lm_head=nn.Linear(config.hidden_size,config.vocab_size,bias=False)
        if config.tie_word_embeddings:
            self.lm_head.weight=self.token_embedding.weight
        self.reset_parameters()

    def reset_parameters(self):
        seen=set()
        for module in self.modules():
            if isinstance(module,(nn.Embedding,nn.Linear)) and id(module.weight) not in seen:
                nn.init.normal_(module.weight,mean=0,std=self.config.initializer_range)
                seen.add(id(module.weight))
            elif isinstance(module,RMSNorm):
                nn.init.ones_(module.weight)
        residual_std=self.config.initializer_range/math.sqrt(2*self.config.num_layers)
        for layer in self.layers:
            nn.init.normal_(layer.attention.o_proj.weight,std=residual_std)
            nn.init.normal_(layer.feedforward.down_proj.weight,std=residual_std)

    @property
    def num_parameters(self):
        return sum(p.numel() for p in self.parameters())

    def forward(self,input_ids,labels=None,attention_mask=None,past_key_values=None,use_cache=False):
        if input_ids.ndim!=2 or 0 in input_ids.shape:
            raise ValueError("input_ids must have nonempty [batch,sequence] shape")
        if input_ids.dtype not in (torch.int64,torch.int32):
            raise ValueError("input_ids must be integer ids")
        batch,length=input_ids.shape
        past_length=0
        if past_key_values is not None:
            if len(past_key_values)!=self.config.num_layers:
                raise ValueError("Cache layer count mismatch")
            past_length=past_key_values[0][0].shape[2]
            shape=(batch,self.config.num_key_value_heads,past_length,self.config.head_dim)
            for pair in past_key_values:
                if len(pair)!=2 or any(tuple(value.shape)!=shape or value.device!=input_ids.device for value in pair):
                    raise ValueError("Invalid KV cache shape or device")
        if (use_cache or past_key_values is not None) and self.training:
            raise ValueError("KV cache requires model.eval()")
        if labels is not None and (use_cache or past_key_values is not None):
            raise ValueError("Cached forward is for generation, not supervised loss")
        if past_length+length>self.config.max_seq_len:
            raise ValueError("Input and cache exceed configured context")
        if attention_mask is not None:
            if tuple(attention_mask.shape)!=(batch,past_length+length):
                raise ValueError("Attention mask shape mismatch")
            if not bool(attention_mask[:,0].bool().all()):
                raise ValueError("This model supports right padding only")
            if bool((attention_mask[:,1:].bool() & ~attention_mask[:,:-1].bool()).any()):
                raise ValueError("Attention mask must use contiguous right padding")
        if labels is not None:
            if labels.shape!=input_ids.shape or labels.dtype!=torch.long or length<2:
                raise ValueError("Labels must match input positions with dtype long")
            if attention_mask is not None and bool((labels[~attention_mask.bool()]!=-100).any()):
                raise ValueError("Padding positions must have ignored labels")
        layer_attention_mask = None if labels is not None and past_key_values is None else attention_mask
        cos,sin=self.rope(length,past_length)
        x=self.token_embedding(input_ids)
        caches=[]
        for i,layer in enumerate(self.layers):
            past=past_key_values[i] if past_key_values is not None else None
            x,cache=layer(x,cos,sin,layer_attention_mask,past,use_cache)
            if use_cache:caches.append(cache)
        logits=self.lm_head(self.final_norm(x))
        loss=loss_sum=None;count=0
        if labels is not None:
            shifted=labels[:,1:]
            count=int((shifted!=-100).sum().item())
            if not count:raise ValueError("No supervised causal targets")
            loss_sum=F.cross_entropy(logits[:,:-1,:].reshape(-1,self.config.vocab_size).float(),
                                     shifted.reshape(-1),ignore_index=-100,reduction="sum")
            loss=loss_sum/count
        return CausalLMOutput(logits,loss,loss_sum,count,tuple(caches) if use_cache else None)

    @torch.inference_mode()
    def generate(self,input_ids,max_new_tokens=64,eos_token_id=2,temperature=0.0,
                 top_k=40,forbidden_token_ids=(),generator=None,use_cache=True,on_token=None):
        if input_ids.ndim!=2 or input_ids.shape[0]!=1 or input_ids.shape[1]<1:
            raise ValueError("Generation supports one nonempty prompt at a time")
        if isinstance(max_new_tokens,bool) or not isinstance(max_new_tokens,int) or max_new_tokens<0:
            raise ValueError("max_new_tokens must be nonnegative")
        if input_ids.shape[1]+max_new_tokens>self.config.max_seq_len:
            raise ValueError("Prompt plus generation budget exceeds context")
        if not math.isfinite(temperature) or temperature<0:
            raise ValueError("Invalid temperature")
        if top_k is not None and (isinstance(top_k,bool) or not isinstance(top_k,int) or top_k<1):
            raise ValueError("Invalid top_k")
        forbidden=tuple(sorted(set(forbidden_token_ids)))
        if eos_token_id in forbidden:
            raise ValueError("EOS must remain available")
        if any(i<0 or i>=self.config.vocab_size for i in forbidden):
            raise ValueError("Forbidden token id outside vocabulary")
        was_training=self.training;self.eval()
        generated=input_ids.clone();cache=None
        try:
            for _ in range(max_new_tokens):
                current=generated[:,-1:] if cache is not None else generated
                output=self(current,past_key_values=cache,use_cache=use_cache)
                cache=output.past_key_values if use_cache else None
                logits=output.logits[:,-1,:].float().clone()
                if forbidden:logits[:,list(forbidden)]=-torch.inf
                if temperature==0:
                    token=logits.argmax(-1,keepdim=True)
                else:
                    logits=logits/temperature
                    if top_k is not None:
                        threshold=torch.topk(logits,min(top_k,self.config.vocab_size)).values[:,-1:]
                        logits=logits.masked_fill(logits<threshold,-torch.inf)
                    token=torch.multinomial(torch.softmax(logits,dim=-1),1,generator=generator)
                token_id=int(token.item())
                generated=torch.cat((generated,token),dim=1)
                if on_token is not None:on_token(token_id)
                if token_id==eos_token_id:break
        finally:
            self.train(was_training)
        return generated

    def save_checkpoint(self,path):
        path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
        torch.save({"config":self.config.to_dict(),"model_state":self.state_dict()},path)

    @classmethod
    def from_checkpoint(cls,path,device="cpu"):
        state=torch.load(path,map_location="cpu",weights_only=True)
        model=cls(ModelConfig(**state["config"]))
        dtype=state["model_state"]["token_embedding.weight"].dtype
        model.to(device=device,dtype=dtype)
        model.load_state_dict(state["model_state"])
        return model
