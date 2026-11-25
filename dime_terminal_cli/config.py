"""Architecture configuration; all weights start randomly initialized."""
from dataclasses import asdict,dataclass
import json
import math
from pathlib import Path

@dataclass(frozen=True)
class ModelConfig:
    vocab_size:int=1580
    hidden_size:int=384
    num_layers:int=8
    num_attention_heads:int=6
    num_key_value_heads:int=2
    intermediate_size:int=1024
    max_seq_len:int=256
    rope_theta:float=10000.0
    rms_norm_eps:float=1e-5
    dropout:float=0.05
    initializer_range:float=0.02
    tie_word_embeddings:bool=True

    def __post_init__(self):
        for name in ("vocab_size","hidden_size","num_layers","num_attention_heads",
                     "num_key_value_heads","intermediate_size","max_seq_len"):
            value=getattr(self,name)
            if isinstance(value,bool) or not isinstance(value,int) or value<1:
                raise ValueError(f"{name} must be a positive integer")
        if self.hidden_size%self.num_attention_heads:
            raise ValueError("hidden_size must be divisible by query heads")
        if self.num_attention_heads%self.num_key_value_heads:
            raise ValueError("query heads must be divisible by KV heads")
        if self.head_dim%2:
            raise ValueError("RoPE requires an even head dimension")
        for name in ("rope_theta","rms_norm_eps","initializer_range"):
            if not math.isfinite(getattr(self,name)) or getattr(self,name)<=0:
                raise ValueError(f"Invalid {name}")
        if not math.isfinite(self.dropout) or not 0<=self.dropout<1:
            raise ValueError("dropout must be in [0,1)")
        if not isinstance(self.tie_word_embeddings,bool):
            raise ValueError("tie_word_embeddings must be Boolean")

    @property
    def head_dim(self):
        return self.hidden_size//self.num_attention_heads

    @property
    def expected_parameters(self):
        d=self.hidden_size
        kv=self.num_key_value_heads*self.head_dim
        per_block=2*d*d+2*d*kv+3*d*self.intermediate_size+2*d
        return self.vocab_size*d*(1 if self.tie_word_embeddings else 2)+self.num_layers*per_block+d

    def to_dict(self):
        return asdict(self)

    @classmethod
    def load(cls,path):
        return cls(**json.loads(Path(path).read_text(encoding="utf-8")))
