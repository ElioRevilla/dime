"""Local inference with streaming and measurable warm time to first token."""
import codecs
from contextlib import nullcontext
import math
from pathlib import Path
import time
import torch
from .config import ModelConfig
from .model import TerminalLlama
from .tokenization import TerminalTokenizer
from .data_audit import sha256_file

class ByteLevelStream:
    def __init__(self,tokenizer):
        visible=list(range(33,127))+list(range(161,173))+list(range(174,256))
        mapping={b:chr(b) for b in visible};offset=0
        for byte in range(256):
            if byte not in mapping:
                mapping[byte]=chr(256+offset);offset+=1
        self.inverse={char:byte for byte,char in mapping.items()}
        self.tokenizer=tokenizer
        self.decoder=codecs.getincrementaldecoder("utf-8")(errors="replace")
        self.closed=False

    def push(self,token_id):
        if self.closed:raise ValueError("Stream already finished")
        if token_id in self.tokenizer.special_ids.values():
            raise ValueError("Control token cannot be decoded as content")
        token=self.tokenizer.backend.id_to_token(token_id)
        if token is None:raise ValueError("Token outside vocabulary")
        raw=bytes(self.inverse[char] for char in token)
        return self.decoder.decode(raw,final=False)

    def finish(self):
        if self.closed:return ""
        self.closed=True
        return self.decoder.decode(b"",final=True)

def resolve_precision(model,precision="auto"):
    device=next(model.parameters()).device
    if precision=="auto":
        precision=("bf16" if torch.cuda.is_bf16_supported() else "fp16") if device.type=="cuda" else "fp32"
    if precision not in ("fp32","bf16","fp16"):raise ValueError("Unsupported precision")
    if device.type=="cpu" and precision!="fp32":raise ValueError("CPU inference uses FP32")
    if precision=="bf16" and not torch.cuda.is_bf16_supported():raise ValueError("BF16 unavailable")
    return precision

@torch.inference_mode()
def generate_response(model,tokenizer,instruction,task="bash",language="es",
                      max_new_tokens=64,precision="auto",on_text=None):
    device=next(model.parameters()).device
    precision=resolve_precision(model,precision)
    if isinstance(max_new_tokens,bool) or not isinstance(max_new_tokens,int) or max_new_tokens<1:
        raise ValueError("A positive generation budget is required")
    if device.type=="cuda":torch.cuda.synchronize()
    started=time.perf_counter()
    prompt=tokenizer.prompt_ids(instruction,task,language)
    remaining=model.config.max_seq_len-len(prompt)
    if remaining<1:raise ValueError("Request leaves no space for a response")
    budget=min(max_new_tokens,remaining)
    inputs=torch.tensor([prompt],dtype=torch.long,device=device)
    stream=ByteLevelStream(tokenizer);parts=[];first_ms=None;token_count=0;ended=False
    def emit(token_id):
        nonlocal first_ms,token_count,ended
        token_count+=1;ended=token_id==tokenizer.eos_token_id
        text=stream.finish() if ended else stream.push(token_id)
        if text:
            parts.append(text)
            if on_text is not None:on_text(text)
        # token.item() in generate synchronizes the selected GPU token.
        # Timing includes tokenization, transfer, decode and callback emission.
        if first_ms is None:first_ms=(time.perf_counter()-started)*1000
    dtype={"bf16":torch.bfloat16,"fp16":torch.float16}.get(precision)
    context=torch.autocast("cuda",dtype=dtype) if dtype is not None else nullcontext()
    forbidden=[i for i in tokenizer.special_ids.values() if i!=tokenizer.eos_token_id]
    with context:
        model.generate(inputs,max_new_tokens=budget,eos_token_id=tokenizer.eos_token_id,
                       temperature=0,forbidden_token_ids=forbidden,use_cache=True,on_token=emit)
    tail=stream.finish()
    if tail:
        parts.append(tail)
        if on_text is not None:on_text(tail)
    if device.type=="cuda":torch.cuda.synchronize()
    text="".join(parts)
    return {"instruction":instruction,"task":task,"language":language,"text":text,
            "prompt_tokens":len(prompt),"generated_tokens_including_eos":token_count,
            "ended_with_eos":ended,"hit_limit":not ended,
            "finish_reason":"eos" if ended else "context_limit" if budget<max_new_tokens else "token_limit",
            "time_to_first_token_ms":first_ms,"total_generation_ms":(time.perf_counter()-started)*1000,
            "precision":precision,"command_executed":False}

class Copilot:
    def __init__(self,model,tokenizer,precision="auto",checkpoint_step=None):
        if model.config.vocab_size!=tokenizer.vocab_size:raise ValueError("Vocabulary mismatch")
        self.model=model;self.tokenizer=tokenizer
        self.precision=resolve_precision(model,precision);self.checkpoint_step=checkpoint_step
        self.model.eval()

    @classmethod
    def from_checkpoint(cls,checkpoint,tokenizer_path=None,device="auto",precision="auto"):
        if device=="auto":device="cuda" if torch.cuda.is_available() else "cpu"
        state=torch.load(checkpoint,map_location="cpu",weights_only=True)
        expected=state.get("data_signature",{}).get("tokenizer")
        if tokenizer_path is None:
            base=Path(__file__).resolve().parents[1]/"tokenizer"
            matches=[path for path in base.glob("*/tokenizer.json") if expected and sha256_file(path)==expected]
            if not matches:
                raise ValueError("Cannot identify checkpoint tokenizer; pass its path explicitly")
            tokenizer_path=sorted(matches)[0]
        tokenizer=TerminalTokenizer(tokenizer_path)
        if expected is not None and expected!=sha256_file(tokenizer.path):
            raise ValueError("Checkpoint was trained with a different tokenizer")
        model=TerminalLlama(ModelConfig(**state["config"]))
        dtype=state["model_state"]["token_embedding.weight"].dtype if device!="cpu" else torch.float32
        model.to(device=device,dtype=dtype);model.load_state_dict(state["model_state"])
        return cls(model,tokenizer,precision,state.get("global_step"))

    def suggest(self,instruction,task="bash",language="es",max_new_tokens=None,on_text=None):
        if max_new_tokens is None:max_new_tokens=192 if self.tokenizer.vocab_size==266 else 64
        return generate_response(self.model,self.tokenizer,instruction,task,language,
                                 max_new_tokens,self.precision,on_text)

    def warmup(self,iterations=3):
        for _ in range(iterations):
            self.suggest('Muestra todo el texto de "notas.txt".',"bash","es",max_new_tokens=1)

    def benchmark_ttft(self,requests,repeats=50,warmup=5,prompt_limit=128):
        if repeats<1 or warmup<0 or not requests:raise ValueError("Invalid benchmark setup")
        for request in requests:
            if len(self.tokenizer.prompt_ids(request["instruction"],request["task"],request["language"]))>prompt_limit:
                raise ValueError("Benchmark request exceeds the declared prompt limit")
        for i in range(warmup):
            self.suggest(**requests[i%len(requests)],max_new_tokens=1)
        values=[]
        for i in range(repeats):
            result=self.suggest(**requests[i%len(requests)],max_new_tokens=1)
            values.append(result["time_to_first_token_ms"])
        ordered=sorted(values)
        percentile=lambda p:ordered[max(0,math.ceil(len(ordered)*p)-1)]
        return {"metric":"warm_local_time_to_first_token_ms","samples":len(values),
                "p50_ms":percentile(.5),"p95_ms":percentile(.95),"min_ms":min(values),"max_ms":max(values),
                "warmup_iterations":warmup,"max_prompt_tokens":prompt_limit,
                "max_new_tokens_per_request":1,"checkpoint_step":self.checkpoint_step,
                "model_loaded_before_timing":True,"tokenization_and_callback_included":True,
                "terminal_repaint_or_remote_network_not_measured":True,
                "cached_answers_used":False,"device":str(next(self.model.parameters()).device),
                "precision":self.precision,"target_ms":50,
                "trained_checkpoint_meets_target":bool(self.checkpoint_step and percentile(.95)<50)}

def evaluate_exact(copilot,source_rows,max_examples=32):
    """Greedy sequence match only; never executes generated commands or regexes."""
    if max_examples<1:raise ValueError("max_examples must be positive")
    results=[];counts={}
    for row in source_rows[:max_examples]:
        prediction=copilot.suggest(row["instruction"],row["task"],row["language"])
        exact=prediction["text"]==row["output"] and prediction["ended_with_eos"]
        group=f"{row['task']}:{row['language']}"
        values=counts.setdefault(group,{"examples":0,"exact_matches":0})
        values["examples"]+=1;values["exact_matches"]+=int(exact)
        results.append({"id":row["id"],"reference":row["output"],"exact_match":exact,**prediction})
    return {"examples":len(results),"exact_matches":sum(row["exact_match"] for row in results),
            "exact_match_rate":sum(row["exact_match"] for row in results)/len(results) if results else None,
            "ended_with_eos_rate":sum(row["ended_with_eos"] for row in results)/len(results) if results else None,
            "by_task_language":counts,"commands_executed":False,
            "metric_note":"Exact string comparison; equivalent alternative commands/regexes may score false.",
            "predictions":results}

def export_for_inference(checkpoint,destination):
    state=torch.load(checkpoint,map_location="cpu",weights_only=True)
    payload={key:state[key] for key in ("config","model_state")}
    for key in ("data_signature","global_step"):
        if key in state:payload[key]=state[key]
    destination=Path(destination)
    destination.parent.mkdir(parents=True,exist_ok=True)
    temporary=destination.with_suffix(destination.suffix+".tmp")
    torch.save(payload,temporary);temporary.replace(destination)
    return destination
