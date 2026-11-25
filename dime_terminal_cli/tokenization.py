"""Byte-level BPE and explicit causal supervision for the terminal copilot."""
from collections import Counter
from datetime import datetime, timezone
import importlib.metadata
import json
import math
from pathlib import Path
import tempfile

from tokenizers import Tokenizer, decoders, models, pre_tokenizers, trainers

from .data_audit import sha256_file, write_jsonl
from .dataset_v1 import read_jsonl

SPECIAL_TOKENS = [
    "<|pad|>", "<|bos|>", "<|eos|>", "<|unk|>",
    "<|task:bash|>", "<|task:regex|>",
    "<|lang:es|>", "<|lang:en|>", "<|input|>", "<|output|>",
]
IGNORE_INDEX = -100
FORMAT_VERSION = "terminal-sft-v1"

class TerminalTokenizer:
    def __init__(self, path):
        self.path = Path(path)
        metadata_path = self.path.with_name("metadata.json")
        metadata = json.loads(metadata_path.read_text(encoding="utf-8")) if metadata_path.is_file() else {}
        self.supported_languages = tuple(metadata.get("supported_languages", ["es", "en"]))
        if not self.supported_languages or any(x not in ("es", "en") for x in self.supported_languages):
            raise ValueError("Invalid supported languages")
        self.backend = Tokenizer.from_file(str(self.path))
        self.special_ids = {text:self.backend.token_to_id(text) for text in SPECIAL_TOKENS}
        if any(value is None for value in self.special_ids.values()):
            raise ValueError("Missing required control tokens")
        if list(self.special_ids.values()) != list(range(len(SPECIAL_TOKENS))):
            raise ValueError("Unexpected special-token ids")
        # True splits literal occurrences of special strings as ordinary content.
        # Header/control ids are inserted explicitly by this wrapper.
        self.backend.encode_special_tokens = True
        self.pad_token_id = self.special_ids["<|pad|>"]
        self.bos_token_id = self.special_ids["<|bos|>"]
        self.eos_token_id = self.special_ids["<|eos|>"]
        self.unk_token_id = self.special_ids["<|unk|>"]

    @property
    def vocab_size(self):
        return self.backend.get_vocab_size()

    def encode_text(self, text):
        if not isinstance(text,str):
            raise TypeError("Text must be a string")
        ids = self.backend.encode(text,add_special_tokens=False).ids
        if any(value in self.special_ids.values() for value in ids):
            raise ValueError("Ordinary content produced a control or unknown token")
        return ids

    def decode_text(self, ids):
        ids=list(ids)
        if any(value in self.special_ids.values() for value in ids):
            raise ValueError("decode_text accepts content ids only")
        return self.backend.decode(ids,skip_special_tokens=False)

    def debug_decode(self, ids):
        return self.backend.decode(list(ids),skip_special_tokens=False)

    def prompt_ids(self, instruction, task, language):
        if task not in ("bash","regex") or language not in self.supported_languages:
            raise ValueError("Unsupported task or language")
        if not isinstance(instruction,str) or not instruction.strip():
            raise ValueError("Empty request")
        return [
            self.bos_token_id,
            self.special_ids[f"<|task:{task}|>"],
            self.special_ids[f"<|lang:{language}|>"],
            self.special_ids["<|input|>"],
            *self.encode_text(instruction),
            self.special_ids["<|output|>"],
        ]

    def supervised_example(self, row, max_length=256):
        if isinstance(max_length,bool) or not isinstance(max_length,int) or max_length<2:
            raise ValueError("max_length must be an integer of at least 2")
        output=row["output"]
        if not isinstance(output,str) or not output:
            raise ValueError("Empty response")
        if row["task"]=="regex" and (
            row.get("target_dialect")!="python_re" or row.get("match_mode")!="fullmatch"
            or row.get("regex_flags")!=["ASCII"]
        ):
            raise ValueError("Regex rows must follow the v1 Python/ASCII/fullmatch contract")
        if row["task"]=="bash" and row.get("target_dialect")!="bash_gnu_linux":
            raise ValueError("Unsupported Bash dialect")
        prompt=self.prompt_ids(row["instruction"],row["task"],row["language"])
        response=self.encode_text(output)+[self.eos_token_id]
        input_ids=prompt+response
        if len(input_ids)>max_length:
            raise ValueError(f"Example {row.get('id')} needs {len(input_ids)} tokens, limit {max_length}; no truncation applied")
        labels=[IGNORE_INDEX]*len(prompt)+response
        return {
            "id":row["id"],"case_id":row["case_id"],"group_id":row["group_id"],
            "task":row["task"],"language":row["language"],
            "input_ids":input_ids,"labels":labels,
            "prompt_length":len(prompt),"response_start":len(prompt),
            "sequence_length":len(input_ids),"supervised_tokens":len(response),
        }

def train_tokenizer(train_file,output_dir,vocab_size=2048,min_frequency=2,supported_languages=("es","en")):
    if not supported_languages or any(x not in ("es","en") for x in supported_languages):
        raise ValueError("Invalid supported languages")
    train_file=Path(train_file).resolve()
    output_dir=Path(output_dir).resolve()
    if train_file.name in ("validation.jsonl","test.jsonl"):
        raise ValueError("The tokenizer must be trained on train only")
    if isinstance(vocab_size,bool) or not isinstance(vocab_size,int) or vocab_size<266:
        raise ValueError("Vocabulary must include the 256-byte alphabet and 10 controls")
    if isinstance(min_frequency,bool) or not isinstance(min_frequency,int) or min_frequency<1:
        raise ValueError("min_frequency must be positive")
    if output_dir.exists():
        raise FileExistsError(f"Tokenizer version exists: {output_dir}; choose another version")
    before=sha256_file(train_file)
    backend=Tokenizer(models.BPE(unk_token="<|unk|>"))
    backend.pre_tokenizer=pre_tokenizers.ByteLevel(add_prefix_space=False,use_regex=True)
    backend.decoder=decoders.ByteLevel()
    trainer=trainers.BpeTrainer(
        vocab_size=vocab_size,min_frequency=min_frequency,
        special_tokens=SPECIAL_TOKENS,
        initial_alphabet=sorted(pre_tokenizers.ByteLevel.alphabet()),
        show_progress=False,
    )
    count=0;characters=0;source_strings=0
    def texts():
        nonlocal count,characters,source_strings
        # Only this training file is opened by the fitting iterator.
        for row in read_jsonl(train_file):
            count+=1
            for key in ("instruction","output"):
                text=row[key]
                if not isinstance(text,str) or not text:
                    raise ValueError("Invalid training text")
                characters+=len(text);source_strings+=1
                yield text
    backend.train_from_iterator(texts(),trainer=trainer)
    if count==0: raise ValueError("Empty training data")
    if sha256_file(train_file)!=before: raise ValueError("Training data changed during tokenizer fit")
    output_dir.parent.mkdir(parents=True,exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=".tokenizer-build-",dir=output_dir.parent) as temporary:
        stage=Path(temporary).resolve()
        assert stage.is_relative_to(output_dir.parent)
        path=stage/"tokenizer.json"
        backend.save(str(path),pretty=True)
        fitted=TerminalTokenizer(path)
        if fitted.vocab_size>vocab_size:
            raise AssertionError("Unexpected vocabulary size")
        metadata={
            "format_version":FORMAT_VERSION,"algorithm":"byte-level BPE",
            "requested_vocab_size":vocab_size,"vocab_size":fitted.vocab_size,
            "min_frequency":min_frequency,"byte_alphabet_size":256,
            "special_tokens":fitted.special_ids,
            "trained_from_scratch":True,
            "training_file":str(train_file),"training_sha256":before,
            "supported_languages":list(supported_languages),
            "training_examples":count,"training_strings":source_strings,
            "training_characters":characters,
            "validation_used_for_fitting":False,"test_used_for_fitting":False,
            "normalization":"None; content is preserved exactly, including Unicode normalization form.",
            "literal_control_strings":"Encoded as ordinary content; header ids inserted explicitly.",
            "tokenizers_version":importlib.metadata.version("tokenizers"),
            "tokenizer_sha256":sha256_file(path),
            "created_at_utc":datetime.now(timezone.utc).isoformat(),
        }
        (stage/"metadata.json").write_text(json.dumps(metadata,ensure_ascii=False,indent=2)+"\n",encoding="utf-8")
        stage.replace(output_dir)
    return TerminalTokenizer(output_dir/"tokenizer.json"),metadata

def lengths_summary(values):
    values=sorted(values)
    if not values: return {"count":0}
    percentile=lambda p:values[max(0,math.ceil(len(values)*p)-1)]
    return {"min":values[0],"p50":percentile(.5),"p95":percentile(.95),
            "p99":percentile(.99),"max":values[-1],"mean":sum(values)/len(values)}

def export_supervised(tokenizer,corpus_dir,output_dir,max_length=256,prompt_benchmark_limit=128):
    corpus_dir=Path(corpus_dir).resolve();output_dir=Path(output_dir).resolve()
    if output_dir.exists():
        raise FileExistsError(f"Encoded version exists: {output_dir}; choose another version")
    manifest=json.loads((corpus_dir/"manifest.json").read_text(encoding="utf-8"))
    source_files={split:corpus_dir/f"{split}.jsonl" for split in ("train","validation","test")}
    before={split:sha256_file(path) for split,path in source_files.items()}
    for split,digest in before.items():
        if digest!=manifest["splits"][split]["sha256"]:
            raise ValueError(f"Changed source split: {split}")
    tokenizer_sha=sha256_file(tokenizer.path)
    tokenizer_metadata=json.loads((tokenizer.path.parent/"metadata.json").read_text(encoding="utf-8"))
    if tokenizer_metadata["training_sha256"]!=before["train"]:
        raise ValueError("Tokenizer was fitted on different training data")
    output_dir.parent.mkdir(parents=True,exist_ok=True)
    statistics={};total_roundtrips=0
    with tempfile.TemporaryDirectory(prefix=".encoded-build-",dir=output_dir.parent) as temporary:
        stage=Path(temporary).resolve()
        assert stage.is_relative_to(output_dir.parent)
        for split,path in source_files.items():
            encoded=[];prompt_lengths=[];response_lengths=[];sequence_lengths=[]
            task_languages=Counter();unknown=0
            for row in read_jsonl(path):
                example=tokenizer.supervised_example(row,max_length)
                boundary=example["response_start"];ids=example["input_ids"];labels=example["labels"]
                assert labels[:boundary]==[IGNORE_INDEX]*boundary
                assert labels[boundary:]==ids[boundary:]
                assert labels[-1]==tokenizer.eos_token_id
                assert ids[boundary-1]==tokenizer.special_ids["<|output|>"]
                assert tokenizer.decode_text(ids[4:boundary-1])==row["instruction"]
                assert tokenizer.decode_text(ids[boundary:-1])==row["output"]
                assert len(ids)==len(labels)
                assert sum(value!=IGNORE_INDEX for value in labels)==example["supervised_tokens"]
                unknown+=ids.count(tokenizer.unk_token_id)
                total_roundtrips+=2
                prompt_lengths.append(boundary)
                response_lengths.append(example["supervised_tokens"])
                sequence_lengths.append(len(ids))
                task_languages[f"{row['task']}:{row['language']}"]+=1
                encoded.append(example)
            if unknown: raise AssertionError("Unknown tokens in encoded corpus")
            destination=stage/f"{split}.jsonl"
            write_jsonl(destination,encoded)
            statistics[split]={
                "rows":len(encoded),"sha256":sha256_file(destination),"file":f"{split}.jsonl",
                "source_sha256":before[split],"input_tokens":sum(sequence_lengths),
                "prompt_tokens":sum(prompt_lengths),"supervised_tokens":sum(response_lengths),
                "unknown_tokens":unknown,"task_language_pairs":dict(task_languages),
                "prompt_length":lengths_summary(prompt_lengths),
                "response_length_including_eos":lengths_summary(response_lengths),
                "sequence_length":lengths_summary(sequence_lengths),
                "prompts_over_benchmark_limit":sum(n>prompt_benchmark_limit for n in prompt_lengths),
            }
        if any(sha256_file(path)!=before[split] for split,path in source_files.items()):
            raise AssertionError("Source corpus changed during export")
        if sha256_file(tokenizer.path)!=tokenizer_sha:
            raise AssertionError("Tokenizer changed during export")
        report={
            "format_version":FORMAT_VERSION,"max_sequence_length":max_length,
            "splits":statistics,"tokenizer_sha256":tokenizer_sha,
            "tokenizer_vocab_size":tokenizer.vocab_size,
            "special_tokens":tokenizer.special_ids,"ignore_index":IGNORE_INDEX,
            "supervision":"response tokens plus EOS only; prompt and padding labels are -100",
            "label_alignment":"labels preserve input positions; causal model must shift internally exactly once",
            "packing":"one request/response per example; no cross-document packing",
            "padding":"dynamic right padding at batching time",
            "truncation":"none; oversized examples raise",
            "roundtrip_checks":total_roundtrips,"all_roundtrips_exact":True,
            "unknown_tokens":0,"source_files_unchanged":True,
            "tokenizer_fitted_on_train_only":True,
            "prompt_benchmark_limit":prompt_benchmark_limit,
            "model_trained":False,"latency_target_measured":False,
        }
        (stage/"manifest.json").write_text(json.dumps(report,ensure_ascii=False,indent=2)+"\n",encoding="utf-8")
        stage.replace(output_dir)
    return report
