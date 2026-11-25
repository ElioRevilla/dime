"""Build a bounded, behavior-checked bilingual dataset v1."""
from collections import Counter, defaultdict
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import re
import unicodedata
from . import bash_tasks, regex_tasks
from .bash_validation import validate_cases
from .data_audit import sha256_file, write_jsonl

def read_jsonl(path):
    with Path(path).open(encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                yield json.loads(line)

def fingerprint(text):
    return re.sub(r"\s+"," ",unicodedata.normalize("NFC",text)).strip().casefold()

class Components:
    def __init__(self,keys): self.parents={key:key for key in keys}
    def find(self,key):
        while self.parents[key]!=key:
            self.parents[key]=self.parents[self.parents[key]]
            key=self.parents[key]
        return key
    def join(self,a,b):
        a,b=self.find(a),self.find(b)
        if a!=b:
            first,second=sorted([a,b])
            self.parents[second]=first

def group_and_split(rows,cases,seed=123):
    case_map={case["case_id"]:case for case in cases}
    groups=Components(case_map)
    seen_outputs={}; seen_inputs={}
    for row in rows:
        cid=row["case_id"]
        output_key=(row["task"],row["output"])
        input_key=(row["task"],row["language"],fingerprint(row["instruction"]))
        if output_key in seen_outputs: groups.join(cid,seen_outputs[output_key])
        else: seen_outputs[output_key]=cid
        if input_key in seen_inputs: groups.join(cid,seen_inputs[input_key])
        else: seen_inputs[input_key]=cid
    members=defaultdict(list)
    for case_id in case_map:
        members[groups.find(case_id)].append(case_id)
    strata=defaultdict(list)
    for group,ids in members.items():
        # Whole task families are not held out. This is held-out argument/case evaluation.
        family=min(case_map[cid]["family"] for cid in ids)
        task=case_map[ids[0]]["task"]
        strata[(task,family)].append(group)
    assignments={}
    train_only_strata=[]
    for stratum,keys in sorted(strata.items()):
        keys.sort(key=lambda key:hashlib.sha256(f"{seed}:{key}".encode()).hexdigest())
        if len(keys)<10:
            assignments.update({key:"train" for key in keys})
            train_only_strata.append({"task":stratum[0],"family":stratum[1],"groups":len(keys)})
        else:
            n=max(1,len(keys)//10)
            assignments.update({key:"test" for key in keys[:n]})
            assignments.update({key:"validation" for key in keys[n:2*n]})
            assignments.update({key:"train" for key in keys[2*n:]})
    result={"train":[],"validation":[],"test":[]}
    exact_seen={}; duplicates=0
    for row in rows:
        group=groups.find(row["case_id"])
        row["group_id"]=group
        split=assignments[group]
        exact=(row["task"],row["language"],row["instruction"],row["output"])
        if exact in exact_seen:
            existing=exact_seen[exact]
            existing["case_ids"]=sorted(set(existing["case_ids"]+[row["case_id"]]))
            duplicates+=1
            continue
        row["case_ids"]=[row["case_id"]]
        exact_seen[exact]=row
        row["split"]=split
        result[split].append(row)
    for values in result.values():
        values.sort(key=lambda row:row["id"])
    verify_splits(result)
    return result,{"group_count":len(members),"duplicate_rows_removed":duplicates,
                   "small_train_only_strata":train_only_strata,
                   "split_seed":seed,
                   "split_claim":"held-out semantic cases/arguments; not unseen task families"}

def verify_splits(splits):
    seen_groups={}; seen_outputs={}; seen_inputs={}; seen_ids=set()
    for split,rows in splits.items():
        for row in rows:
            if row["id"] in seen_ids: raise AssertionError("Duplicate row id")
            seen_ids.add(row["id"])
            if not row["instruction"] or not row["output"]: raise AssertionError("Empty row")
            if "{{" in row["output"]: raise AssertionError("Unresolved template")
            keys=[
                (seen_groups,row["group_id"]),
                (seen_outputs,(row["task"],row["output"])),
                (seen_inputs,(row["task"],row["language"],fingerprint(row["instruction"]))),
            ]
            for seen,key in keys:
                if key in seen and seen[key]!=split:
                    raise AssertionError(("Cross-split leakage",key,seen[key],split))
                seen[key]=split
    return {"cross_split_group_overlap":0,"cross_split_exact_output_overlap":0,
            "cross_split_normalized_instruction_overlap":0,
            "semantic_equivalence_of_different_regexes_not_proven":True}

def deduplicate_source_pool(root):
    root=Path(root)
    destination=root/"datasets/review/v1"
    destination.mkdir(parents=True,exist_ok=True)
    unique={}; duplicates=[]
    for row in read_jsonl(root/"datasets/normalized/v0/candidates.jsonl"):
        key=(row["task"],row["language"],row["instruction"],row["output"])
        ref={field:row.get(field) for field in [
            "id","source","source_revision","source_file","source_line","license_spdx","attribution_url"
        ]}
        if key in unique:
            unique[key]["provenance"].append(ref)
            duplicates.append({"duplicate_id":row["id"],"retained_id":unique[key]["id"]})
        else:
            unique[key]={
                "id":row["id"],"task":row["task"],"language":row["language"],
                "instruction":row["instruction"],"output":row["output"],
                "target_dialect":row["target_dialect"],"provenance":[ref],
                "status":"reserve_unvalidated_not_in_v1_training",
            }
    write_jsonl(destination/"deduplicated_source_pool.jsonl",unique.values())
    write_jsonl(destination/"duplicates.jsonl",duplicates)
    return {"source_rows":sum(len(row["provenance"]) for row in unique.values()),
            "unique_source_pairs":len(unique),"duplicate_rows":len(duplicates),
            "reserve_file":"datasets/review/v1/deduplicated_source_pool.jsonl",
            "reserve_is_in_training":False}

def provenance_for_bash(case,root,sources):
    tool=case["source_tool"]
    prefix=root/"datasets/raw/tldr"
    refs=[]
    for language,folder in [("en","pages"),("es","pages.es")]:
        relative=f"{folder}/common/{tool}.md"
        if (prefix/relative).is_file():
            refs.append({
                "source":"tldr","revision":sources["tldr"]["revision"],
                "source_file":relative,"language":language,"license_spdx":"CC-BY-4.0",
                "url":f"https://github.com/tldr-pages/tldr/blob/{sources['tldr']['revision']}/{relative}",
            })
    if not refs: raise AssertionError(("Missing source documentation",tool))
    return refs

def build_dataset(root,seed=123,case_limit=64):
    root=Path(root).resolve()
    source_manifest=json.loads((root/"datasets/normalized/v0/manifest.json").read_text(encoding="utf-8"))
    sources=json.loads((root/"datasets/sources.json").read_text(encoding="utf-8"))
    candidate_file=root/"datasets/normalized/v0/candidates.jsonl"
    before_hash=sha256_file(candidate_file)
    if before_hash!=source_manifest["candidates_file_sha256"]:
        raise ValueError("The source inventory hash changed")
    out=root/"datasets/processed/v1"
    out.mkdir(parents=True,exist_ok=True)
    print("Deduplicating source inventory...",flush=True)
    reserve=deduplicate_source_pool(root)
    bash_cases=bash_tasks.cases(case_limit)
    print(f"Validating {len(bash_cases)} authored Bash cases in fixtures...",flush=True)
    bash_checks,bash_report=validate_cases(bash_cases,root)
    regex_cases=regex_tasks.cases(case_limit)+regex_tasks.adapted_cases()
    print(f"Validating {len(regex_cases)} Python regex cases...",flush=True)
    regex_checks={case["case_id"]:regex_tasks.validate(case) for case in regex_cases}
    original_kb={row["source_line"]:row for row in read_jsonl(root/"datasets/normalized/v0/deep_regex.jsonl")
                 if row["original_corpus"]=="KB13"}
    all_cases=bash_cases+regex_cases
    rows=[]
    for case in all_cases:
        cid=case["case_id"]
        if case["task"]=="bash":
            case["validation"]=bash_checks[cid]
            case["provenance"]=provenance_for_bash(case,root,sources)
            case["authoring"]="project_authored_bilingual_templates_with_explicit_arguments"
            dialect="bash_gnu_linux"
        else:
            case["validation"]=regex_checks[cid]
            dialect="python_re"
            if case["family"].startswith("kb13_"):
                source=original_kb[case["source_line"]]
                case["provenance"]=[{
                    "source":"deep_regex","source_id":source["id"],"revision":source["source_revision"],
                    "source_file":source["source_file"],"source_line":source["source_line"],
                    "original_instruction":source["instruction"],"original_output":source["output"],
                    "license_spdx":"MIT",
                    "transformation":"manual_subset_adaptation_to_ASCII_single_line_fullmatch_and_authored_es_en_requests",
                }]
                case["authoring"]="adapted_KB13_subset_with_authored_bilingual_descriptions"
            else:
                case["provenance"]=[{"source":"project_authored",
                                     "definition_file":"terminal_copilot/regex_tasks.py"}]
                case["authoring"]="project_authored_parameterized_regex_cases"
        for language in ["es","en"]:
            prompts=(bash_tasks.prompts(case["family"],case["parameters"],language)
                     if case["task"]=="bash" else regex_tasks.prompts(case,language))
            for index,instruction in enumerate(prompts):
                rows.append({
                    "id":f"{cid}:{language}:{index}","case_id":cid,
                    "task":case["task"],"language":language,"family":case["family"],
                    "instruction":instruction,"output":case["output"],
                    "target_dialect":dialect,
                    "match_mode":"fullmatch" if case["task"]=="regex" else None,
                    "regex_flags":["ASCII"] if case["task"]=="regex" else None,
                    "validation_case_file":"cases.jsonl",
                    "authoring":case["authoring"],
                    "provenance":case["provenance"],
                })
    splits,split_report=group_and_split(rows,all_cases,seed)
    stats={}
    for split,values in splits.items():
        path=out/f"{split}.jsonl"
        write_jsonl(path,values)
        stats[split]={
            "rows":len(values),"case_groups":len({r["group_id"] for r in values}),
            "tasks":dict(Counter(r["task"] for r in values)),
            "task_language_pairs":dict(Counter(f"{r['task']}:{r['language']}" for r in values)),
            "file":path.relative_to(root).as_posix(),"sha256":sha256_file(path),
        }
    group_by_case={cid:r["group_id"] for values in splits.values() for r in values for cid in r["case_ids"]}
    for case in all_cases: case["group_id"]=group_by_case[case["case_id"]]
    write_jsonl(out/"cases.jsonl",all_cases)
    issues=[
      {"source_id":"tldr:pages.es/common/tail.md:17","status":"excluded_original",
       "reason":"Description requests bytes, but original template uses -n/--lines."},
      {"source_id":original_kb[2]["id"],"status":"excluded_original",
       "reason":"Description says only letters agde; target literal agde is ambiguous/inconsistent for that task. No automatic correction applied."},
    ]
    write_jsonl(root/"datasets/review/v1/confirmed_or_suspected_issues.jsonl",issues)
    examples=[]
    for split,values in splits.items():
        for task in ["bash","regex"]:
            for language in ["es","en"]:
                example=next(row for row in values if row["task"]==task and row["language"]==language)
                examples.append(example)
    write_jsonl(out/"samples.jsonl",examples)
    report={
        "created_at_utc":datetime.now(timezone.utc).isoformat(),
        "version":"v1","training_ready_for_bounded_experiment":True,
        "general_terminal_copilot_ready":False,
        "construction":"controlled_parameterized_cases_and_authored_paraphrases; not all downloaded examples",
        "total_rows":sum(len(values) for values in splits.values()),
        "parameterized_cases":len(all_cases),"unique_case_ids":len({c["case_id"] for c in all_cases}),
        "case_limit_per_family":case_limit,"splits":stats,
        "tasks":dict(Counter(case["task"] for case in all_cases)),
        "bash_families":bash_tasks.KINDS,"regex_families":regex_tasks.KINDS,
        "adapted_KB13_cases":len(regex_tasks.KB13),
        "languages":["es","en"],
        "scope":{"bash":"query/read/list/filter/count/sort; no arbitrary command generation",
                 "regex":"Python re.fullmatch, re.ASCII, ASCII single-line inputs",
                 "locale":"C","paraphrases_per_case_per_language":3},
        "validation":{"bash":bash_report,
                      "regex_behavior_comparisons":sum(v["behavior_checks"] for v in regex_checks.values()),
                      "regex_verification":"finite positive/negative examples vs independent predicates; not a formal proof"},
        "leakage_checks":verify_splits(splits),"grouping":split_report,
        "source_reserve":reserve,
        "source_candidates_unchanged":sha256_file(candidate_file)==before_hash,
        "tokenizer_trained":False,"model_trained":False,
        "latency_target_measured":False,
        "limitations":[
            "Bounded templated data: counts do not mean independently authored tasks.",
            "Train/test share task families and some vocabulary; evaluate held-out cases, not arbitrary tools.",
            "Syntactically or behaviorally valid commands do not guarantee model predictions.",
            "Raw NL2Bash, Mango and unadapted regex remain reserve, not default training.",
            "Bash behavior was checked in Git Bash/GNU tools on Windows, not yet a Linux container.",
            "Regex assertions cover finite ASCII single-line fixtures only.",
        ],
        "case_file_sha256":sha256_file(out/"cases.jsonl"),
        "generator_files":{name:sha256_file(root/name) for name in [
            "terminal_copilot/bash_tasks.py", "terminal_copilot/regex_tasks.py",
            "terminal_copilot/bash_validation.py", "terminal_copilot/dataset_v1.py"]},
    }
    (out/"manifest.json").write_text(json.dumps(report,ensure_ascii=False,indent=2)+"\n",encoding="utf-8")
    (root/"artifacts/dataset_v1_check.json").write_text(json.dumps(report,ensure_ascii=False,indent=2)+"\n",encoding="utf-8")
    print(json.dumps({key:report[key] for key in ["total_rows","parameterized_cases","tasks","adapted_KB13_cases"]},indent=2),flush=True)
    for split,info in stats.items(): print(split,info["rows"],info["task_language_pairs"],flush=True)
    return report
