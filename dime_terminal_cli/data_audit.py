"""Read-only source audit and normalization. Commands are never executed."""
from collections import Counter, defaultdict
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import re
import unicodedata

def sha256_file(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()

def write_jsonl(path, rows):
    with Path(path).open("w", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")

def normalize_instruction(text):
    return re.sub(r"\s+", " ", unicodedata.normalize("NFC", text)).strip()

def group_hint(task, dialect, output):
    # Only a hint for future grouped splits, not a semantic equivalence test.
    canonical = re.sub(r"\{\{.*?\}\}", "<ARG>", output) if task == "bash" else output
    return hashlib.sha256((task + "\0" + dialect + "\0" + canonical).encode("utf-8")).hexdigest()

def record(source, revision, path, line, task, language, instruction, output,
           dialect, license_spdx, issues=(), **extra):
    if not isinstance(instruction, str) or not isinstance(output, str):
        raise ValueError("Instruction and output must be strings")
    instruction = normalize_instruction(instruction)
    if not instruction or not output.strip():
        raise ValueError("Empty instruction or output")
    if "\x00" in instruction or "\x00" in output:
        raise ValueError("NUL character in source text")
    return {
        "id": f"{source}:{path}:{line}",
        "source": source, "source_revision": revision,
        "source_file": path, "source_line": line,
        "task": task, "language": language,
        "instruction": instruction, "output": output,
        "target_dialect": dialect, "license_spdx": license_spdx,
        "review_status": "unreviewed", "blocking_issues": sorted(set(issues)),
        "target_group_hint": group_hint(task, dialect, output),
        **extra,
    }

def paired_lines(source_path, target_path):
    left = Path(source_path).read_text(encoding="utf-8-sig").splitlines()
    right = Path(target_path).read_text(encoding="utf-8-sig").splitlines()
    if len(left) != len(right):
        raise ValueError(f"Unaligned parallel files: {len(left)} != {len(right)}")
    return list(zip(left, right))

def tldr_examples(text):
    description = None
    description_line = None
    for line_no, line in enumerate(text.splitlines(), 1):
        if line.startswith("- "):
            description = normalize_instruction(line[2:]).rstrip(":：")
            description_line = line_no
        elif description:
            match = re.fullmatch(r"\s*" + chr(96) + r"(.+)" + chr(96) + r"\s*", line)
            if match:
                yield description_line, line_no, description, match.group(1)
            elif line.startswith("# ") or line.startswith("> "):
                description = None

def regex_diagnostics(pattern):
    # Original-dialect expressions may compile with DIFFERENT Python semantics.
    result = {"python_re_compile_is_semantic_validation": False}
    try:
        re.compile(pattern)
        result["python_re_compiles"] = True
    except re.error as exc:
        result.update(python_re_compiles=False, python_re_error=str(exc))
    result["legacy_operator_characters_present"] = bool(re.search(r"(?<!\\)[&~]", pattern))
    return result

def verify_raw(root):
    root = Path(root).resolve()
    manifest = json.loads((root / "datasets/download_manifest.json").read_text(encoding="utf-8"))
    entries = manifest["downloads"] + manifest["extracted_pages"]
    for entry in entries:
        path = (root / entry["local_path"]).resolve()
        if not path.is_relative_to(root / "datasets/raw"):
            raise ValueError(f"Raw manifest path escapes project: {path}")
        if not path.is_file() or path.stat().st_size != entry["bytes"]:
            raise ValueError(f"Missing or changed source file: {path}")
        if sha256_file(path) != entry["sha256"]:
            raise ValueError(f"Source hash changed: {path}")
    return len(entries)

def build_candidates(root):
    root = Path(root).resolve()
    verified_files = verify_raw(root)
    sources = json.loads((root / "datasets/sources.json").read_text(encoding="utf-8"))
    raw = root / "datasets/raw"
    output_dir = root / "datasets/normalized/v0"
    output_dir.mkdir(parents=True, exist_ok=True)
    rows, rejected = [], []
    def add_or_reject(arguments):
        try:
            rows.append(record(**arguments))
        except ValueError as exc:
            rejected.append({"source": arguments["source"], "source_file": arguments["path"],
                             "source_line": arguments["line"], "reason": str(exc)})

    rev = sources["nl2bash"]["revision"]
    for line, (instruction, output) in enumerate(paired_lines(
        raw / "nl2bash/data/bash/all.nl", raw / "nl2bash/data/bash/all.cm"), 1):
        issues = []
        platform = "unix_unspecified"
        if re.search(r"\b(BSD|macOS|OSX)\b", instruction):
            platform = "bsd_or_macos"
            issues.append("non_linux_platform_review")
        add_or_reject(dict(
            source="nl2bash", revision=rev, path="data/bash/all.nl + data/bash/all.cm",
            line=line, task="bash", language="en", instruction=instruction, output=output,
            dialect="bash", license_spdx="MIT", issues=issues, platform=platform,
            original_split=None,
            split_note="Raw all.* files; not the paper's prepared train/dev/test splits.",
        ))

    rev = sources["deep_regex"]["revision"]
    for corpus in ["KB13", "NL-RX-Synth", "NL-RX-Turk"]:
        prefix = f"datasets/{corpus}"
        for line, (instruction, output) in enumerate(paired_lines(
            raw / "deep_regex" / prefix / "src.txt",
            raw / "deep_regex" / prefix / "targ.txt"), 1):
            add_or_reject(dict(
                source="deep_regex", revision=rev,
                path=f"{prefix}/src.txt + targ.txt", line=line,
                task="regex", language="en", instruction=instruction, output=output,
                dialect="deep_regex_original", license_spdx="MIT",
                issues=["regex_dialect_requires_conversion", "regex_behavior_examples_missing"],
                original_corpus=corpus, diagnostics=regex_diagnostics(output),
                original_split=None,
            ))

    rev = sources["tldr"]["revision"]
    for lang_dir, language in [("pages", "en"), ("pages.en", "en"), ("pages.es", "es")]:
        for platform in ["common", "linux"]:
            for page in sorted((raw / "tldr" / lang_dir / platform).glob("*.md")):
                relative = page.relative_to(raw / "tldr").as_posix()
                for description_line, line, instruction, output in tldr_examples(page.read_text(encoding="utf-8")):
                    issues = ["tool_platform_and_behavior_review"]
                    if "{{" in output:
                        issues.append("template_parameters_uninstantiated")
                    add_or_reject(dict(
                        source="tldr", revision=rev, path=relative, line=line,
                        task="bash", language=language, instruction=instruction,
                        output=output, dialect="bash", license_spdx="CC-BY-4.0",
                        issues=issues, platform=platform, tool=page.stem,
                        description_line=description_line, original_split=None,
                        attribution_url=f"https://github.com/tldr-pages/tldr/blob/{rev}/{relative}",
                    ))

    rev = sources["mango"]["revision"]
    path = raw / "mango/MangoNLTbash.jsonl"
    for line, text in enumerate(path.read_text(encoding="utf-8-sig").splitlines(), 1):
        if not text.strip():
            continue
        try:
            item = json.loads(text)
        except json.JSONDecodeError as exc:
            rejected.append({"source": "mango", "source_file": path.name,
                             "source_line": line, "reason": f"Invalid JSON: {exc}"})
            continue
        add_or_reject(dict(
            source="mango", revision=rev, path=path.name, line=line,
            task="bash", language="es", instruction=item.get("instruction"), output=item.get("cmd"),
            dialect="bash", license_spdx="Apache-2.0",
            issues=["source_provenance_and_semantics_review"], platform="linux_unspecified",
            original_split=None,
        ))

    ids = [row["id"] for row in rows]
    if len(ids) != len(set(ids)):
        raise ValueError("Duplicate source IDs")
    exact_pairs = Counter((r["task"], r["language"], r["instruction"], r["output"]) for r in rows)
    instruction_outputs = defaultdict(set)
    targets = defaultdict(list)
    for row in rows:
        instruction_outputs[(row["task"], row["language"], row["instruction"])].add(row["output"])
        targets[row["target_group_hint"]].append(row)
    multi_language_groups = sum(
        len({r["language"] for r in members}) > 1 for members in targets.values()
    )
    source_summaries = {}
    for name in sources:
        selected = [row for row in rows if row["source"] == name]
        filename = output_dir / f"{name}.jsonl"
        write_jsonl(filename, selected)
        source_summaries[name] = {
            "candidate_pairs": len(selected),
            "languages": dict(Counter(r["language"] for r in selected)),
            "tasks": dict(Counter(r["task"] for r in selected)),
            "license_spdx": sources[name]["declared_data_license"],
            "revision": sources[name]["revision"],
            "file": filename.relative_to(root).as_posix(),
            "bytes": filename.stat().st_size, "sha256": sha256_file(filename),
        }
    write_jsonl(output_dir / "candidates.jsonl", rows)
    write_jsonl(output_dir / "rejected.jsonl", rejected)
    examples, slots = [], Counter()
    for row in rows:
        slot = (row["source"], row["language"], row["task"])
        if slots[slot] < 3:
            examples.append(row)
            slots[slot] += 1
    write_jsonl(output_dir / "samples.jsonl", examples)
    issue_examples = []
    observed_issues = set()
    for row in rows:
        if set(row["blocking_issues"]) - observed_issues:
            issue_examples.append(row)
            observed_issues.update(row["blocking_issues"])
    write_jsonl(output_dir / "review_examples.jsonl", issue_examples)
    summary = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "stage": "source_audit_and_candidate_normalization",
        "training_ready": False, "training_splits_created": False,
        "commands_executed": False,
        "raw_files_verified": verified_files,
        "candidate_pairs": len(rows),
        "unique_exact_pairs": len(exact_pairs),
        "duplicate_exact_pair_rows": sum(n-1 for n in exact_pairs.values()),
        "instructions_with_multiple_targets": sum(len(v)>1 for v in instruction_outputs.values()),
        "languages": dict(Counter(r["language"] for r in rows)),
        "tasks": dict(Counter(r["task"] for r in rows)),
        "task_language_pairs": dict(Counter(f"{r['task']}:{r['language']}" for r in rows)),
        "blocking_issues": dict(Counter(issue for r in rows for issue in r["blocking_issues"])),
        "rejected_rows": len(rejected),
        "sources": source_summaries,
        "original_regex_corpora": dict(Counter(
            r["original_corpus"] for r in rows if r["task"]=="regex")),
        "multilingual_target_hint_groups": multi_language_groups,
        "group_hint_is_final_split_or_semantic_equivalence": False,
        "regex_python_compilation_is_semantic_validation": False,
        "remaining_work": [
            "Review per-source correctness and final licenses/attributions.",
            "Deduplicate while retaining every source attribution.",
            "Instantiate tldr placeholders and verify instruction/argument alignment.",
            "Select Bash/Linux tool scope and validate syntax without executing on the host.",
            "Convert regex dialect and add positive/negative behavior tests.",
            "Add reviewed Spanish regex requests; current original regex sources are English.",
            "Create grouped train/validation/test splits before augmentation/translation.",
            "Train a new tokenizer only on the training split; then supervised training.",
        ],
        "candidates_file_sha256": sha256_file(output_dir / "candidates.jsonl"),
    }
    for destination in [output_dir / "manifest.json", root / "artifacts/dataset_audit.json"]:
        destination.write_text(json.dumps(summary, ensure_ascii=False, indent=2)+"\n", encoding="utf-8")
    if verify_raw(root) != verified_files:
        raise AssertionError("Raw source count changed during audit")
    return summary
