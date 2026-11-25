"""Execute only authored task renderers in a disposable fixture; never raw corpus commands."""
from pathlib import Path
import hashlib
import json
import os
import shutil
import subprocess
import tempfile
import uuid
from . import bash_tasks

def find_bash():
    configured=os.environ.get("TERMINAL_COPILOT_TEST_BASH")
    candidates=[configured,r"C:\Program Files\Git\bin\bash.exe"]
    if os.name!="nt": candidates.append(shutil.which("bash"))
    for candidate in candidates:
        if candidate and Path(candidate).is_file():
            return str(Path(candidate).resolve())
    raise RuntimeError("Bash for fixture tests unavailable. Set TERMINAL_COPILOT_TEST_BASH to a real Bash executable.")

def fixture_hashes(root):
    return {p.relative_to(root).as_posix():hashlib.sha256(p.read_bytes()).hexdigest()
            for p in Path(root).rglob("*") if p.is_file()}

def validate_cases(cases,project_root):
    # Do not accept command strings from callers. Re-render controlled definitions.
    for case in cases:
        if case["family"] not in bash_tasks.KINDS:
            raise ValueError("Unsupported authored task family")
        if case["parameters"] not in bash_tasks.arguments(case["family"]):
            raise ValueError("Parameters outside controlled fixture domain")
        if case["output"]!=bash_tasks.render(case["family"],case["parameters"]):
            raise ValueError("Case command differs from the controlled renderer")
    bash=find_bash()
    project_root=Path(project_root).resolve()
    temporary_parent=project_root/"artifacts"/"fixture_runs"
    temporary_parent.mkdir(parents=True,exist_ok=True)
    assert temporary_parent.resolve().is_relative_to(project_root)
    marker="TC_"+uuid.uuid4().hex
    env=dict(os.environ,BASH_ENV="/dev/null",ENV="/dev/null",LC_ALL="C",LANG="C",
             MSYS_NO_PATHCONV="1",TERM="dumb")
    with tempfile.TemporaryDirectory(prefix="bash_",dir=temporary_parent) as temporary:
        fixture=Path(temporary).resolve()
        assert fixture.is_relative_to(temporary_parent.resolve())
        bash_tasks.make_fixture(fixture)
        before=fixture_hashes(fixture)
        commands="\n".join(case["output"] for case in cases)+"\n"
        syntax=subprocess.run([bash,"--noprofile","--norc","-n"],input=commands,
                              text=True,encoding="utf-8",capture_output=True,env=env,cwd=fixture,timeout=30)
        if syntax.returncode:
            raise AssertionError(("Bash syntax failed",syntax.stderr))
        script=[]
        for case in cases:
            cid=case["case_id"]
            script.append(f"printf '{marker}|START|{cid}\\n'")
            script.append(case["output"])
            script.append(f"printf '{marker}|END|{cid}|%s\\n' \"$?\"")
        execution=subprocess.run(
            [bash,"--noprofile","--norc","-s"],input="\n".join(script)+"\n",
            text=True,encoding="utf-8",capture_output=True,env=env,cwd=fixture,timeout=180,
        )
        if execution.returncode or execution.stderr:
            raise AssertionError(("Fixture process error",execution.returncode,execution.stderr[:1500]))
        results={}; current=None; lines=[]
        for line in execution.stdout.splitlines():
            if line.startswith(marker+"|START|"):
                current=line.split("|",2)[2]; lines=[]
            elif line.startswith(marker+"|END|"):
                _,_,cid,status=line.split("|")
                if cid!=current: raise AssertionError("Mismatched fixture output marker")
                results[cid]=("\n".join(lines),int(status))
                current=None
            elif current is not None:
                lines.append(line)
        if len(results)!=len(cases):
            raise AssertionError(("Missing fixture results",len(results),len(cases)))
        checks={}
        for case in cases:
            expected,status,unordered=bash_tasks.expected(case["family"],case["parameters"],fixture)
            actual,actual_status=results[case["case_id"]]
            if case["family"].startswith("count_") or case["family"]=="grep_fixed_count":
                actual=actual.strip()
            compare=lambda text: sorted(text.splitlines()) if unordered else text
            if actual_status!=status or compare(actual)!=compare(expected):
                raise AssertionError((case["case_id"],case["family"],case["parameters"],
                                      status,actual_status,expected[:400],actual[:400]))
            checks[case["case_id"]]={
                "verification":"authored_command_in_disposable_fixture_vs_python_oracle",
                "bash_syntax_checked":True,"expected_exit_code":status,
                "output_line_order": "unordered" if unordered else "ordered",
            }
        if before!=fixture_hashes(fixture):
            raise AssertionError("A fixture file was unexpectedly modified")
    return checks,{"bash_executable":bash,"commands_checked":len(cases),
                   "raw_source_commands_executed":False,"fixture_files_unchanged":True,
                   "environment":{"LC_ALL":"C","target":"Bash/GNU tools"}}
