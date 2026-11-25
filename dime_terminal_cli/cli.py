"""Console entry point for the packaged Spanish model."""
import argparse
import os
from pathlib import Path
import re
import sys


# Protect quoted literals and delimit bare filenames after common Spanish argument markers.
_FILENAME_REFERENCE = re.compile(
    r""" "[^"\n]*" | '[^'\n]*' | \x60[^\x60\n]*\x60
        | (?P<cue>\b(?:de|archivo|fichero)\s+)
          (?P<name>[\w-]+(?:\.[\w-]+)+)(?=$|[\s.,;!?])
    """,
    re.IGNORECASE | re.VERBOSE,
)


def _prepare_query(query, task):
    if task != "bash":
        return query

    def delimit(match):
        if match["name"] is None:
            return match[0]
        return f'{match["cue"]}"{match["name"]}"'

    return _FILENAME_REFERENCE.sub(delimit, query)


def _suggest(copilot, query, task):
    result = copilot.suggest(_prepare_query(query, task), task, "es")
    if not result["ended_with_eos"] or not result["text"].strip():
        print("La sugerencia quedó incompleta. Especifica la tarea y el archivo o la carpeta.", file=sys.stderr)
        return False
    print(result["text"])
    return True


def main(argv=None):
    parser = argparse.ArgumentParser(
        prog="dime",
        description="Sugiere comandos Bash de consulta y expresiones regulares en español.",
        epilog="Ejemplo: dime \"Cuenta las líneas de notas.txt\". Omite la consulta para abrir una sesión.",
    )
    parser.add_argument("consulta", nargs="?", help="petición en español")
    parser.add_argument("--tarea", choices=("bash", "regex"), default="bash")
    parser.add_argument("--dispositivo", choices=("auto", "cpu", "cuda"), default="auto")
    args = parser.parse_intermixed_args(argv)
    if args.consulta is not None and not args.consulta.strip():
        parser.error("Escribe una consulta.")

    from .inference import Copilot
    import torch

    if args.dispositivo == "cuda" and not torch.cuda.is_available():
        parser.error("CUDA no está disponible. Usa --dispositivo auto para usar CPU.")
    torch.set_num_threads(min(4, os.cpu_count() or 1))
    model_dir = Path(__file__).resolve().parent / "models" / "v6"
    copilot = Copilot.from_checkpoint(
        model_dir / "model_inference.pt",
        model_dir / "tokenizer.json",
        device=args.dispositivo,
    )

    if args.consulta is not None:
        return 0 if _suggest(copilot, args.consulta.strip(), args.tarea) else 1

    copilot.warmup()
    print(f"Dime · {args.tarea}. Escribe :quit para salir.", file=sys.stderr)
    while True:
        try:
            query = input("> ").strip()
        except (EOFError, KeyboardInterrupt):
            return 0
        if query == ":quit":
            return 0
        if query:
            _suggest(copilot, query, args.tarea)
