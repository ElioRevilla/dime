"""Controlled Bash task families and fixture oracles for dataset v1."""
from itertools import groupby, product
from pathlib import Path
import hashlib
import json
import shlex

FILES = [f"datos/archivo_{i:02d}.txt" for i in range(16)] + [
    "notas.txt", "datos/mi archivo.txt", "datos/O'Neil.txt",
    "datos/-inicio.txt", "datos/café.txt", "src/main.py", "datos/report.log",
    "mi carpeta/datos.txt",
]
NUMFILES = [f"numeros/serie_{i:02d}.txt" for i in range(12)]
CSVFILES = [("csv/comas.txt", ","), ("csv/puntos.txt", ";"), ("csv/dos_puntos.txt", ":")]
DIRECTORIES = [".", "./datos", "./src", "./mi carpeta", "./proyecto"]
NEEDLES = ["error", "ERROR", "warning", "INFO", "TODO", "alpha", "a.b", "[x]", "O'Neil", "-WARN", "ausente"]
NUMBERS = [1, 2, 3, 5, 8, 10, 15, 20]
KINDS = [
    "cat", "head", "tail", "count_lines", "count_words", "count_bytes",
    "sort", "sort_reverse", "sort_numeric", "sort_unique", "uniq_adjacent",
    "grep_fixed", "grep_fixed_ignore_case", "grep_fixed_numbered",
    "grep_fixed_invert", "grep_fixed_count", "find_extension",
    "find_directory", "find_files", "cut_field", "list_all",
]
SOURCE_TOOLS = {
    "count_lines":"wc", "count_words":"wc", "count_bytes":"wc",
    "sort_reverse":"sort", "sort_numeric":"sort", "sort_unique":"sort",
    "uniq_adjacent":"uniq", "find_extension":"find", "find_directory":"find",
    "find_files":"find", "cut_field":"cut", "list_all":"ls",
}
for _kind in KINDS:
    if _kind.startswith("grep_"):
        SOURCE_TOOLS[_kind]="grep"

def file_lines(name):
    if name in NUMFILES:
        i=NUMFILES.index(name)
        return [str(n+i) for n in [3, -2, 10, 0, 3, 22, -10]]
    if name.startswith("csv/"):
        delimiter=dict(CSVFILES)[name]
        return [delimiter.join(row) for row in [
            ("ana","10","ok"), ("luis","20","error"), ("eva","30","ok"),
        ]]
    tag=hashlib.sha256(name.encode("utf-8")).hexdigest()[:6]
    return [
        "alpha", "alpha", "ERROR uno", "warning dos", "INFO tres", "TODO cuatro",
        "error cinco", "a.b literal", "[x] marca", "O'Neil", "-WARN aviso",
        "", f"archivo {tag}", "alpha",
    ]

def make_fixture(root):
    root=Path(root)
    for name in FILES+NUMFILES+[name for name,_ in CSVFILES]:
        path=root/name
        path.parent.mkdir(parents=True,exist_ok=True)
        path.write_bytes(("\n".join(file_lines(name))+"\n").encode("utf-8"))
    for directory in ["proyecto/src","proyecto/docs","proyecto/a b","proyecto/vacio",
                      "datos/src","datos/docs","mi carpeta/src"]:
        (root/directory).mkdir(parents=True,exist_ok=True)
    for name in ["proyecto/src/a.py","proyecto/src/b.py","proyecto/docs/guia.txt",
                 "proyecto/readme.md","datos/info.log","datos/.oculto.txt"]:
        (root/name).write_bytes(b"fixture\n")

def render(kind,p):
    q=shlex.quote
    if kind=="cat": return f"cat -- {q(p['file'])}"
    if kind in ("head","tail"): return f"{kind} -n {p['n']} -- {q(p['file'])}"
    flags={"count_lines":"-l","count_words":"-w","count_bytes":"-c"}
    if kind in flags: return f"wc {flags[kind]} < {q(p['file'])}"
    flags={"sort":"","sort_reverse":"-r ","sort_numeric":"-n ","sort_unique":"-u "}
    if kind in flags: return f"sort {flags[kind]}-- {q(p['file'])}"
    if kind=="uniq_adjacent": return f"uniq -- {q(p['file'])}"
    if kind=="grep_fixed_ignore_case":
        # Basic-regex escaping preserves literal matching and works with Git's GNU grep 3.0.
        pattern="".join("\\"+c if c in "\\.^$*[]" else c for c in p["needle"])
        return f"grep -i -- {q(pattern)} {q(p['file'])}"
    flags={"grep_fixed":"-F","grep_fixed_ignore_case":"-Fi","grep_fixed_numbered":"-Fn",
           "grep_fixed_invert":"-Fv","grep_fixed_count":"-Fc"}
    if kind in flags: return f"grep {flags[kind]} -- {q(p['needle'])} {q(p['file'])}"
    if kind=="find_extension":
        return f"find {q(p['directory'])} -type f -name {q('*.'+p['extension'])}"
    if kind=="find_directory":
        return f"find {q(p['directory'])} -type d -name {q(p['name'])}"
    if kind=="find_files": return f"find {q(p['directory'])} -type f"
    if kind=="cut_field":
        return f"cut -d {q(p['delimiter'])} -f {p['field']} -- {q(p['file'])}"
    if kind=="list_all": return f"ls -1A -- {q(p['directory'])}"
    raise ValueError(kind)

def arguments(kind):
    if kind in ("head","tail"):
        return [dict(file=f,n=n) for f,n in product(FILES,NUMBERS)]
    if kind=="sort_numeric": return [dict(file=f) for f in NUMFILES]
    if kind.startswith("grep_"):
        return [dict(file=f,needle=s) for f,s in product(FILES,NEEDLES)]
    if kind=="find_extension":
        return [dict(directory=d,extension=e) for d,e in product(DIRECTORIES,["txt","py","log","md","csv"])]
    if kind=="find_directory":
        return [dict(directory=d,name=n) for d,n in product(DIRECTORIES,["src","docs","vacio","a b"])]
    if kind in ("find_files","list_all"): return [dict(directory=d) for d in DIRECTORIES]
    if kind=="cut_field":
        return [dict(file=f,delimiter=d,field=n) for (f,d),n in product(CSVFILES,[1,2,3])]
    return [dict(file=f) for f in FILES]

def cases(limit=64):
    result=[]
    for kind in KINDS:
        options=arguments(kind)
        options.sort(key=lambda p:hashlib.sha256(json.dumps(p,sort_keys=True).encode()).hexdigest())
        for p in options[:limit]:
            key=json.dumps({"task":"bash","family":kind,"parameters":p},sort_keys=True,ensure_ascii=False)
            result.append({
                "case_id":"bash-"+hashlib.sha256(key.encode()).hexdigest()[:20],
                "task":"bash","family":kind,"parameters":p,"output":render(kind,p),
                "source_tool":SOURCE_TOOLS.get(kind,kind),
            })
    return result

def prompts(kind,p,language):
    # Authored paraphrases: every value required by the command is in the request.
    q=lambda s:json.dumps(s,ensure_ascii=False)
    f=q(p.get("file","")); d=q(p.get("directory","")); n=p.get("n",0)
    s=q(p.get("needle",""))
    en={
      "cat":[f"Print the entire contents of file {f}.",f"Show me all the text in {f}.",f"Read file {f} and display its contents."],
      "head":[f"Print the first {n} lines of file {f}.",f"Show the top {n} lines from {f}.",f"Read only the first {n} lines in {f}."],
      "tail":[f"Print the last {n} lines of file {f}.",f"Show the bottom {n} lines from {f}.",f"Read only the final {n} lines in {f}."],
      "count_lines":[f"Count the lines in {f}, output only the number.",f"How many lines are in {f}? Return just the count.",f"Print only the line count for file {f}."],
      "count_words":[f"Count the whitespace-separated words in {f}, output only the number.",f"How many words are in {f}? Return just the count.",f"Print only the word count for file {f}."],
      "count_bytes":[f"Count the bytes in {f}, output only the number.",f"How many bytes does {f} contain? Return just the count.",f"Print only the byte count for file {f}."],
      "sort":[f"Sort the lines of {f} in ascending order using the C locale.",f"Display {f} sorted ascending with C collation.",f"Order all lines in {f} ascending in the C locale."],
      "sort_reverse":[f"Sort the lines of {f} in descending order using the C locale.",f"Display {f} sorted in reverse C order.",f"Order all lines in {f} descending in the C locale."],
      "sort_numeric":[f"Sort the integer lines in {f} in ascending numeric order.",f"Order the numbers from {f} smallest to largest.",f"Show {f} sorted numerically from low to high."],
      "sort_unique":[f"Sort {f} in C order and remove duplicate lines.",f"Show unique lines from {f}, sorted with C collation.",f"Deduplicate and sort all lines of {f} in the C locale."],
      "uniq_adjacent":[f"Remove adjacent duplicate lines from {f} without sorting.",f"Collapse consecutive repeated lines in {f}, keeping their order.",f"Show {f} with each run of identical lines printed once."],
      "grep_fixed":[f"Find lines containing literal text {s} in {f}, case-sensitive.",f"Show lines in {f} containing {s} literally, matching case.",f"Search {f} for literal {s} with case-sensitive matching."],
      "grep_fixed_ignore_case":[f"Find lines containing literal text {s} in {f}, ignoring case.",f"Show lines in {f} containing {s} literally, without case sensitivity.",f"Search {f} for literal {s} regardless of letter case."],
      "grep_fixed_numbered":[f"Find literal {s} in {f}, matching case, and print line numbers.",f"Show matching lines and their numbers for literal {s} in {f}, case-sensitive.",f"Search for exact text {s} in {f} with line numbers and case-sensitive matching."],
      "grep_fixed_invert":[f"Show lines of {f} that do not contain literal {s}, case-sensitive.",f"Exclude lines containing literal {s} from {f}, matching case.",f"Print nonmatching lines for literal {s} in {f}, with case sensitivity."],
      "grep_fixed_count":[f"Count lines containing literal {s} in {f}, case-sensitive.",f"Print the number of lines matching literal {s} in {f}, matching case.",f"How many lines contain exact text {s} in {f}? Match letter case."],
      "find_extension":[f"Recursively find regular files with extension {q('.'+p.get('extension',''))} under {d}.",f"Search all subdirectories of {d} for regular files ending in {q('.'+p.get('extension',''))}.",f"List regular files matching {q('*.'+p.get('extension',''))} recursively inside {d}."],
      "find_directory":[f"Recursively find directories named {q(p.get('name',''))} under {d}.",f"Search inside {d} for directories whose exact name is {q(p.get('name',''))}.",f"List directories called {q(p.get('name',''))} below {d}, including all subdirectories."],
      "find_files":[f"Recursively list every regular file under {d}.",f"Find all regular files inside {d} and its subdirectories.",f"Show the paths of regular files recursively below {d}."],
      "cut_field":[f"Print field {p.get('field',1)} of each line in {f}, using {q(p.get('delimiter',''))} as the delimiter.",f"Extract column {p.get('field',1)} from {f}; fields are separated by {q(p.get('delimiter',''))}.",f"Split each line of {f} on {q(p.get('delimiter',''))} and output field {p.get('field',1)}."],
      "list_all":[f"List names in directory {d}, including hidden entries, excluding dot and dot-dot, one per line.",f"Show all entry names in {d}, one per line, including hidden names but excluding the special dot and dot-dot entries.",f"List the contents of {d} one name per line, with hidden entries and excluding the special dot and dot-dot entries."],
    }
    es={
      "cat":[f"Imprime todo el contenido del archivo {f}.",f"Muestra todo el texto de {f}.",f"Lee el archivo {f} y muestra su contenido."],
      "head":[f"Imprime las primeras {n} líneas del archivo {f}.",f"Muestra las {n} líneas iniciales de {f}.",f"Lee solamente las primeras {n} líneas de {f}."],
      "tail":[f"Imprime las últimas {n} líneas del archivo {f}.",f"Muestra las {n} líneas finales de {f}.",f"Lee solamente las últimas {n} líneas de {f}."],
      "count_lines":[f"Cuenta las líneas de {f} y muestra solo el número.",f"¿Cuántas líneas tiene {f}? Devuelve únicamente la cantidad.",f"Imprime solo el número de líneas del archivo {f}."],
      "count_words":[f"Cuenta las palabras separadas por espacios en {f} y muestra solo el número.",f"¿Cuántas palabras tiene {f}? Devuelve únicamente la cantidad.",f"Imprime solo el número de palabras del archivo {f}."],
      "count_bytes":[f"Cuenta los bytes de {f} y muestra solo el número.",f"¿Cuántos bytes contiene {f}? Devuelve únicamente la cantidad.",f"Imprime solo el número de bytes del archivo {f}."],
      "sort":[f"Ordena las líneas de {f} de forma ascendente usando la configuración regional C.",f"Muestra {f} ordenado de menor a mayor con el orden de C.",f"Ordena todas las líneas de {f} ascendentemente con la configuración regional C."],
      "sort_reverse":[f"Ordena las líneas de {f} de forma descendente usando la configuración regional C.",f"Muestra {f} ordenado en orden inverso de C.",f"Ordena todas las líneas de {f} descendentemente con la configuración regional C."],
      "sort_numeric":[f"Ordena las líneas enteras de {f} de menor a mayor numéricamente.",f"Ordena los números de {f} del más pequeño al más grande.",f"Muestra {f} ordenado numéricamente de menor a mayor."],
      "sort_unique":[f"Ordena {f} con el orden de C y elimina las líneas duplicadas.",f"Muestra las líneas únicas de {f}, ordenadas según C.",f"Elimina duplicados y ordena las líneas de {f} con la configuración regional C."],
      "uniq_adjacent":[f"Elimina las líneas duplicadas adyacentes de {f} sin ordenar.",f"Colapsa las repeticiones consecutivas de líneas en {f}, conservando su orden.",f"Muestra {f} imprimiendo una sola vez cada grupo de líneas idénticas consecutivas."],
      "grep_fixed":[f"Busca líneas que contengan literalmente {s} en {f}, distinguiendo mayúsculas.",f"Muestra las líneas de {f} que incluyan el texto literal {s}, respetando las mayúsculas.",f"Busca el texto exacto {s} en {f} distinguiendo mayúsculas y minúsculas."],
      "grep_fixed_ignore_case":[f"Busca líneas que contengan literalmente {s} en {f}, sin distinguir mayúsculas.",f"Muestra las líneas de {f} que incluyan el texto literal {s}, ignorando las mayúsculas.",f"Busca el texto exacto {s} en {f} sin distinguir mayúsculas y minúsculas."],
      "grep_fixed_numbered":[f"Busca literalmente {s} en {f}, distinguiendo mayúsculas, y muestra los números de línea.",f"Muestra las líneas y sus números que contienen el texto literal {s} en {f}, respetando mayúsculas.",f"Busca el texto exacto {s} en {f} con números de línea y distinguiendo mayúsculas."],
      "grep_fixed_invert":[f"Muestra las líneas de {f} que no contienen literalmente {s}, distinguiendo mayúsculas.",f"Excluye de {f} las líneas que contienen el texto literal {s}, respetando mayúsculas.",f"Imprime las líneas sin el texto exacto {s} en {f}, distinguiendo mayúsculas."],
      "grep_fixed_count":[f"Cuenta las líneas que contienen literalmente {s} en {f}, distinguiendo mayúsculas.",f"Imprime cuántas líneas de {f} contienen el texto literal {s}, respetando las mayúsculas.",f"¿Cuántas líneas contienen el texto exacto {s} en {f}? Distingue mayúsculas."],
      "find_extension":[f"Busca recursivamente archivos regulares con extensión {q('.'+p.get('extension',''))} dentro de {d}.",f"Busca en todas las subcarpetas de {d} archivos regulares terminados en {q('.'+p.get('extension',''))}.",f"Lista recursivamente los archivos regulares que coinciden con {q('*.'+p.get('extension',''))} dentro de {d}."],
      "find_directory":[f"Busca recursivamente directorios llamados {q(p.get('name',''))} dentro de {d}.",f"Busca dentro de {d} directorios cuyo nombre exacto sea {q(p.get('name',''))}.",f"Lista los directorios llamados {q(p.get('name',''))} bajo {d}, incluyendo todas las subcarpetas."],
      "find_files":[f"Lista recursivamente todos los archivos regulares dentro de {d}.",f"Busca todos los archivos regulares de {d} y sus subcarpetas.",f"Muestra las rutas de archivos regulares recursivamente bajo {d}."],
      "cut_field":[f"Imprime el campo {p.get('field',1)} de cada línea de {f}, usando {q(p.get('delimiter',''))} como separador.",f"Extrae la columna {p.get('field',1)} de {f}; los campos están separados por {q(p.get('delimiter',''))}.",f"Divide cada línea de {f} por {q(p.get('delimiter',''))} y muestra el campo {p.get('field',1)}."],
      "list_all":[f"Lista los nombres de {d}, incluyendo ocultos pero excluyendo punto y doble punto, uno por línea.",f"Muestra todos los nombres de entradas de {d}, uno por línea, incluyendo ocultos pero excluyendo las entradas especiales punto y doble punto.",f"Lista el contenido de {d} con un nombre por línea, entradas ocultas y excluyendo las entradas especiales punto y doble punto."],
    }
    return (es if language=="es" else en)[kind]

def expected(kind,p,root):
    if "file" in p:
        lines=file_lines(p["file"])
    if kind=="cat": out=lines
    elif kind=="head": out=lines[:p["n"]]
    elif kind=="tail": out=lines[-p["n"]:]
    elif kind=="count_lines": return str(len(lines)),0,False
    elif kind=="count_words": return str(len(("\n".join(lines)+"\n").split())),0,False
    elif kind=="count_bytes": return str(len(("\n".join(lines)+"\n").encode("utf-8"))),0,False
    elif kind=="sort": out=sorted(lines,key=lambda s:s.encode("utf-8"))
    elif kind=="sort_reverse": out=sorted(lines,key=lambda s:s.encode("utf-8"),reverse=True)
    elif kind=="sort_numeric": out=sorted(lines,key=int)
    elif kind=="sort_unique": out=sorted(set(lines),key=lambda s:s.encode("utf-8"))
    elif kind=="uniq_adjacent": out=[key for key,_ in groupby(lines)]
    elif kind.startswith("grep_"):
        matches=[(i,line) for i,line in enumerate(lines,1)
                 if (p["needle"].lower() in line.lower() if kind=="grep_fixed_ignore_case" else p["needle"] in line)]
        if kind=="grep_fixed_invert":
            selected={i for i,_ in matches}
            matches=[(i,line) for i,line in enumerate(lines,1) if i not in selected]
        if kind=="grep_fixed_count": return str(len(matches)),0 if matches else 1,False
        out=[f"{i}:{line}" if kind=="grep_fixed_numbered" else line for i,line in matches]
        return "\n".join(out),0 if matches else 1,False
    elif kind=="cut_field":
        out=[line.split(p["delimiter"])[p["field"]-1] if p["delimiter"] in line else line for line in lines]
    elif kind in ("find_extension","find_directory","find_files"):
        base=Path(root)/p["directory"]
        selected=[]
        candidates=[base]+list(base.rglob("*"))
        for path in candidates:
            if kind=="find_directory":
                accept=path.is_dir() and path.name==p["name"]
            else:
                accept=path.is_file()
                if kind=="find_extension": accept=accept and path.name.endswith("."+p["extension"])
            if accept:
                relative=path.relative_to(base).as_posix()
                value=p["directory"] if relative=="." else p["directory"].rstrip("/")+"/"+relative
                selected.append(value)
        return "\n".join(selected),0,True
    elif kind=="list_all":
        out=sorted([p.name for p in (Path(root)/p["directory"]).iterdir()],key=lambda s:s.encode("utf-8"))
    else: raise ValueError(kind)
    return "\n".join(out),0,False
