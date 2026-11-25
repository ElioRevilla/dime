"""Python-re task definitions, independent predicates, and adapted KB13 cases."""
from itertools import product
import hashlib
import json
import random
import re
import string

WORDS = ["alpha","beta","log","datos","foo","bar","rojo","azul","ERROR","ok",
         "a.b","a+b","[x]","(ok)","x|y","foo-bar","a b","a_b","txt","csv",
         "todo","warn","42","id","p0","nota"]
N = list(range(1,17))
KINDS = [
    "digits_exact","digits_range","lower_exact","upper_exact","alnum_exact",
    "hex_exact","literal_exact","starts_literal","ends_literal","contains_literal",
    "either_literal","prefix_digits","digits_suffix","separated_digits",
    "filename_extension","identifier","signed_integer","letters_spaces",
    "without_character","whole_word","decimal_places","optional_prefix",
]

def render(kind,p):
    e=re.escape
    n=p.get("n",1)
    simple={
        "digits_exact":rf"[0-9]{{{n}}}", "lower_exact":rf"[a-z]{{{n}}}",
        "upper_exact":rf"[A-Z]{{{n}}}", "alnum_exact":rf"[A-Za-z0-9]{{{n}}}",
        "hex_exact":rf"[0-9A-Fa-f]{{{n}}}",
        "identifier":r"[A-Za-z_][A-Za-z0-9_]*",
        "signed_integer":r"-?[0-9]+", "letters_spaces":r"[A-Za-z ]+",
        "decimal_places":rf"-?[0-9]+\.[0-9]{{{n}}}",
    }
    if kind in simple: return simple[kind]
    if kind=="digits_range": return rf"[0-9]{{{p['lo']},{p['hi']}}}"
    if kind=="literal_exact": return e(p["text"])
    if kind=="starts_literal": return e(p["text"])+".*"
    if kind=="ends_literal": return ".*"+e(p["text"])
    if kind=="contains_literal": return ".*"+e(p["text"])+".*"
    if kind=="either_literal": return "(?:"+e(p["a"])+"|"+e(p["b"])+")"
    if kind=="prefix_digits": return e(p["text"])+rf"[0-9]{{{n}}}"
    if kind=="digits_suffix": return rf"[0-9]{{{n}}}"+e(p["text"])
    if kind=="separated_digits": return rf"[0-9]{{{p['left']}}}"+e(p["separator"])+rf"[0-9]{{{p['right']}}}"
    if kind=="filename_extension": return r"[A-Za-z0-9_-]+\."+e(p["extension"])
    if kind=="without_character": return "[^"+e(p["character"])+"]*"
    if kind=="whole_word": return r".*\b"+e(p["word"])+r"\b.*"
    if kind=="optional_prefix": return "(?:"+e(p["prefix"])+")?"+e(p["suffix"])
    raise ValueError(kind)

def arguments(kind):
    if kind in ("digits_exact","lower_exact","upper_exact","alnum_exact","hex_exact","decimal_places"):
        return [dict(n=n) for n in N]
    if kind=="digits_range": return [dict(lo=a,hi=b) for a,b in product(N,N) if a<b]
    if kind in ("literal_exact","starts_literal","ends_literal","contains_literal"):
        return [dict(text=s) for s in WORDS]
    if kind=="either_literal": return [dict(a=a,b=b) for a,b in product(WORDS,WORDS) if a<b]
    if kind in ("prefix_digits","digits_suffix"): return [dict(text=s,n=n) for s,n in product(WORDS,N)]
    if kind=="separated_digits":
        return [dict(left=a,right=b,separator=s) for a,b,s in product(range(1,7),range(1,7),["-","_",":",".","/"])]
    if kind=="filename_extension": return [dict(extension=e) for e in ["txt","log","csv","py","json","md","yaml","xml"]]
    if kind=="without_character": return [dict(character=c) for c in list("aeiouxyz012_-.[]")]
    if kind=="whole_word": return [dict(word=w) for w in WORDS if w.isascii() and w.isalpha()]
    if kind=="optional_prefix": return [dict(prefix=a,suffix=b) for a,b in product(WORDS,WORDS) if a!=b]
    return [{}]

def ascii_words(text):
    # Independent word-boundary oracle; no regex engine.
    words=[]; current=""
    for c in text:
        if c in string.ascii_letters+string.digits+"_":
            current+=c
        else:
            if current: words.append(current)
            current=""
    if current: words.append(current)
    return words

def accepts(kind,p,s):
    digits=lambda text: bool(text) and all(c in string.digits for c in text)
    n=p.get("n",1)
    if kind=="digits_exact": return len(s)==n and digits(s)
    if kind=="digits_range": return p["lo"]<=len(s)<=p["hi"] and digits(s)
    if kind=="lower_exact": return len(s)==n and all(c in string.ascii_lowercase for c in s)
    if kind=="upper_exact": return len(s)==n and all(c in string.ascii_uppercase for c in s)
    if kind=="alnum_exact": return len(s)==n and all(c in string.ascii_letters+string.digits for c in s)
    if kind=="hex_exact": return len(s)==n and all(c in string.hexdigits for c in s)
    if kind=="literal_exact": return s==p["text"]
    if kind=="starts_literal": return s.startswith(p["text"])
    if kind=="ends_literal": return s.endswith(p["text"])
    if kind=="contains_literal": return p["text"] in s
    if kind=="either_literal": return s in (p["a"],p["b"])
    if kind=="prefix_digits": return s.startswith(p["text"]) and len(s)==len(p["text"])+n and digits(s[len(p["text"]):])
    if kind=="digits_suffix": return s.endswith(p["text"]) and len(s)==len(p["text"])+n and digits(s[:n])
    if kind=="separated_digits":
        return (len(s)==p["left"]+1+p["right"] and s[p["left"]]==p["separator"]
                and digits(s[:p["left"]]) and digits(s[p["left"]+1:]))
    if kind=="filename_extension":
        tail="."+p["extension"]
        return s.endswith(tail) and bool(s[:-len(tail)]) and all(
            c in string.ascii_letters+string.digits+"-_" for c in s[:-len(tail)])
    if kind=="identifier": return bool(s) and s[0] in string.ascii_letters+"_" and all(c in string.ascii_letters+string.digits+"_" for c in s[1:])
    if kind=="signed_integer": return digits(s[1:] if s.startswith("-") else s)
    if kind=="letters_spaces": return bool(s) and all(c in string.ascii_letters+" " for c in s)
    if kind=="without_character": return p["character"] not in s
    if kind=="whole_word": return p["word"] in ascii_words(s)
    if kind=="decimal_places":
        text=s[1:] if s.startswith("-") else s
        pieces=text.split(".")
        return len(pieces)==2 and digits(pieces[0]) and len(pieces[1])==n and digits(pieces[1])
    if kind=="optional_prefix": return s in (p["suffix"],p["prefix"]+p["suffix"])
    if kind=="kb13_no_e": return "e" not in s
    if kind=="kb13_lower": return all(c in string.ascii_lowercase for c in s)
    if kind=="kb13_word_dance": return "dance" in ascii_words(s)
    if kind=="kb13_contains_bar": return "bar" in s
    if kind=="kb13_three_upper": return sum(c in string.ascii_uppercase for c in s)>=3
    if kind=="kb13_word_z": return any(w.isalpha() and w.startswith("z") for w in ascii_words(s))
    if kind=="kb13_five_digits": return sum(c in string.digits for c in s)>=5
    if kind=="kb13_ends_f": return s.endswith("f")
    if kind=="kb13_word_er": return any(w.isalpha() and w.endswith("er") for w in ascii_words(s))
    if kind=="kb13_word_spoon": return any(w.isalpha() and w.endswith("spoon") for w in ascii_words(s))
    if kind=="kb13_word_der": return any(w.isalpha() and "der" in w for w in ascii_words(s))
    raise ValueError(kind)

def cases(limit=64):
    result=[]
    for kind in KINDS:
        options=arguments(kind)
        options.sort(key=lambda p:hashlib.sha256(json.dumps(p,sort_keys=True).encode()).hexdigest())
        for p in options[:limit]:
            key=json.dumps({"task":"regex","family":kind,"parameters":p},sort_keys=True,ensure_ascii=False)
            result.append({
                "case_id":"regex-"+hashlib.sha256(key.encode()).hexdigest()[:20],
                "task":"regex","family":kind,"parameters":p,"output":render(kind,p),
            })
    return result

KB13 = {
  1:("kb13_no_e",r"[^e]*","no contengan la letra minúscula e","do not contain lowercase e"),
  3:("kb13_word_er",r".*\b[A-Za-z]*er\b.*","contengan una palabra formada solo por letras ASCII que termine en er","contain an ASCII-letter word ending in er"),
  4:("kb13_lower",r"[a-z]*","contengan solo letras minúsculas ASCII, permitiendo la cadena vacía","contain only lowercase ASCII letters, allowing the empty string"),
  5:("kb13_word_dance",r".*\bdance\b.*","contengan la palabra completa dance","contain the whole word dance"),
  7:("kb13_contains_bar",r".*bar.*","contengan literalmente bar","contain literal bar"),
  8:("kb13_word_spoon",r".*\b[A-Za-z]*spoon\b.*","contengan una palabra formada solo por letras ASCII que termine en spoon","contain an ASCII-letter word ending in spoon"),
  9:("kb13_three_upper",r"(.*[A-Z].*){3}","contengan al menos tres letras mayúsculas ASCII","contain at least three uppercase ASCII letters"),
  10:("kb13_word_z",r".*\bz[A-Za-z]*\b.*","contengan una palabra formada solo por letras ASCII que empiece por z","contain an ASCII-letter word starting with z"),
  11:("kb13_five_digits",r"(.*[0-9].*){5,}","contengan al menos cinco dígitos ASCII","contain at least five ASCII digits"),
  12:("kb13_ends_f",r".*f","terminen en la letra minúscula f","end in lowercase f"),
  13:("kb13_word_der",r".*\b[A-Za-z]*der[A-Za-z]*\b.*","contengan una palabra formada solo por letras ASCII que incluya der","contain an ASCII-letter word containing der"),
}

def adapted_cases():
    return [dict(case_id=f"kb13-adapted-{line}",task="regex",family=kind,parameters={},
                 output=pattern,source_line=line,adapted_description_es=es,adapted_description_en=en)
            for line,(kind,pattern,es,en) in KB13.items()]

def descriptions(kind,p):
    q=lambda text:json.dumps(text,ensure_ascii=False)
    n=p.get("n",1); t=q(p.get("text",""))
    texts={
      "digits_exact":(f"exactamente {n} dígitos ASCII",f"exactly {n} ASCII digits"),
      "digits_range":(f"entre {p.get('lo',1)} y {p.get('hi',2)} dígitos ASCII, incluidos ambos límites",f"between {p.get('lo',1)} and {p.get('hi',2)} ASCII digits, inclusive"),
      "lower_exact":(f"exactamente {n} letras minúsculas ASCII",f"exactly {n} lowercase ASCII letters"),
      "upper_exact":(f"exactamente {n} letras mayúsculas ASCII",f"exactly {n} uppercase ASCII letters"),
      "alnum_exact":(f"exactamente {n} letras o dígitos ASCII",f"exactly {n} ASCII letters or digits"),
      "hex_exact":(f"exactamente {n} dígitos hexadecimales, aceptando letras a-f y A-F",f"exactly {n} hexadecimal digits, accepting a-f and A-F"),
      "literal_exact":(f"exactamente el texto literal {t}",f"exactly literal text {t}"),
      "starts_literal":(f"cualquier cadena que empiece literalmente por {t}",f"any string starting with literal {t}"),
      "ends_literal":(f"cualquier cadena que termine literalmente en {t}",f"any string ending with literal {t}"),
      "contains_literal":(f"cualquier cadena que contenga literalmente {t}",f"any string containing literal {t}"),
      "either_literal":(f"exactamente {q(p.get('a',''))} o exactamente {q(p.get('b',''))}, como textos literales",f"exactly {q(p.get('a',''))} or exactly {q(p.get('b',''))}, both literal"),
      "prefix_digits":(f"el prefijo literal {t} seguido de exactamente {n} dígitos ASCII",f"literal prefix {t} followed by exactly {n} ASCII digits"),
      "digits_suffix":(f"exactamente {n} dígitos ASCII seguidos del sufijo literal {t}",f"exactly {n} ASCII digits followed by literal suffix {t}"),
      "separated_digits":(f"exactamente {p.get('left',1)} dígitos ASCII, el separador literal {q(p.get('separator',''))} y exactamente {p.get('right',1)} dígitos ASCII",f"exactly {p.get('left',1)} ASCII digits, literal separator {q(p.get('separator',''))}, and exactly {p.get('right',1)} ASCII digits"),
      "filename_extension":(f"un nombre no vacío formado por letras ASCII, dígitos, guion o guion bajo, seguido de la extensión literal {q('.'+p.get('extension',''))}",f"a nonempty name of ASCII letters, digits, hyphens or underscores followed by literal extension {q('.'+p.get('extension',''))}"),
      "identifier":("un identificador no vacío que empiece por una letra ASCII o guion bajo y continúe solo con letras ASCII, dígitos o guiones bajos","a nonempty identifier starting with an ASCII letter or underscore and followed only by ASCII letters, digits or underscores"),
      "signed_integer":("un entero con uno o más dígitos ASCII y un signo menos opcional, sin signo más","an integer of one or more ASCII digits with an optional minus sign and no plus sign"),
      "letters_spaces":("una cadena no vacía formada solo por letras ASCII y espacios","a nonempty string containing only ASCII letters and spaces"),
      "without_character":(f"cualquier cadena que no contenga el carácter literal {q(p.get('character',''))}, permitiendo la cadena vacía",f"any string without literal character {q(p.get('character',''))}, allowing the empty string"),
      "whole_word":(f"cualquier cadena que contenga la palabra completa {q(p.get('word',''))} con límites de palabra ASCII",f"any string containing whole word {q(p.get('word',''))} using ASCII word boundaries"),
      "decimal_places":(f"un número con signo menos opcional, uno o más dígitos ASCII, punto decimal y exactamente {n} dígitos decimales","a number with an optional minus sign, one or more ASCII digits, a decimal point, and exactly "+str(n)+" fractional digits"),
      "optional_prefix":(f"el prefijo literal opcional {q(p.get('prefix',''))} seguido del sufijo literal obligatorio {q(p.get('suffix',''))}",f"optional literal prefix {q(p.get('prefix',''))} followed by required literal suffix {q(p.get('suffix',''))}"),
    }
    return texts[kind]

def prompts(case,language):
    if case["family"].startswith("kb13_"):
        content=case["adapted_description_es"] if language=="es" else case["adapted_description_en"]
        if language=="es": content="cadenas que "+content
        else: content="strings that "+content
    else:
        pair=descriptions(case["family"],case["parameters"])
        content=pair[0 if language=="es" else 1]
    if language=="es":
        return [
            f"Regex de Python para coincidencia completa de {content}. Usa ASCII y cadenas de una línea.",
            f"Crea una expresión regular Python que acepte {content} y nada más. Modo fullmatch, ASCII, una línea.",
            f"Con Python re.fullmatch y ASCII, reconoce {content}. La entrada es de una sola línea.",
        ]
    return [
        f"Python regex for a full match of {content}. Use ASCII and single-line strings.",
        f"Create a Python regular expression accepting {content} and nothing else. Fullmatch mode, ASCII, one line.",
        f"Using Python re.fullmatch and ASCII, recognize {content}. Input is a single line.",
    ]

def test_strings(case):
    p=case["parameters"]; kind=case["family"]; n=p.get("n",1)
    pool=set(["","a","e","f","z","0","12345","ABC","AaBbC","bar","foobar","barista",
              "dance","we dance now","dancer","dance_1","zoo","z1","spoon","teaspoon",
              "under","der","der_1","other","fff","A1B2C3456","-12","12.34","a b","a_b","foo.txt"])
    for value in p.values():
        if isinstance(value,str):
            pool.update([value,"x"+value,value+"x","x"+value+"x",value+"_1",value+" "+value])
    pool.update(["7"*n,"a"*n,"A"*n,"f"*n,"Z"*n,"a7"*n,"7"*(n+1),"7"*max(0,n-1)])
    if "lo" in p:
        for length in [p["lo"]-1,p["lo"],p["hi"],p["hi"]+1]: pool.add("7"*length)
    if kind=="prefix_digits": pool.add(p["text"]+"7"*n)
    if kind=="digits_suffix": pool.add("7"*n+p["text"])
    if kind=="separated_digits": pool.add("7"*p["left"]+p["separator"]+"3"*p["right"])
    if kind=="filename_extension": pool.update(["name."+p["extension"],"my_name-2."+p["extension"],"."+p["extension"],"a."+p["extension"]+".x"])
    if kind=="decimal_places": pool.update(["12."+"3"*n,"-12."+"3"*n,"+12."+"3"*n])
    if kind=="optional_prefix": pool.update([p["suffix"],p["prefix"]+p["suffix"]])
    if kind=="without_character":
        alternatives=[c for c in "abcxyz012" if c!=p["character"]]
        pool.add("".join(alternatives))
    if kind=="whole_word": pool.update([p["word"],"a "+p["word"]+" b",p["word"]+"_x"])
    rng=random.Random(int(hashlib.sha256(case["case_id"].encode()).hexdigest()[:16],16))
    alphabet="abefzXYZ019_ .-[]"
    for _ in range(120):
        pool.add("".join(rng.choice(alphabet) for _ in range(rng.randrange(0,25))))
    return sorted(pool)

def validate(case):
    pattern=re.compile(case["output"],flags=re.ASCII)
    positives=[]; negatives=[]
    for text in test_strings(case):
        expected=bool(accepts(case["family"],case["parameters"],text))
        actual=pattern.fullmatch(text) is not None
        if actual!=expected:
            raise AssertionError((case["case_id"],case["output"],text,expected,actual))
        (positives if expected else negatives).append(text)
    if not positives or not negatives:
        raise AssertionError(("Missing positive/negative fixtures",case["case_id"]))
    return {"engine":"python.re","mode":"fullmatch","flags":["ASCII"],
            "input_domain":"ASCII single-line strings","behavior_checks":len(positives)+len(negatives),
            "positive_examples":positives[:8],"negative_examples":negatives[:8],
            "verification":"finite_examples_against_independent_predicate"}
