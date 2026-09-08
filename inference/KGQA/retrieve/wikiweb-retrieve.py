import os
import json
from typing import Dict, Any, Optional,List, Set, Tuple
from tqdm import tqdm
import re
from concurrent.futures import ProcessPoolExecutor, as_completed

from rdflib import Variable, URIRef, Literal, BNode
from rdflib.paths import (
    SequencePath,
    AlternativePath,
    MulPath,
    InvPath,
    NegatedPath,
)
from rdflib.plugins.sparql import parser as sparql_parser
from rdflib.plugins.sparql.algebra import translateQuery
from rdflib.plugins.sparql.parserutils import CompValue
from query import run_sparql

class VarGenerator:
    """Generate fresh rdflib Variables with a prefix (e.g. _path1, _path2 ...)."""

    def __init__(self, prefix: str = "_path", used_names: Optional[Set[str]] = None):
        self.prefix = prefix
        self.counter = 0
        self.used_names = set(used_names or set())

    def new(self) -> Variable:
        """Return a fresh Variable that doesn't collide with existing ones."""
        while True:
            self.counter += 1
            name = f"{self.prefix}{self.counter}"
            if name not in self.used_names:
                self.used_names.add(name)
                return Variable(name)


# ----------------------------
# Short formatting helpers (for triple extraction output)
# ----------------------------

def _shorten_iri_like(s: str) -> str:
    """Shorten an IRI string to its last token (after / or #)."""
    s = s.strip().replace("<", "").replace(">", "")
    pos = max(s.rfind("/"), s.rfind("#"))
    return s[pos + 1 :] if pos != -1 else s


def _term_to_str_short(term):
    """Convert rdflib terms/paths to short readable strings."""
    if isinstance(term, Variable):
        return f"?{term}"
    if isinstance(term, URIRef):
        return _shorten_iri_like(term.n3())
    if isinstance(term, (SequencePath, AlternativePath, MulPath, InvPath, NegatedPath)):
        return _shorten_iri_like(term.n3())
    if isinstance(term, (Literal, BNode)):
        return term.n3()
    return str(term)


def _term_to_sparql(term) -> str:
    """
    Convert an rdflib term to a SPARQL fragment.
    NOTE: This may produce full IRIs (<...>) depending on rdflib namespace handling,
    but it will remain semantically correct SPARQL.
    """
    if isinstance(term, Variable):
        return f"?{term}"
    if isinstance(term, (URIRef, Literal, BNode)):
        return term.n3()
    if isinstance(term, (SequencePath, AlternativePath, MulPath, InvPath, NegatedPath)):
        return term.n3()
    return str(term)


# ----------------------------
# SERVICE stripping (textual)
# ----------------------------

def _strip_service_blocks(query: str) -> str:
    """
    Replace: SERVICE <something> { inner }
    with:    inner

    This is a textual preprocessing step, robust to nested braces and quoted strings.
    """
    s = query
    out = []
    i = 0
    n = len(s)

    def is_word_boundary(idx: int) -> bool:
        if idx <= 0 or idx >= n:
            return True
        return not (s[idx].isalnum() or s[idx] == "_")

    in_str = None  # '"' or "'"
    while i < n:
        ch = s[i]

        # Handle string literals (basic, sufficient for skipping SERVICE parsing inside strings)
        if in_str:
            out.append(ch)
            if ch == in_str:
                in_str = None
            elif ch == "\\" and i + 1 < n:
                i += 1
                out.append(s[i])
            i += 1
            continue
        else:
            if ch in ("'", '"'):
                in_str = ch
                out.append(ch)
                i += 1
                continue

        # Detect SERVICE keyword (case-insensitive) at word boundary
        if (i + 7 <= n and s[i:i+7].lower() == "service"
                and is_word_boundary(i - 1) and is_word_boundary(i + 7)):
            # Skip "SERVICE"
            j = i + 7

            # Skip whitespace until we find the opening "{"
            while j < n and s[j].isspace():
                j += 1

            # Find the first '{' after SERVICE target (could be IRI or VAR)
            while j < n and s[j] != "{":
                if s[j] in ("'", '"'):
                    quote = s[j]
                    j += 1
                    while j < n and s[j] != quote:
                        if s[j] == "\\" and j + 1 < n:
                            j += 2
                        else:
                            j += 1
                    j += 1
                    continue
                j += 1

            if j >= n or s[j] != "{":
                out.append(ch)
                i += 1
                continue

            # j is '{' of SERVICE block; find matching '}'
            brace = 0
            k = j
            while k < n:
                if s[k] in ("'", '"'):
                    quote = s[k]
                    k += 1
                    while k < n and s[k] != quote:
                        if s[k] == "\\" and k + 1 < n:
                            k += 2
                        else:
                            k += 1
                    k += 1
                    continue

                if s[k] == "{":
                    brace += 1
                elif s[k] == "}":
                    brace -= 1
                    if brace == 0:
                        break
                k += 1

            if k >= n:
                out.append(ch)
                i += 1
                continue

            inner = s[j+1:k]  # content inside { ... }
            out.append(" ")
            out.append(inner)
            out.append(" ")
            i = k + 1
            continue

        out.append(ch)
        i += 1

    return "".join(out)


# ----------------------------
# Query normalization (fragment -> full query)
# ----------------------------

def _ensure_full_query(query: str) -> str:
    """
    If the input is only a fragment starting with SERVICE / WHERE / { ... },
    wrap it into a valid query so rdflib can parse it.
    Preserve PREFIX/BASE declarations.
    """
    q = query.strip()

    # Split PREFIX/BASE header from body
    header = []
    lines = q.splitlines()
    idx = 0
    while idx < len(lines):
        line = lines[idx].strip()
        if line.lower().startswith("prefix ") or line.lower().startswith("base "):
            header.append(lines[idx])
            idx += 1
        else:
            break
    body = "\n".join(lines[idx:]).strip()

    # Already a full query?
    if re.match(r"^(select|ask|construct|describe)\b", body, flags=re.I):
        return "\n".join(header + [body]).strip()

    # Fragment cases
    if re.match(r"^service\b", body, flags=re.I):
        body = f"SELECT * WHERE {{ {body} }}"
    elif re.match(r"^where\b", body, flags=re.I):
        body = f"SELECT * {body}"
    elif body.startswith("{"):
        body = f"SELECT * WHERE {body}"

    return "\n".join(header + [body]).strip()


# ----------------------------
# Algebra traversal & path expansion
# ----------------------------

def _expand_path_triple(subject, predicate, obj, var_gen: VarGenerator):
    """
    Expand SequencePath (p1/p2/...) into a chain of triples with fresh intermediate variables.
    Other path types are kept as-is.
    """
    if isinstance(predicate, SequencePath):
        triples = []
        cur_s = subject
        for i, seg in enumerate(predicate.args):
            cur_o = obj if i == len(predicate.args) - 1 else var_gen.new()
            triples.extend(_expand_path_triple(cur_s, seg, cur_o, var_gen))
            cur_s = cur_o
        return triples
    return [(subject, predicate, obj)]


def _collect_triples_from_algebra(algebra) -> List[Tuple[object, object, object]]:
    """
    Collect triples from rdflib SPARQL algebra, expanding SequencePath into multiple triples.
    """
    triples: List[Tuple[object, object, object]] = []

    # Collect existing variable names to avoid collisions for generated vars
    existing = set(re.findall(r"\?([A-Za-z_]\w*)", str(algebra.algebra)))
    var_gen = VarGenerator("_path", used_names=existing)

    def walk(node):
        if isinstance(node, CompValue):
            if node.name == "BGP":
                for (s, p, o) in getattr(node, "triples", []):
                    triples.extend(_expand_path_triple(s, p, o, var_gen))
            for v in node.values():
                if isinstance(v, (CompValue, list, tuple)):
                    walk(v)
        elif isinstance(node, (list, tuple)):
            for item in node:
                walk(item)

    walk(algebra.algebra)
    return triples


def _collect_path_triples_from_algebra(algebra) -> List[Tuple[object, object, object, List[Tuple[object, object, object]]]]:
    """
    Collect ONLY triples whose predicate is a SequencePath, and provide their expanded replacement triples.
    Returns:
        [(s, path_pred, o, expanded_triples), ...]
    """
    results = []

    existing = set(re.findall(r"\?([A-Za-z_]\w*)", str(algebra.algebra)))
    var_gen = VarGenerator("_path", used_names=existing)

    def walk(node):
        if isinstance(node, CompValue):
            if node.name == "BGP":
                for (s, p, o) in getattr(node, "triples", []):
                    if isinstance(p, SequencePath):
                        expanded = _expand_path_triple(s, p, o, var_gen)
                        # Only consider real expansions (more than 1 triple)
                        if len(expanded) > 1:
                            results.append((s, p, o, expanded))
            for v in node.values():
                if isinstance(v, (CompValue, list, tuple)):
                    walk(v)
        elif isinstance(node, (list, tuple)):
            for item in node:
                walk(item)

    walk(algebra.algebra)
    return results


# ----------------------------
# Minimal query rewriting:
#   - only expand / paths (SequencePath)
#   - only change SELECT projection to "*"
#   - keep OPTIONAL/FILTER/UNION/etc text unchanged
# ----------------------------

def _rewrite_select_to_star(query: str) -> str:
    """
    Rewrite SELECT projection to SELECT [DISTINCT|REDUCED] *.
    Keep everything else unchanged.
    """
    # This targets the first SELECT ... WHERE occurrence.
    # It preserves DISTINCT/REDUCED if present.
    pattern = re.compile(r"(?is)\bselect\b\s*(distinct|reduced)?\s*(.*?)\bwhere\b", re.DOTALL)
    m = pattern.search(query)
    if not m:
        return query

    modifier = (m.group(1) or "").strip()
    repl = "SELECT " + (modifier + " " if modifier else "") + "* WHERE"
    return query[:m.start()] + repl + query[m.end():]


def _replace_once(haystack: str, needle: str, replacement: str) -> Tuple[str, bool]:
    """
    Replace the first occurrence of needle with replacement.
    Return (new_string, replaced_flag).
    """
    idx = haystack.find(needle)
    if idx == -1:
        return haystack, False
    return haystack[:idx] + replacement + haystack[idx + len(needle):], True


def rewrite_sparql_keep_structure(query_str: str, strip_service: bool = True) -> str:
    """
    Rewrite SPARQL:
      - Expand only SequencePath '/' into multiple triples with intermediate vars
      - Change SELECT projection to query all variables (SELECT *), keep modifiers
      - Keep all other textual structure unchanged (OPTIONAL/FILTER/UNION/...).

    NOTE:
      - The path expansions are located via rdflib algebra, then we perform a minimal textual substitution:
        we replace the exact original triple string (as rendered by rdflib n3()) with expanded triples.
      - If the original query uses prefixes, rdflib's n3() rendering may output full IRIs. This may cause
        a mismatch when locating the exact triple text in the original query. To improve match rate,
        we attempt multiple renderings and a whitespace-tolerant fallback.
    """
    original = query_str
    q = _ensure_full_query(original)

    if strip_service:
        q = _strip_service_blocks(q)

    # Parse & translate
    parsed = sparql_parser.parseQuery(q)
    algebra = translateQuery(parsed)

    # 1) Rewrite SELECT -> SELECT * (minimal)
    q2 = _rewrite_select_to_star(q)

    # 2) Find all SequencePath triples and expand them
    path_items = _collect_path_triples_from_algebra(algebra)

    # Nothing to do
    if not path_items:
        return q2

    # We will apply replacements on q2 (the query after SELECT rewrite).
    rewritten = q2

    for (s, path_p, o, expanded_triples) in path_items:
        # Render the original triple in a "canonical" way
        # rdflib typically renders SequencePath as "p1 / p2" (with spaces) in n3()
        orig_stmt = f"{_term_to_sparql(s)} {_term_to_sparql(path_p)} {_term_to_sparql(o)} ."

        # Render expanded triples
        expanded_stmts = "\n".join(
            f"{_term_to_sparql(ss)} {_term_to_sparql(pp)} {_term_to_sparql(oo)} ."
            for (ss, pp, oo) in expanded_triples
        )

        # Try direct replacement first
        rewritten, ok = _replace_once(rewritten, orig_stmt, expanded_stmts)
        if ok:
            continue

        # Fallback #1: try collapsing multiple spaces in orig_stmt and do a regex whitespace-tolerant match
        # This helps when query has different spacing/newlines.
        # WARNING: Still "minimal" because we only replace one matched occurrence.
        ws_norm = re.sub(r"\s+", r"\\s+", re.escape(orig_stmt.strip()))
        regex = re.compile(ws_norm, flags=re.MULTILINE)

        m = regex.search(rewritten)
        if m:
            rewritten = rewritten[:m.start()] + expanded_stmts + rewritten[m.end():]
            continue

        # If we cannot find the exact statement, we skip it to avoid corrupting structure.
        # You can log/raise here if you prefer strict behavior.

    return rewritten


# ----------------------------
# Triple extraction API (merged)
# ----------------------------

def sparql_to_triples(query_str: str, strip_service: bool = True):
    """
    Extract expanded triples from a SPARQL query:
      - Optionally strip SERVICE wrappers
      - Expand '/' paths with intermediate variables
      - Shorten IRIs to last token (e.g., P31, Q32096)

    Returns:
        [[s, p, o], ...]
    """
    q = _ensure_full_query(query_str)
    if strip_service:
        q = _strip_service_blocks(q)

    parsed = sparql_parser.parseQuery(q)
    algebra = translateQuery(parsed)

    triple_terms = _collect_triples_from_algebra(algebra)

    return [[_term_to_str_short(s), _term_to_str_short(p), _term_to_str_short(o)]
            for (s, p, o) in triple_terms]


# ----------------------------
# Convenience: do both
# ----------------------------

def rewrite_and_extract(query_str: str, strip_service: bool = True):
    rewritten_query = rewrite_sparql_keep_structure(query_str, strip_service=strip_service)
    triples = sparql_to_triples(rewritten_query, strip_service=False)  # already stripped if enabled
    return rewritten_query, triples

def normalize(graph):
    gmid = []
    gstr = []
    for g in graph:
        if g[-1][0] == 'Q' and g[-1][1:].isdigit():
            gmid.append(g)
        else:
            gstr.append(g)
    return gmid + gstr

# remove the * at the end of the relation in the graph
def remove_relation_star(graph):
    return [[s, (p[:-1] if p.endswith('*') else p), o] for s,p,o in graph]
     
def process_one_sample(sample: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    process_sample = dict()
    query = sample["sparql"]
    #print(query)
    triples=sparql_to_triples(query)
    #print(triples)
    triples_used=remove_relation_star(triples)
    #print(triples_used)
    rewrite_sparql=rewrite_sparql_keep_structure(query)
    #print(rewrite_sparql)
    result = run_sparql(rewrite_sparql)
    #print(result)
    # --- Safely extract bindings and decide whether results are non-empty ---
    if result is None:
        bindings = []
    else:
        bindings = result.get("results", {}).get("bindings", [])
    FLAG = (len(bindings) > 0) and (bindings != [{}])
    
    if FLAG:
        # --- Convert endpoint bindings into a list of { "?var": value } dicts ---
        relist = []
        for r in bindings:
            temp = {}
            for var, val_dict in r.items():
                v = val_dict.get("value", "")
                if isinstance(v, str) and v.startswith("http"):
                    last = v.rsplit("/", 1)[-1]
                    # Shorten to Q-id only if it looks like Q123; otherwise keep full IRI
                    v = last if last.startswith("Q") else v
                temp["?" + var] = v
            relist.append(temp)
    
        # --- Construct gold subgraphs ---
        # Strategy A: if a variable is unbound, skip this triple only
        gold_mid = []
        for redict in relist:
            sub_one = []
            for (s, p, o) in triples_used:
                # Defensive: ensure we really have a triple
                try:
                    parts = [s, p, o]
                except Exception:
                    continue
    
                filled = []
                skip = False
                for token in parts:
                    if isinstance(token, str) and token.startswith("?"):
                        v = redict.get(token)
                        if v is None:
                            skip = True
                            break
                        filled.append(v)
                    else:
                        filled.append(token)
    
                if skip:
                    continue  # Strategy A: skip this triple if any variable is unbound
    
                sub_one.append(filled)
    
            gold_mid.append(sub_one)
    else:
        gold_mid = []
    
    process_sample["question"] = sample["utterance"]
    
    # --- Do not assume the answer variable name is "x";
    #     take the first variable in each result item instead ---
    answer_mid = []
    for item in sample.get("results", []):
        if not item:
            continue
        var = next(iter(item.keys()))  # e.g., x / answer / v / item
        v = item[var].get("value", "")
        if isinstance(v, str) and v.startswith("http"):
            v = v.rsplit("/", 1)[-1]
        answer_mid.append(v)
    
    process_sample["answer_mid"] = list(set(answer_mid))
    process_sample["graph_mid"] = gold_mid
    
    # --- Extract entities and relations with defensive length checks ---
    enset, reset = set(), set()
    for g in gold_mid:
        for t in g:
            if len(t) != 3:
                continue
            s, r, o = t
            if isinstance(s, str) and len(s) > 1 and s[0] == "Q" and s[1:].isdigit():
                enset.add(s)
            if isinstance(o, str) and len(o) > 1 and o[0] == "Q" and o[1:].isdigit():
                enset.add(o)
            reset.add(r)
    
    process_sample["entity"] = list(enset)
    process_sample["relation"] = list(reset)
    process_sample["sparql"] = sample["sparql"]
    return process_sample
    
def extract_graph_multiprocess(data, max_workers: Optional[int] = None):
    results = []
    total = len(data)
    if max_workers is None:
        max_workers = os.cpu_count()

    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        futures = [executor.submit(process_one_sample, sample) for sample in data]
        for f in tqdm(as_completed(futures), total=total, desc="Processing (multiprocess)"):
            res = f.result()
            if res is not None:
                results.append(res)
    return results

if __name__ == "__main__":
    data = json.load(open('../dataset/WikiWebQuestion/test.json', 'r', encoding='utf-8'))
    processed = extract_graph_multiprocess(data)
    os.makedirs('../graph/WikiWebQuestion',exist_ok=True)
    out_path = '../graph/WikiWebQuestion/origin.json'
    json.dump(processed, open(out_path, 'w', encoding='utf-8'), indent=2, ensure_ascii=False)
    print(f"Saved {len(processed)} items to {out_path}")