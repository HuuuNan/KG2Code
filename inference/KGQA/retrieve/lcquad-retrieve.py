import os
import json
from typing import Dict, Any, Optional, List, Set, Tuple
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

PREFIX = '''PREFIX wd:       <http://www.wikidata.org/entity/>
PREFIX wds:      <http://www.wikidata.org/entity/statement/>
PREFIX wdv:      <http://www.wikidata.org/value/>
PREFIX wdt:      <http://www.wikidata.org/prop/direct/>
PREFIX wdtn:     <http://www.wikidata.org/prop/direct-normalized/>
PREFIX p:        <http://www.wikidata.org/prop/>
PREFIX ps:       <http://www.wikidata.org/prop/statement/>
PREFIX psv:      <http://www.wikidata.org/prop/statement/value/>
PREFIX psn:      <http://www.wikidata.org/prop/statement/value-normalized/>
PREFIX pq:       <http://www.wikidata.org/prop/qualifier/>
PREFIX pqv:      <http://www.wikidata.org/prop/qualifier/value/>
PREFIX pqn:      <http://www.wikidata.org/prop/qualifier/value-normalized/>
PREFIX pr:       <http://www.wikidata.org/prop/reference/>
PREFIX prv:      <http://www.wikidata.org/prop/reference/value/>
PREFIX prn:      <http://www.wikidata.org/prop/reference/value-normalized/>
PREFIX wikibase: <http://wikiba.se/ontology#>
PREFIX bd:       <http://www.bigdata.com/rdf#>

PREFIX rdf:      <http://www.w3.org/1999/02/22-rdf-syntax-ns#>
PREFIX rdfs:     <http://www.w3.org/2000/01/rdf-schema#>
PREFIX xsd:      <http://www.w3.org/2001/XMLSchema#>
PREFIX owl:      <http://www.w3.org/2002/07/owl#>

PREFIX schema:   <http://schema.org/>
PREFIX skos:     <http://www.w3.org/2004/02/skos/core#>
PREFIX prov:     <http://www.w3.org/ns/prov#>
PREFIX geo:      <http://www.opengis.net/ont/geosparql#>
PREFIX foaf:     <http://xmlns.com/foaf/0.1/>
'''


class VarGenerator:
    """Generate fresh rdflib Variables with a prefix (e.g., _path1, _path2, ...)."""

    def __init__(self, prefix: str = "_path", used_names: Optional[Set[str]] = None):
        self.prefix = prefix
        self.counter = 0
        self.used_names = set(used_names or set())

    def new(self) -> Variable:
        """Return a fresh Variable that does not collide with existing ones."""
        while True:
            self.counter += 1
            name = f"{self.prefix}{self.counter}"
            if name not in self.used_names:
                self.used_names.add(name)
                return Variable(name)


# ----------------------------
# Short formatting helpers
# ----------------------------

def _shorten_iri_like(s: str) -> str:
    """Shorten an IRI string to its last token (after / or #)."""
    s = s.strip().replace("<", "").replace(">", "")
    pos = max(s.rfind("/"), s.rfind("#"))
    return s[pos + 1:] if pos != -1 else s


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
    """Convert an rdflib term to a SPARQL fragment."""
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

        # Handle string literals
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

        # Detect SERVICE keyword
        if (i + 7 <= n and s[i:i + 7].lower() == "service"
                and is_word_boundary(i - 1) and is_word_boundary(i + 7)):
            j = i + 7

            # Skip whitespace
            while j < n and s[j].isspace():
                j += 1

            # Find the first '{' after SERVICE target
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

            # Find matching '}'
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

            inner = s[j + 1:k]
            out.append(" ")
            out.append(inner)
            out.append(" ")
            i = k + 1
            continue

        out.append(ch)
        i += 1

    return "".join(out)


# ----------------------------
# Query normalization
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


def _extract_header_and_where_block(full_query: str) -> Tuple[str, str]:
    """
    Extract:
      - header: PREFIX/BASE lines
      - where_content: text inside WHERE { ... } (excluding the outer braces)
    """
    q = _ensure_full_query(full_query)

    # Extract header
    lines = q.splitlines()
    header_lines = []
    idx = 0
    while idx < len(lines):
        line = lines[idx].strip()
        if line.lower().startswith("prefix ") or line.lower().startswith("base "):
            header_lines.append(lines[idx])
            idx += 1
        else:
            break
    header = "\n".join(header_lines).strip()

    body = "\n".join(lines[idx:])

    # Find WHERE keyword
    m = re.search(r"(?is)\bwhere\b", body)
    if not m:
        raise ValueError("No WHERE found in query body.")

    pos = m.end()
    n = len(body)

    # Find the first '{' after WHERE (skip strings safely)
    i = pos
    in_str = None
    while i < n:
        ch = body[i]
        if in_str:
            if ch == in_str:
                in_str = None
            elif ch == "\\" and i + 1 < n:
                i += 1
        else:
            if ch in ("'", '"'):
                in_str = ch
            elif ch == "{":
                break
        i += 1

    if i >= n or body[i] != "{":
        raise ValueError("Cannot find '{' after WHERE.")

    # Match braces to find the end of WHERE block
    brace = 0
    j = i
    in_str = None
    while j < n:
        ch = body[j]
        if in_str:
            if ch == in_str:
                in_str = None
            elif ch == "\\" and j + 1 < n:
                j += 1
        else:
            if ch in ("'", '"'):
                in_str = ch
            elif ch == "{":
                brace += 1
            elif ch == "}":
                brace -= 1
                if brace == 0:
                    break
        j += 1

    if j >= n:
        raise ValueError("Unclosed WHERE braces.")

    where_content = body[i + 1:j]
    return header, where_content


def _rebuild_as_select_star(full_query: str, strip_service: bool = True) -> str:
    """
    Rebuild an arbitrary SPARQL query (ASK / COUNT / SELECT / fragments) into:

        [PREFIX/BASE...]
        SELECT * WHERE { ... }

    Why this is needed:
      - Some queries do not explicitly include the WHERE keyword, e.g.:
            SELECT (COUNT(?sub) AS ?value ) { ?sub wdt:P1716 wd:Q2766 }
      - Some queries may be fragments like "{ ... }" or start with "SERVICE ...".
      - For subgraph construction, we keep only the graph pattern block and drop
        solution modifiers (GROUP BY / HAVING / ORDER BY / LIMIT / OFFSET) by design.

    Strategy:
      1) Normalize query into a parseable form.
      2) Optionally strip SERVICE wrappers.
      3) Extract PREFIX/BASE header.
      4) Extract the first top-level group graph pattern "{ ... }" from the body:
         - Prefer the group after WHERE if WHERE exists.
         - Otherwise, take the first top-level "{ ... }" found.
      5) Wrap it as "SELECT * WHERE { ... }".
    """
    q = _ensure_full_query(full_query)
    if strip_service:
        q = _strip_service_blocks(q)

    # Split PREFIX/BASE header from the rest
    lines = q.splitlines()
    header_lines = []
    idx = 0
    while idx < len(lines):
        line = lines[idx].strip()
        if line.lower().startswith("prefix ") or line.lower().startswith("base "):
            header_lines.append(lines[idx])
            idx += 1
        else:
            break
    header = "\n".join(header_lines).strip()
    body = "\n".join(lines[idx:])

    def _find_group_block(text: str, start_pos: int = 0) -> Tuple[int, int]:
        """
        Find the first top-level '{ ... }' block in `text` starting at `start_pos`.
        Returns (lbrace_index, rbrace_index) inclusive indices for the braces.
        This scanner is robust to quoted strings and escaped quotes.
        """
        n = len(text)
        i = start_pos
        in_str = None

        # Find the first '{'
        while i < n:
            ch = text[i]
            if in_str:
                if ch == in_str:
                    in_str = None
                elif ch == "\\" and i + 1 < n:
                    i += 1
            else:
                if ch in ("'", '"'):
                    in_str = ch
                elif ch == "{":
                    break
            i += 1

        if i >= n or text[i] != "{":
            raise ValueError("Cannot find a group graph pattern '{ ... }' in query.")

        # Find its matching '}'
        brace = 0
        j = i
        in_str = None
        while j < n:
            ch = text[j]
            if in_str:
                if ch == in_str:
                    in_str = None
                elif ch == "\\" and j + 1 < n:
                    j += 1
            else:
                if ch in ("'", '"'):
                    in_str = ch
                elif ch == "{":
                    brace += 1
                elif ch == "}":
                    brace -= 1
                    if brace == 0:
                        return i, j
            j += 1

        raise ValueError("Unclosed '{ ... }' block in query.")

    # Prefer extracting the group after WHERE if WHERE exists; otherwise fall back to first '{...}'
    m_where = re.search(r"(?is)\bwhere\b", body)
    if m_where:
        l, r = _find_group_block(body, start_pos=m_where.end())
    else:
        l, r = _find_group_block(body, start_pos=0)

    where_content = body[l + 1:r]

    rebuilt_parts = []
    if header:
        rebuilt_parts.append(header)
    rebuilt_parts.append(f"SELECT * WHERE {{\n{where_content}\n}}")
    return "\n".join(rebuilt_parts).strip()


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
    """Collect triples from rdflib SPARQL algebra, expanding SequencePath into multiple triples."""
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
    """Rewrite SELECT projection to SELECT [DISTINCT|REDUCED] * (keep modifier)."""
    pattern = re.compile(r"(?is)\bselect\b\s*(distinct|reduced)?\s*(.*?)\bwhere\b", re.DOTALL)
    m = pattern.search(query)
    if not m:
        return query
    modifier = (m.group(1) or "").strip()
    repl = "SELECT " + (modifier + " " if modifier else "") + "* WHERE"
    return query[:m.start()] + repl + query[m.end():]


def _replace_once(haystack: str, needle: str, replacement: str) -> Tuple[str, bool]:
    """Replace the first occurrence of needle with replacement."""
    idx = haystack.find(needle)
    if idx == -1:
        return haystack, False
    return haystack[:idx] + replacement + haystack[idx + len(needle):], True


def rewrite_sparql_keep_structure(query_str: str, strip_service: bool = True) -> str:
    """
    Rewrite SPARQL:
      - Expand only SequencePath '/' into multiple triples with intermediate vars
      - Change SELECT projection to SELECT * (keep modifiers)
      - Keep all other textual structure unchanged
    """
    original = query_str
    q = _ensure_full_query(original)

    if strip_service:
        q = _strip_service_blocks(q)

    parsed = sparql_parser.parseQuery(q)
    algebra = translateQuery(parsed)

    # 1) Rewrite SELECT -> SELECT * (minimal)
    q2 = _rewrite_select_to_star(q)

    # 2) Find all SequencePath triples and expand them
    path_items = _collect_path_triples_from_algebra(algebra)
    if not path_items:
        return q2

    rewritten = q2

    for (s, path_p, o, expanded_triples) in path_items:
        orig_stmt = f"{_term_to_sparql(s)} {_term_to_sparql(path_p)} {_term_to_sparql(o)} ."

        expanded_stmts = "\n".join(
            f"{_term_to_sparql(ss)} {_term_to_sparql(pp)} {_term_to_sparql(oo)} ."
            for (ss, pp, oo) in expanded_triples
        )

        rewritten, ok = _replace_once(rewritten, orig_stmt, expanded_stmts)
        if ok:
            continue

        # Whitespace-tolerant fallback (replace only first match)
        ws_norm = re.sub(r"\s+", r"\\s+", re.escape(orig_stmt.strip()))
        regex = re.compile(ws_norm, flags=re.MULTILINE)
        m = regex.search(rewritten)
        if m:
            rewritten = rewritten[:m.start()] + expanded_stmts + rewritten[m.end():]
            continue

        # If not found, skip to avoid corrupting the query

    return rewritten


# ----------------------------
# Triple extraction API
# ----------------------------

def sparql_to_triples(query_str: str, strip_service: bool = True):
    """
    Extract expanded triples from a SPARQL query:
      - Optionally strip SERVICE wrappers
      - Expand '/' paths with intermediate variables
      - Shorten IRIs to last token (e.g., P31, Q32096)
    """
    q = _ensure_full_query(query_str)
    if strip_service:
        q = _strip_service_blocks(q)

    parsed = sparql_parser.parseQuery(q)
    algebra = translateQuery(parsed)

    triple_terms = _collect_triples_from_algebra(algebra)

    return [[_term_to_str_short(s), _term_to_str_short(p), _term_to_str_short(o)]
            for (s, p, o) in triple_terms]


def normalize(graph):
    """Sort graph so Q-ids are placed before string nodes."""
    gmid = []
    gstr = []
    for g in graph:
        if g[-1][0] == 'Q' and g[-1][1:].isdigit():
            gmid.append(g)
        else:
            gstr.append(g)
    return gmid + gstr


def remove_relation_star(graph):
    """Remove trailing '*' in relation names if present."""
    return [[s, (p[:-1] if isinstance(p, str) and p.endswith('*') else p), o] for s, p, o in graph]


def has_unknown_var(triples: List[List[Any]]) -> bool:
    """
    Return True if any triple contains an unknown SPARQL variable (string starting with '?').
    If no variables exist in the extracted triples, we can skip endpoint querying in some cases.
    """
    for s, p, o in triples:
        if isinstance(s, str) and s.startswith("?"):
            return True
        if isinstance(p, str) and p.startswith("?"):
            return True
        if isinstance(o, str) and o.startswith("?"):
            return True
    return False


def _is_ask_answer_no(ans_raw: Any) -> bool:
    """Return True if the answer corresponds to ASK=no."""
    if isinstance(ans_raw, list) and len(ans_raw) == 1 and ans_raw[0] == "no":
        return True
    if isinstance(ans_raw, str) and ans_raw == "no":
        return True
    return False


def _is_ask_answer_yes(ans_raw: Any) -> bool:
    """Return True if the answer corresponds to ASK=yes."""
    if isinstance(ans_raw, list) and len(ans_raw) == 1 and ans_raw[0] == "yes":
        return True
    if isinstance(ans_raw, str) and ans_raw == "yes":
        return True
    return False


def process_one_sample(sample: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    """
    Process one sample:
      - Keep the original ASK/COUNT special handling:
          * ASK with answer "no" -> no subgraph (graph_mid = [])
          * ASK with answer "yes" -> rebuild to SELECT * WHERE {...} and query endpoint
          * COUNT -> rebuild to SELECT * WHERE {...} and query endpoint
          * normal SELECT -> expand paths and rewrite SELECT -> *
      - Additional optimization:
          * If no extracted triples -> return empty graph and skip endpoint querying
          * If extracted triples contain no unknown variables -> directly use them as a single subgraph,
            EXCEPT for ASK(no) which must return empty graph.
    """
    ans = sample.get("answer")
    if isinstance(ans, list) and len(ans) == 0:
        return None
    process_sample: Dict[str, Any] = {}

    full_query = PREFIX + sample["sparql_wikidata"]

    # Extract triples for later subgraph construction
    triples = sparql_to_triples(full_query, strip_service=True)
    triples_used = remove_relation_star(triples)

    # Detect query types based on the provided answer format
    ans_raw = sample.get("answer")
    is_ask = False
    is_count = False
    if isinstance(ans_raw, list) and len(ans_raw) == 1:
        if ans_raw[0] in ["yes", "no"]:
            is_ask = True
        if ans_raw[0].isdigit():
            is_count = True
    elif isinstance(ans_raw, str):
        if ans_raw in ["yes", "no"]:
            is_ask = True
        if ans_raw.isdigit():
            is_count = True

    # ----------------------------
    # ASK(no) must have NO subgraph, regardless of variable-free triples
    # ----------------------------
    if is_ask and _is_ask_answer_no(ans_raw):
        process_sample["question"] = sample.get("question")
        if isinstance(ans_raw, list):
            process_sample["answer_mid"] = list(set(ans_raw))
        else:
            process_sample["answer_mid"] = [ans_raw]
        process_sample["graph_mid"] = []
        process_sample["entity"] = []
        process_sample["relation"] = []
        process_sample["sparql"] = sample.get("sparql_wikidata")
        return process_sample

    # ----------------------------
    # Optimization: If no triples were extracted, return empty graph and skip endpoint querying
    # ----------------------------
    if not triples_used:
        process_sample["question"] = sample.get("question")
        if isinstance(ans_raw, list):
            process_sample["answer_mid"] = list(set(ans_raw))
        else:
            process_sample["answer_mid"] = [ans_raw]
        process_sample["graph_mid"] = []
        process_sample["entity"] = []
        process_sample["relation"] = []
        process_sample["sparql"] = sample.get("sparql_wikidata")
        return process_sample

    # ----------------------------
    # Optimization: If triples contain no unknown variables, directly use them as ONE subgraph
    # This is allowed for non-ASK(no) cases, including ASK(yes), COUNT, and normal SELECT.
    # (For ASK(yes) this provides a shortcut when there is nothing to bind.)
    # ----------------------------
    if not has_unknown_var(triples_used):
        process_sample["question"] = sample.get("question")
        if isinstance(ans_raw, list):
            process_sample["answer_mid"] = list(set(ans_raw))
        else:
            process_sample["answer_mid"] = [ans_raw]

        gold_mid = [triples_used]
        process_sample["graph_mid"] = gold_mid

        enset, reset = set(), set()
        for t in triples_used:
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
        process_sample["sparql"] = sample.get("sparql_wikidata")
        return process_sample

    # ----------------------------
    # Original behavior: decide rewriting and query endpoint
    # ----------------------------
    if is_ask and _is_ask_answer_yes(ans_raw):
        # ASK + yes: rebuild as SELECT * WHERE { ... } and expand paths while keeping structure
        base_select = _rebuild_as_select_star(full_query, strip_service=True)
        rewrite_sparql = rewrite_sparql_keep_structure(base_select, strip_service=False)

    elif is_count:
        # COUNT query: rebuild as SELECT * WHERE { ... } to retrieve all relevant bindings for subgraph
        base_select = _rebuild_as_select_star(full_query, strip_service=True)
        rewrite_sparql = rewrite_sparql_keep_structure(base_select, strip_service=False)

    else:
        # Normal SELECT: expand paths and rewrite SELECT -> *
        rewrite_sparql = rewrite_sparql_keep_structure(full_query, strip_service=True)
    # Query the endpoint
    endpoint_result = run_sparql(rewrite_sparql)

    # Safely extract bindings and decide whether results are non-empty
    if endpoint_result is None:
        bindings = []
    else:
        bindings = endpoint_result.get("results", {}).get("bindings", [])
    flag = (len(bindings) > 0) and (bindings != [{}])

    if flag:
        # Convert endpoint bindings into a list of { "?var": value } dicts
        relist = []
        for r in bindings:
            temp = {}
            for var, val_dict in r.items():
                v = val_dict.get("value", "")
                if isinstance(v, str) and v.startswith("http"):
                    last = v.rsplit("/", 1)[-1]
                    v = last if last.startswith("Q") else v
                temp["?" + var] = v
            relist.append(temp)

        # Construct gold subgraphs:
        # If a variable is unbound in a specific solution mapping, skip that triple for that solution
        gold_mid = []
        for redict in relist:
            sub_one = []
            for (s, p, o) in triples_used:
                parts = [s, p, o]
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
                    continue
                sub_one.append(filled)
            gold_mid.append(sub_one)
    else:
        gold_mid = []

    process_sample["question"] = sample.get("question")
    if isinstance(ans_raw, list):
        process_sample["answer_mid"] = list(set(ans_raw))
    else:
        process_sample["answer_mid"] = [ans_raw]
    process_sample["graph_mid"] = gold_mid

    # Extract entities and relations from the constructed subgraphs
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
    process_sample["sparql"] = sample.get("sparql_wikidata")
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
    data = json.load(open('../dataset/LC-QuAD2.0/test_with_answer.json', 'r', encoding='utf-8'))
    processed = extract_graph_multiprocess(data)
    os.makedirs('../graph/LC-QuAD2.0', exist_ok=True)
    out_path = '../graph/LC-QuAD2.0/origin.json'
    json.dump(processed, open(out_path, 'w', encoding='utf-8'), indent=2, ensure_ascii=False)
    print(f"Saved {len(processed)} items to {out_path}")
