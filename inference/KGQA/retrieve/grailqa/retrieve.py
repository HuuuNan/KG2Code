# -*- coding: utf-8 -*-
"""
SPARQL rewriting + triple extraction for subgraph construction (multiprocess version).

Key features:
- Normalize fragments into parseable SPARQL.
- Strip SERVICE blocks (optional).
- Expand ONLY SequencePath "/" into multiple triples with fresh intermediate vars.
- Rewrite SELECT projection to SELECT *:
    - If query is two-level (outer WHERE starts with a subquery), rewrite ALL SELECTs to *.
      (Robust to WHERE { { SELECT ... } } as well.)
    - Otherwise, rewrite only the outermost SELECT to *.
- Fix invalid subquery placement:
    WHERE { SELECT ... }  ->  WHERE { { SELECT ... } }
- Remove solution modifiers that can become invalid after SELECT * rewriting:
    GROUP BY / HAVING / ORDER BY / LIMIT / OFFSET
  (This is intended for subgraph extraction, not for exact query semantics.)

Multiprocessing:
- Use ProcessPoolExecutor to process each sample independently.
- Each worker returns one processed sample dict (or None to skip).
"""

import os
import json
import re
from typing import Dict, Any, Optional, List, Set, Tuple
from concurrent.futures import ProcessPoolExecutor, as_completed

from tqdm import tqdm

from rdflib import Variable, URIRef, Literal, BNode
from rdflib.plugins.sparql import parser as sparql_parser
from rdflib.plugins.sparql.algebra import translateQuery
from rdflib.plugins.sparql.parserutils import CompValue
from rdflib.paths import SequencePath, AlternativePath, MulPath, InvPath, NegatedPath

from query import run_sparql


# =========================================================
# Variable generator
# =========================================================

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


# =========================================================
# Formatting helpers
# =========================================================

def _shorten_iri_like(s: str) -> str:
    """Shorten an IRI string to its last token (after / or #)."""
    s = s.strip().replace("<", "").replace(">", "")
    pos = max(s.rfind("/"), s.rfind("#"))
    return s[pos + 1:] if pos != -1 else s


def _term_to_str_short(term) -> str:
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


# =========================================================
# SERVICE stripping (textual)
# =========================================================

def _strip_service_blocks(query: str) -> str:
    """
    Replace: SERVICE <something> { inner }
    with:    inner

    Textual preprocessing step. Robust to nested braces and quoted strings.
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


# =========================================================
# Query normalization
# =========================================================

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


def _rebuild_as_select_star(full_query: str, strip_service: bool = True) -> str:
    """
    Rebuild an arbitrary SPARQL query (ASK / COUNT / SELECT / fragments) into:

        [PREFIX/BASE...]
        SELECT * WHERE { ... }

    This keeps only the first main group graph pattern.
    It is designed for subgraph extraction, not exact query semantics.
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
        """Find the first top-level '{ ... }' block in `text` starting at `start_pos`."""
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

        # Find matching '}'
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

    # Prefer extracting group after WHERE if WHERE exists, else first block
    m_where = re.search(r"(?is)\bwhere\b", body)
    if m_where:
        l, r = _find_group_block(body, start_pos=m_where.end())
    else:
        l, r = _find_group_block(body, start_pos=0)

    where_content = body[l + 1:r].strip()

    # If WHERE content begins with a subquery, wrap it to make it syntactically valid:
    # WHERE { { SELECT ... } }
    if re.match(r"(?is)^select\b", where_content):
        where_content = "{\n" + where_content + "\n}"

    rebuilt_parts = []
    if header:
        rebuilt_parts.append(header)
    rebuilt_parts.append(f"SELECT * WHERE {{\n{where_content}\n}}")
    return "\n".join(rebuilt_parts).strip()


# =========================================================
# Algebra traversal & path expansion
# =========================================================

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


# =========================================================
# SELECT rewriting / subquery wrapping / modifier stripping
# =========================================================

def _outer_where_starts_with_subquery_select(query: str) -> bool:
    """
    Detect "two-level" pattern where the outer WHERE group starts with a subquery SELECT,
    allowing optional extra braces:
        WHERE { SELECT ... }
        WHERE { { SELECT ... } }
        WHERE { { { SELECT ... } } }
    """
    m = re.search(r"(?is)\bwhere\b", query)
    if not m:
        return False

    n = len(query)
    i = m.end()

    # Skip whitespace
    while i < n and query[i].isspace():
        i += 1

    # Find first '{' after WHERE, skipping strings
    in_str = None
    while i < n:
        ch = query[i]
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

    if i >= n or query[i] != "{":
        return False

    # Inside outer WHERE block, skip whitespace and any number of leading '{'
    j = i + 1
    while j < n:
        while j < n and query[j].isspace():
            j += 1
        if j < n and query[j] == "{":
            j += 1
            continue
        break

    if j >= n:
        return False

    return re.match(r"(?is)select\b", query[j:]) is not None


def rewrite_all_selects_to_star(query: str) -> str:
    """
    Rewrite EVERY occurrence of:
        SELECT [DISTINCT|REDUCED]? <projection> WHERE
    into:
        SELECT [DISTINCT|REDUCED]? * WHERE
    Includes subqueries.
    """
    pattern = re.compile(r"(?is)\bselect\b\s*(distinct|reduced)?\s*(.*?)\bwhere\b")

    def repl(m: re.Match) -> str:
        modifier = (m.group(1) or "").strip()
        return "SELECT " + (modifier + " " if modifier else "") + "* WHERE"

    return pattern.sub(repl, query)


def rewrite_outer_select_to_star(query: str) -> str:
    """
    Rewrite ONLY the outermost SELECT projection to SELECT [DISTINCT|REDUCED] *.
    Does not rewrite subqueries.
    """
    pattern = re.compile(r"(?is)\bselect\b\s*(distinct|reduced)?\s*(.*?)\bwhere\b", re.DOTALL)
    m = pattern.search(query)
    if not m:
        return query
    modifier = (m.group(1) or "").strip()
    repl = "SELECT " + (modifier + " " if modifier else "") + "* WHERE"
    return query[:m.start()] + repl + query[m.end():]


def wrap_subquery_in_outer_where_if_needed(query: str) -> str:
    """
    If outer WHERE block is:
        WHERE { SELECT ... }
    wrap it as:
        WHERE { { SELECT ... } }

    This is a minimal textual fix robust to nested braces and quoted strings.
    """
    m = re.search(r"(?is)\bwhere\b", query)
    if not m:
        return query

    n = len(query)
    i = m.end()

    # Skip whitespace
    while i < n and query[i].isspace():
        i += 1

    # Find first '{' after WHERE, skipping strings
    in_str = None
    while i < n:
        ch = query[i]
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

    if i >= n or query[i] != "{":
        return query

    # Check first non-space token inside braces
    j = i + 1
    while j < n and query[j].isspace():
        j += 1
    if j >= n or not re.match(r"(?is)select\b", query[j:]):
        return query

    # Find matching '}' for this outer WHERE block
    brace = 0
    k = i
    in_str = None
    while k < n:
        ch = query[k]
        if in_str:
            if ch == in_str:
                in_str = None
            elif ch == "\\" and k + 1 < n:
                k += 1
        else:
            if ch in ("'", '"'):
                in_str = ch
            elif ch == "{":
                brace += 1
            elif ch == "}":
                brace -= 1
                if brace == 0:
                    end = k
                    break
        k += 1
    else:
        return query

    inner = query[i + 1:end]
    return query[:i + 1] + "\n{\n" + inner + "\n}\n" + query[end:]


def strip_solution_modifiers(query: str) -> str:
    """
    Remove solution modifiers that frequently become invalid after SELECT * rewriting:
      - GROUP BY ...
      - HAVING ...
      - ORDER BY ...
      - LIMIT ...
      - OFFSET ...

    Best-effort textual stripper intended for subgraph extraction.
    This changes query semantics by design.
    """
    s = query

    # Remove GROUP BY / ORDER BY / HAVING blocks up to next modifier/brace/end
    s = re.sub(
        r"(?is)\bgroup\s+by\b.*?(?=(\border\s+by\b|\bhaving\b|\blimit\b|\boffset\b|\}|\bselect\b|$))",
        " ",
        s
    )
    s = re.sub(
        r"(?is)\border\s+by\b.*?(?=(\bhaving\b|\blimit\b|\boffset\b|\}|\bselect\b|$))",
        " ",
        s
    )
    s = re.sub(
        r"(?is)\bhaving\b.*?(?=(\blimit\b|\boffset\b|\}|\bselect\b|$))",
        " ",
        s
    )

    # Remove LIMIT and OFFSET (simple forms)
    s = re.sub(r"(?is)\blimit\b\s+\d+\s*", " ", s)
    s = re.sub(r"(?is)\boffset\b\s+\d+\s*", " ", s)

    # Clean up extra whitespace
    s = re.sub(r"[ \t]+\n", "\n", s)
    s = re.sub(r"\n{3,}", "\n\n", s)
    return s.strip()


def _replace_once(haystack: str, needle: str, replacement: str) -> Tuple[str, bool]:
    """Replace the first occurrence of needle with replacement."""
    idx = haystack.find(needle)
    if idx == -1:
        return haystack, False
    return haystack[:idx] + replacement + haystack[idx + len(needle):], True


def rewrite_sparql_keep_structure(query_str: str, strip_service: bool = True) -> str:
    """
    Rewrite SPARQL for subgraph extraction:

    - Expand only SequencePath '/' into multiple triples with intermediate vars.
    - If the query has an outer WHERE whose first token (allowing extra braces) is a subquery SELECT:
        - Rewrite ALL SELECT projections to SELECT * (keep DISTINCT/REDUCED).
        - Wrap subquery in an extra { ... } if needed to make it syntactically valid.
        - Strip solution modifiers (GROUP BY/HAVING/ORDER BY/LIMIT/OFFSET).
    - Otherwise:
        - Rewrite only outermost SELECT projection to SELECT *.
        - Keep subqueries as-is.
        - Still strip modifiers (prevents invalid query after projection change).
    - Keep other textual structure unchanged as much as possible.
    """
    q = _ensure_full_query(query_str)
    if strip_service:
        q = _strip_service_blocks(q)

    # Parse algebra from the normalized query (used only for path detection)
    parsed = sparql_parser.parseQuery(q)
    algebra = translateQuery(parsed)

    # 1) Rewrite SELECT projection(s)
    if _outer_where_starts_with_subquery_select(q):
        rewritten = rewrite_all_selects_to_star(q)
        rewritten = wrap_subquery_in_outer_where_if_needed(rewritten)
    else:
        rewritten = rewrite_outer_select_to_star(q)

    # 2) Strip modifiers to avoid invalid queries after SELECT * rewriting
    rewritten = strip_solution_modifiers(rewritten)

    # 3) Expand only SequencePath triples
    path_items = _collect_path_triples_from_algebra(algebra)
    if not path_items:
        return wrap_subquery_in_outer_where_if_needed(rewritten)

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

    return wrap_subquery_in_outer_where_if_needed(rewritten)


# =========================================================
# Triple extraction API
# =========================================================

def sparql_to_triples(query_str: str, strip_service: bool = True) -> List[List[str]]:
    """
    Extract expanded triples from a SPARQL query:
      - Optionally strip SERVICE wrappers
      - Expand '/' paths with intermediate variables
      - Shorten IRIs to last token
    """
    q = _ensure_full_query(query_str)
    if strip_service:
        q = _strip_service_blocks(q)

    parsed = sparql_parser.parseQuery(q)
    algebra = translateQuery(parsed)

    triple_terms = _collect_triples_from_algebra(algebra)
    return [[_term_to_str_short(s), _term_to_str_short(p), _term_to_str_short(o)]
            for (s, p, o) in triple_terms]


def remove_relation_star(graph: List[List[Any]]) -> List[List[Any]]:
    """Remove trailing '*' in relation names if present."""
    return [[s, (p[:-1] if isinstance(p, str) and p.endswith('*') else p), o] for s, p, o in graph]


def has_unknown_var(triples: List[List[Any]]) -> bool:
    """Return True if any triple contains an unbound SPARQL variable (string starting with '?')."""
    for s, p, o in triples:
        if isinstance(s, str) and s.startswith("?"):
            return True
        if isinstance(p, str) and p.startswith("?"):
            return True
        if isinstance(o, str) and o.startswith("?"):
            return True
    return False


# =========================================================
# VALUES extraction and replacement
# =========================================================

def extract_values_from_sparql(query_str: str) -> Dict[str, List[str]]:
    """
    Extract variable values from the VALUES clause of the SPARQL query.
    Returns a dict like {"?x1": ["m.abc", "m.def"], ...}.
    """
    values_dict: Dict[str, List[str]] = {}
    pattern = re.compile(r"VALUES\s+(\?[A-Za-z_]\w*)\s+\{(.*?)\}", re.DOTALL | re.IGNORECASE)
    matches = pattern.findall(query_str)

    for var, values in matches:
        values_list = [value.strip().lstrip(':') for value in values.splitlines() if value.strip()]
        values_dict[var] = values_list

    return values_dict


def replace_variables_in_triples(triples: List[List[Any]], values_dict: Dict[str, List[str]]) -> List[List[Any]]:
    """
    Replace variables in the triples with their values from the VALUES dictionary.
    If multiple values exist, use the first one (simple strategy).
    """
    new_triples: List[List[Any]] = []
    for triple in triples:
        new_triple = []
        for term in triple:
            if isinstance(term, str) and term.startswith('?'):
                if term in values_dict and values_dict[term]:
                    new_triple.append(values_dict[term][0])
                else:
                    new_triple.append(term)
            else:
                new_triple.append(term)
        new_triples.append(new_triple)
    return new_triples


# =========================================================
# ASK helpers
# =========================================================

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


# =========================================================
# Per-sample processing (worker function)
# =========================================================

def process_one_sample(sample: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    """
    Process one dataset sample (GrailQA-style):
      - Build triples from SPARQL (expand paths; apply VALUES replacement).
      - Detect ASK/COUNT by answer_name format.
      - ASK(no) -> empty subgraph.
      - If no triples -> empty subgraph.
      - If no unknown vars after VALUES replacement -> one subgraph, no endpoint query.
      - Otherwise:
          * Rebuild to SELECT * WHERE { ... }
          * Rewrite (SELECT *, wrap subquery, strip modifiers, expand paths)
          * Query endpoint for bindings
          * Fill triples per binding to construct subgraphs
    """
    process_sample: Dict[str, Any] = {}
    process_sample["question"] = sample.get("question")

    # Collect answers
    answer_mid = []
    answer_name = []
    for a in sample.get("answer", []):
        answer_mid.append(a.get("answer_argument"))
        if a.get("entity_name"):
            answer_name.append(a["entity_name"])
        else:
            answer_name.append(a.get("answer_argument"))
    process_sample["answer_mid"] = answer_mid
    process_sample["answer_name"] = answer_name

    sparql = sample.get("sparql_query", "")

    # Extract VALUES bindings and triples
    values_dict = extract_values_from_sparql(sparql)
    triples = sparql_to_triples(sparql, strip_service=True)
    updated_triples = replace_variables_in_triples(triples, values_dict)

    # Detect query types based on answer_name format
    ans_raw = sample.get("answer_name")
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

    # ASK(no): force empty subgraph
    if is_ask and _is_ask_answer_no(ans_raw):
        process_sample["graph_mid"] = []
        process_sample["entity"] = []
        process_sample["relation"] = []
        process_sample["sparql"] = sparql
        return process_sample

    # If no triples extracted: empty graph
    if len(updated_triples) == 0:
        process_sample["graph_mid"] = []
        process_sample["entity"] = []
        process_sample["relation"] = []
        process_sample["sparql"] = sparql
        return process_sample

    # If no unknown variables remain: use as one subgraph without querying endpoint
    if not has_unknown_var(updated_triples):
        gold_mid = [updated_triples]
        process_sample["graph_mid"] = gold_mid

        enset, reset = set(), set()
        for t in updated_triples:
            if len(t) != 3:
                continue
            s, r, o = t
            if isinstance(s, str) and s[:2] in ["m.", "n.", "g."]:
                enset.add(s)
            if isinstance(o, str) and o[:2] in ["m.", "n.", "g."]:
                enset.add(o)
            reset.add(r)

        process_sample["entity"] = list(enset)
        process_sample["relation"] = list(reset)
        process_sample["sparql"] = sparql
        return process_sample

    # Build a "binding-harvesting" query
    base_select = _rebuild_as_select_star(sparql, strip_service=True)
    rewrite_sparql = rewrite_sparql_keep_structure(base_select, strip_service=False)+"\nLIMIT 100"

    # Query the endpoint
    endpoint_result = run_sparql(rewrite_sparql)

    # Safely extract bindings
    if endpoint_result is None:
        bindings = []
    else:
        bindings = endpoint_result.get("results", {}).get("bindings", [])

    flag = (len(bindings) > 0) and (bindings != [{}])

    if flag:
        # Convert endpoint bindings into list of { "?var": value } dicts
        relist = []
        for r in bindings:
            temp = {}
            for var, val_dict in r.items():
                v = val_dict.get("value", "")
                if isinstance(v, str) and v.startswith("http"):
                    last = v.rsplit("/", 1)[-1]
                    v = last if last[:2] in ["m.", "n.", "g."] else v
                temp["?" + var] = v
            relist.append(temp)

        # Construct gold subgraphs per solution mapping
        gold_mid = []
        for redict in relist:
            sub_one = []
            for (s, p, o) in updated_triples:
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

    process_sample["graph_mid"] = gold_mid

    # Collect entities and relations from gold subgraphs
    enset = set()
    reset = set()
    for g in gold_mid:
        for t in g:
            if len(t) != 3:
                continue
            s, r, o = t
            if isinstance(s, str) and s[:2] in ["m.", "n.", "g."]:
                enset.add(s)
            if isinstance(o, str) and o[:2] in ["m.", "n.", "g."]:
                enset.add(o)
            reset.add(r)

    process_sample["entity"] = list(enset)
    process_sample["relation"] = list(reset)
    process_sample["sparql"] = sparql
    return process_sample


# =========================================================
# Multiprocessing driver
# =========================================================

def extract_graph_multiprocess(data: List[Dict[str, Any]], max_workers: Optional[int] = None) -> List[Dict[str, Any]]:
    """
    Run multiprocessing over all samples and aggregate results.

    Notes:
    - Uses as_completed() to stream finished tasks.
    - Exceptions in workers will be raised when calling future.result().
    """
    if max_workers is None:
        max_workers = os.cpu_count() or 1

    results: List[Dict[str, Any]] = []
    total = len(data)

    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        futures = [executor.submit(process_one_sample, sample) for sample in data]
        for f in tqdm(as_completed(futures), total=total, desc="Processing (multiprocess)"):
            res = f.result()
            if res is not None:
                results.append(res)

    return results


# =========================================================
# Main
# =========================================================

if __name__ == "__main__":
    # Load the dataset
    data_path = "../../dataset/grailqa/grailqa_v1.0_dev.json"
    data = json.load(open(data_path, "r", encoding="utf-8"))

    # Run multiprocessing extraction
    processed = extract_graph_multiprocess(data)

    # Save output
    os.makedirs("../../graph/grailqa", exist_ok=True)
    out_path = "../../graph/grailqa/origin.json"
    json.dump(processed, open(out_path, "w", encoding="utf-8"), indent=2, ensure_ascii=False)
    print(f"Saved {len(processed)} items to {out_path}")
