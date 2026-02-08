import json
from collections import defaultdict
from typing import List, Tuple, Iterable, Mapping, Any, Optional
import re

ResultT = defaultdict[str, defaultdict[str, List[str]]]


def split_text_and_json(s: str) -> Tuple[str, Optional[dict]]:
    # 1) Prefer a fenced ```json ... ``` block if present (last one wins)
    fenced = list(re.finditer(
        r"```(?:json)?\s*(\{[\s\S]*?\})\s*```", s, flags=re.IGNORECASE))
    if fenced:
        m = fenced[-1]
        json_str = m.group(1)
        try:
            obj = json.loads(json_str)
            text = (s[:m.start()] + s[m.end():]).strip()
            return text, obj
        except json.JSONDecodeError:
            pass

    # 2) Otherwise, extract the last top-level {...} object (brace-balanced, ignore braces in strings)
    stack = []
    starts = []
    objects = []
    in_str = False
    esc = False
    for i, ch in enumerate(s):
        if in_str:
            if esc:
                esc = False
            elif ch == "\\":
                esc = True
            elif ch == '"':
                in_str = False
            continue
        else:
            if ch == '"':
                in_str = True
            elif ch == '{':
                stack.append(i)
            elif ch == '}' and stack:
                start = stack.pop()
                if not stack:  # closed a top-level object
                    objects.append((start, i+1))

    if objects:
        start, end = objects[-1]
        json_str = s[start:end]
        try:
            obj = json.loads(json_str)
            text = (s[:start] + s[end:]).strip()
            return text, obj
        except json.JSONDecodeError:
            pass

    # 3) Fallback: no JSON found
    return s.strip(), None


def format_sources(contexts: Iterable[Mapping[str, Any]],
                   sources: list[dict],
                   dedupe: bool = True,
                   debug: bool = False,
                   debug_samples: int = 3) -> ResultT:
    result: ResultT = defaultdict(lambda: defaultdict(list))
    seen: set[Tuple[str, str, str]] = set() if dedupe else set()
    for i, doc in enumerate(contexts):
        raw = doc.get("_node_content")
        if not raw:
            if debug:
                print(f"[warn] idx={i}: missing '_node_content'")
            continue
        try:
            node = json.loads(raw)
        except json.JSONDecodeError:
            if debug:
                print(f"[warn] idx={i}: invalid JSON")
            continue
        meta = node.get("metadata") or {}
        source = str(meta.get("file_name") or meta.get("source") or "unknown")
        # ensure page key is a string to match the annotated type
        page_val = meta.get("page_label", meta.get("page", "unknown"))
        page = str(page_val) if page_val is not None else "unknown"
        if not any(
            d.get("file_name") == source and str(page) in {
                str(p) for p in d.get("page_labels", [])}
            for d in sources
        ):
            continue

        text = node.get("text") or ""
        if not text:
            if debug:
                print(f"[info] idx={i}: empty text, skipped")
            continue

        if dedupe:
            key = (source, page, text)
            if key in seen:
                if debug:
                    print(
                        f"[info] idx={i}: duplicate skipped (source={source}, page={page})")
                continue
            seen.add(key)
        result[source][page].append(text)
        if debug and i < debug_samples:
            print(
                f"[dbg] idx={i} source={source!r} page={page!r} text_len={len(text)}")
    return result
