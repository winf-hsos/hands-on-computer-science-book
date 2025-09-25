#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import re
import sys
import argparse
import unicodedata
from pathlib import Path
from typing import Any, List, Dict, Tuple
import yaml  # pip install pyyaml

# ---------- Regex ----------------------------------------------------------

# \gloss{Begriff}{Definition}{Kategorie?}
GLOSS_RE = re.compile(
    r'\\gloss\{([\s\S]*?)\}\{([\s\S]*?)\}(?:\{([\s\S]*?)\})?',
    re.MULTILINE
)

# Kapitel-Überschrift mit {#id}
HEAD_ID_RE = re.compile(
    r'^\s{0,3}#{1,2}\s+.*?\{[^}]*#([A-Za-z0-9:_\-\.\+]+)[^}]*\}\s*$',
    re.MULTILINE
)

# Abschnittsebene 2 (## … {#id})
SEC2_RE = re.compile(
    r'^\s{0,3}##\s+.*?\{[^}]*#([A-Za-z0-9:_\-\.\+]+)[^}]*\}\s*$',
    re.MULTILINE
)

# Block-Marker für Kapitel-Glossar
BLOCK_PATTERN = re.compile(
    r"<!-- CHAPTER_GLOSSARY_START -->([\s\S]*?)<!-- CHAPTER_GLOSSARY_END -->",
    re.MULTILINE,
)

# ---------- Helpers --------------------------------------------------------

def nfc(s: str) -> str:
    return unicodedata.normalize("NFC", s)

def norm_space(s: str) -> str:
    return re.sub(r"\s+", " ", s.strip())

def ascii_slug(s: str) -> str:
    s_norm = unicodedata.normalize("NFKD", s)
    s_ascii = s_norm.encode("ascii", "ignore").decode("ascii")
    s_ascii = s_ascii.lower().strip()
    s_ascii = re.sub(r"\s+", "-", s_ascii)
    s_ascii = re.sub(r"[^a-z0-9\-_]", "", s_ascii)
    return s_ascii

def strip_code(text: str) -> str:
    text = re.sub(r"```[\s\S]*?```", "", text)
    text = re.sub(r"~~~[\s\S]*?~~~", "", text)
    text = re.sub(r"<pre[\s\S]*?</pre>", "", text, flags=re.IGNORECASE)
    text = re.sub(r"<code[\s\S]*?</code>", "", text, flags=re.IGNORECASE)
    text = re.sub(r"`[^`]*`", "", text)
    return text

def md_escape(s: str) -> str:
    return s.replace("\\", "\\\\").replace("|", "\\|")

# ---------- Kapitel aus _quarto.yml lesen ---------------------------------

def _flatten_chapters(node: Any) -> List[str]:
    out: List[str] = []
    if not node:
        return out
    if isinstance(node, str):
        out.append(node)
    elif isinstance(node, list):
        for it in node:
            out.extend(_flatten_chapters(it))
    elif isinstance(node, dict):
        if "chapters" in node:
            out.extend(_flatten_chapters(node["chapters"]))
        elif "appendices" in node:
            out.extend(_flatten_chapters(node["appendices"]))
        elif "file" in node:
            out.append(node["file"])
    return out

def discover_chapter_files(quarto_yml: Path) -> List[Path]:
    cfg = yaml.safe_load(quarto_yml.read_text(encoding="utf-8"))
    book = cfg.get("book") or {}
    files = _flatten_chapters(book.get("chapters")) + _flatten_chapters(book.get("appendices"))
    base = quarto_yml.parent
    return [(base / f).resolve() for f in files if isinstance(f, str) and f.endswith(".qmd")]

# ---------- Extraktion ----------------------------------------------------

def first_heading_id(text: str, path: Path) -> str:
    m = HEAD_ID_RE.search(text)
    return m.group(1) if m else path.stem.replace(" ", "-").lower()

def extract_gloss_entries_with_section(text: str) -> List[Dict[str, str]]:
    out: List[Dict[str, str]] = []

    # Abschnitts-Labels (## … {#id}) merken
    sec_positions: List[Tuple[int, str]] = []
    for m in SEC2_RE.finditer(text):
        sec_positions.append((m.start(), m.group(1)))

    txt = strip_code(text)

    for m in GLOSS_RE.finditer(txt):
        begriff = norm_space(nfc(m.group(1)))
        definition = norm_space(nfc(m.group(2)))
        kategorie = norm_space(nfc(m.group(3))) if m.group(3) else ""
        if ascii_slug(begriff) == "":
            continue

        # Abschnitt bestimmen
        pos = m.start()
        sec_id = ""
        for (spos, sid) in reversed(sec_positions):
            if spos <= pos:
                sec_id = sid
                break

        out.append(
            {
                "begriff": begriff,
                "definition": definition,
                "kategorie": kategorie,
                "section": sec_id,
            }
        )
    return out

# ---------- Tabellenbau ---------------------------------------------------

def global_table(rows: List[Dict[str, Any]]) -> str:
    header = "| Begriff | Kategorie | Definition | Kapitel |\n|---|---|---|---|\n"
    body = []
    for r in rows:
        term = r["begriff"]
        anchor = f"gl-{ascii_slug(term)}"
        begriff_cell = f'<span id="{anchor}">{md_escape(term)}</span>'
        kat = md_escape(r.get("kategorie", ""))
        definition = md_escape(r["definition"])
        kapitel_cell = ", ".join(f"@{kid}" for kid in r["kapitel_ids"])
        body.append(f"| {begriff_cell} | {kat} | {definition} | {kapitel_cell} |")
    return header + "\n".join(body) + "\n"

def chapter_table(rows: List[Dict[str, str]]) -> str:
    header = "| Begriff | Kategorie | Wo? |\n|---|---|---|\n"
    body = []
    for r in rows:
        term = md_escape(r["begriff"])
        begriff_cell = f"**{term}**"  # fett
        kat = md_escape(r.get("kategorie", ""))

        sec_id = r.get("section", "")
        if sec_id:
            wo_cell = f"@{sec_id}"
        else:
            wo_cell = ""

        body.append(f"| {begriff_cell} | {kat} | {wo_cell} |")
    return header + "\n".join(body) + "\n"

# ---------- Hauptlogik ----------------------------------------------------

def build(quarto_yml: Path, out_path: Path, glossary_html: str):
    files = discover_chapter_files(quarto_yml)
    if not files:
        print("Warnung: Keine .qmd Kapitel aus _quarto.yml gefunden.", file=sys.stderr)

    out_resolved = out_path.resolve()
    files = [f for f in files if f.resolve() != out_resolved and f.stem != out_resolved.stem]

    chap_order: Dict[str, int] = {}
    rows_all: List[Dict[str, str]] = []
    per_chapter: Dict[Path, Tuple[str, List[Dict[str, str]]]] = {}

    for idx, f in enumerate(files):
        txt = f.read_text(encoding="utf-8")
        chap_id = first_heading_id(txt, f)
        chap_order.setdefault(chap_id, idx)
        entries = extract_gloss_entries_with_section(txt)
        per_chapter[f] = (chap_id, entries)
        for e in entries:
            rows_all.append(
                {
                    "begriff": e["begriff"],
                    "definition": e["definition"],
                    "kategorie": e["kategorie"],
                    "kapitel": chap_id,
                }
            )

    # Globale Aggregation
    agg: Dict[tuple, Dict[str, Any]] = {}
    for r in rows_all:
        key = (r["begriff"].casefold(), r["kategorie"].casefold())
        cur = agg.get(key)
        if cur is None:
            agg[key] = {
                "begriff": r["begriff"],
                "kategorie": r["kategorie"],
                "definition": r["definition"],
                "kapitel_ids": {r["kapitel"]},
            }
        else:
            cur["kapitel_ids"].add(r["kapitel"])

    rows_global: List[Dict[str, Any]] = []
    for v in agg.values():
        kids = sorted(v["kapitel_ids"], key=lambda kid: chap_order.get(kid, 10**9))
        rows_global.append(
            {
                "begriff": v["begriff"],
                "kategorie": v["kategorie"],
                "definition": v["definition"],
                "kapitel_ids": kids,
            }
        )
    rows_global.sort(key=lambda r: (r["begriff"].casefold(), r["kategorie"].casefold()))

    # --- globale Glossarseite ---
    dt_open = '::: {#glossar-table .datatable data-order=\'[[0,"asc"]]\' }'
    dt_close = ":::"
    global_qmd = (
        "# Glossar {.unnumbered}\n\n"
        f"{dt_open}\n{global_table(rows_global)}{dt_close}\n\n"
    )
    out_path.write_text(global_qmd, encoding="utf-8")
    print(f"Wrote {out_path} ({len(rows_global)} Einträge)")

    # --- Kapitel-Glossare ---
    for f, (chap_id, entries) in per_chapter.items():
        if not entries:
            continue
        txt = f.read_text(encoding="utf-8")
        if not BLOCK_PATTERN.search(txt):
            continue

        # Einträge pro Kapitel (erste Definition gewinnt)
        seen: Dict[Tuple[str, str], Dict[str, str]] = {}
        for e in entries:
            key = (e["begriff"].casefold(), e["kategorie"].casefold())
            if key not in seen:
                seen[key] = e
        rows_chapter = list(seen.values())  # NICHT alphabetisch sortieren!

        chap_dt_open = "::: {.datatable-nosearch data-order='[[2,\"asc\"]]'}"
        chap_dt_close = ":::"
        table_block = (
            "<!-- CHAPTER_GLOSSARY_START -->\n"
            + chap_dt_open + "\n"
            + chapter_table(rows_chapter)
            + chap_dt_close + "\n"
            + "<!-- CHAPTER_GLOSSARY_END -->"
        )

        new_txt = BLOCK_PATTERN.sub(table_block, txt)
        if new_txt != txt:
            f.write_text(new_txt, encoding="utf-8")
            print(f"Updated chapter glossary in {f}")

# ---------- CLI -----------------------------------------------------------

def main():
    ap = argparse.ArgumentParser(
        description="Baut globales Glossar und Kapitel-Zusammenfassungen aus \\gloss{...}{...}{Kategorie?}."
    )
    ap.add_argument("-q", "--quarto", default="_quarto.yml", help="Pfad zu _quarto.yml")
    ap.add_argument("-o", "--output", default="glossary.qmd", help="Zieldatei für globales Glossar (QMD)")
    ap.add_argument("--glossary-html", default="glossary.html",
                    help="Linkziel zum globalen Glossar (Basename, z.B. 'glossary.html')")
    args = ap.parse_args()

    build(
        quarto_yml=Path(args.quarto).resolve(),
        out_path=Path(args.output).resolve(),
        glossary_html=args.glossary_html,
    )

if __name__ == "__main__":
    main()
