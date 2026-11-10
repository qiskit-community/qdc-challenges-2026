#!/usr/bin/env python3
"""Check the QDC 2026 cheat sheet and rebuild its generated sections.

    python3 tools/build_cheatsheet.py           rebuild the README views and the versions table
    python3 tools/build_cheatsheet.py --check   change nothing; exit 1 if a check fails or a generated section is out of date

Needs Python 3.11 or later and nothing else. Run it from anywhere; paths are relative to this file.

What it checks:
  * every document has front matter with title, tool_type, owner and last_verified;
  * every entry (a second-level heading with a Tags line) has When to use and Tags, uses known tag values,
    keeps the field order in FIELD_ORDER, and has an `updated` month only when it says what changed that month;
  * every relative link and anchor resolves, including the task links in tools/taxonomy.toml;
  * no sentence of 60 characters or more appears in two places (each fact lives once);
  * the generated sections match what the documents say.
"""
import argparse
import difflib
import posixpath
import re
import sys
import tomllib
import unicodedata
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
TAXONOMY = ROOT / "tools" / "taxonomy.toml"
README = "README.md"
VERSIONS_DOC = "guides/getting-started.md"
DOC_DIRS = ("reference", "guides")
FIELD_ORDER = ["When to use", "Tags", "Recent changes", "Current version", "Key APIs", "Watch out", "Related", "Docs"]
REQUIRED_FIELDS = ["When to use", "Tags"]
ENTRY_FIELDS = {"When to use", "Recent changes", "Current version", "Key APIs", "Watch out", "Related"}
TAG_KEYS = ["family", "step", "noise", "runs", "status", "updated"]
REQUIRED_TAGS = ["family", "step", "runs", "status"]
SINGLE_TAGS = ["status", "updated"]
REQUIRED_FRONT_MATTER = ["title", "tool_type", "owner", "last_verified"]
TAXONOMY_KEYS = {"family": ["id", "title", "short", "description"], "step": ["id", "title", "short", "description"],
                 "noise": ["id", "title", "short", "description"], "runs": ["id", "description"],
                 "status": ["id", "title", "description"], "task": ["text", "links"]}  # a task may also name a group
PACKAGE_HEADER = "| Package | Version checked | Notes |"
MONTHS = ["January", "February", "March", "April", "May", "June", "July", "August", "September", "October", "November", "December"]
MIN_SENTENCE = 60
HEADING = re.compile(r"^(#{1,6})[ \t]+(.+?)(?:[ \t]+#+)?[ \t]*$")
FIELD = re.compile(r"^(?:- )?\*\*([^*]+?):\*\*", re.M)
TAGS_LINE = re.compile(r"^- \*\*Tags:\*\* *(.*)$", re.M)
TAGS_NEAR_MISS = re.compile(r"^\s*(?:[-*+]\s+)?\*\*\s*tags\s*:?\s*\*\*", re.M | re.I)
CODE_SPAN = re.compile(r"(`+)(.+?)\1")
LINK = re.compile(r"\[[^\]]*\]\(([^)\s]+)(?:\s+\"[^\"]*\")?\)")


def read(rel):
    return (ROOT / rel).read_text(encoding="utf-8-sig")


# ---------------------------------------------------------------- parsing

def strip_code(text):
    """Return the text with fenced code blocks blanked out (line count kept), so headings and links inside code are ignored."""
    out, fence = [], None
    for line in text.split("\n"):
        m = re.match(r"^\s*(```+|~~~+)", line)
        if fence is None and m:
            fence = m.group(1)
            out.append("")
        elif fence is not None:
            if line.strip().startswith(fence):
                fence = None
            out.append("")
        else:
            out.append(line)
    return "\n".join(out)


def strip_generated(text):
    return re.sub(r"<!-- BEGIN GENERATED: [\w-]+ -->.*?<!-- END GENERATED: [\w-]+ -->", "", text, flags=re.S)


def strip_emphasis(md):
    return re.sub(r"\*\*|(?<!\w)__|__(?!\w)", "", md)


def plain_text(md):
    """Markdown inline text as GitHub renders it: links become their text, code spans keep their content, emphasis marks go."""
    md = re.sub(r"!?\[([^\]]*)\]\([^)]*\)", r"\1", md)
    out, pos = [], 0
    for m in CODE_SPAN.finditer(md):
        out += [strip_emphasis(md[pos:m.start()]), m.group(2)]
        pos = m.end()
    return "".join(out) + strip_emphasis(md[pos:])


def gh_slug(heading):
    """GitHub's heading anchor: lower case; keep letters, marks, digits, '_', '-' and spaces; spaces become hyphens."""
    text = plain_text(heading).strip().lower()
    keep = lambda ch: ch in "- " or unicodedata.category(ch)[0] in "LM" or unicodedata.category(ch) in ("Nd", "Nl", "Pc")
    return "".join(filter(keep, text)).replace(" ", "-")


def split_front_matter(text):
    if not text.startswith("---\n"):
        return {}, text
    end = text.find("\n---\n", 4)
    if end < 0:
        return {}, text
    meta = {}
    for line in text[4:end].split("\n"):
        if not line.strip() or line.lstrip().startswith("#"):
            continue
        key, _, value = line.partition(":")
        meta[key.strip()] = value.strip().strip('"').strip("'")
    return meta, text[end + 5:]


class Doc:
    def __init__(self, rel):
        self.rel = rel
        self.text = read(rel)
        self.meta, self.body = split_front_matter(self.text)
        front = self.text[:len(self.text) - len(self.body)]
        self.code_free = "\n" * front.count("\n") + strip_code(self.body)  # front matter blanked, line numbers kept
        self.problems = []  # (anchor, message), reported by check()
        self.headings = []  # (level, title, slug, line index)
        counts = defaultdict(int)
        for i, line in enumerate(self.code_free.split("\n")):
            if m := HEADING.match(line):
                title = m.group(2).strip()
                base = gh_slug(title)
                slug = base if counts[base] == 0 else f"{base}-{counts[base]}"
                counts[base] += 1
                self.headings.append((len(m.group(1)), title, slug, i))
        self.anchors = {h[2] for h in self.headings}
        self.anchors |= set(re.findall(r'<a\s+(?:id|name)="([^"]+)"', self.code_free))
        self.title = self.meta.get("title") or next((t for lvl, t, _, _ in self.headings if lvl == 1), rel)
        self.short = self.meta.get("short_title", self.title)
        self.entries = self._entries()
        self.packages = self._packages()

    def sections(self, level=2):
        """(title, slug, raw text, code-free text) for each heading of the given level; the text runs to the next heading of that level or higher."""
        lines, raw = self.code_free.split("\n"), self.text.split("\n")
        out = []
        for n, (lvl, title, slug, i) in enumerate(self.headings):
            if lvl == level:
                end = next((j for l2, _, _, j in self.headings[n + 1:] if l2 <= level), len(lines))
                out.append((title, slug, "\n".join(raw[i + 1:end]), "\n".join(lines[i + 1:end])))
        return out

    def _entries(self):
        entries = []
        for title, slug, raw, code_free in self.sections(2):
            fields = FIELD.findall(code_free)
            m = TAGS_LINE.search(code_free)
            if not m:
                if TAGS_NEAR_MISS.search(code_free):
                    self.problems.append((slug, "has a malformed Tags line; write it as '- **Tags:** family: ...; step: ...'"))
                elif self.rel.startswith("reference/") or ENTRY_FIELDS & set(fields):
                    self.problems.append((slug, "has no '- **Tags:**' line, so it is left out of every view"))
                continue
            tags = {}
            for part in m.group(1).split(";"):
                if part.strip():
                    key, _, values = part.partition(":")
                    tags.setdefault(key.strip(), []).extend(v.strip() for v in values.split(",") if v.strip())
            first = FIELD.search(code_free)
            if not code_free[:first.start()].strip():
                self.problems.append((slug, "has no description before its first field"))
            entries.append({"doc": self, "title": title, "slug": slug, "tags": tags, "fields": fields, "text": raw})
        return entries

    def _packages(self):
        lines = self.body.split("\n")
        for i, line in enumerate(lines):
            if line.startswith("## "):
                return []
            if line.strip() == PACKAGE_HEADER:
                rows = []
                for row in lines[i + 2:]:
                    if not row.startswith("|"):
                        break
                    cells = [c.strip() for c in re.split(r"(?<!\\)\|", row.strip().strip("|"))]
                    if len(cells) > 3:
                        self.problems.append(("", f"package row has {len(cells)} cells instead of 3; escape a '|' inside a cell as '\\|': {row.strip()[:60]}"))
                    rows.append((cells + ["", ""])[:3])
                return rows
        return []


def load_docs(order=()):
    """Documents in the order taxonomy.toml gives, then any others alphabetically."""
    rels = []
    for d in DOC_DIRS:
        rels += sorted(p.relative_to(ROOT).as_posix() for p in (ROOT / d).glob("*.md"))
    rank = {rel: i for i, rel in enumerate(order)}
    rels.sort(key=lambda r: (rank.get(r, len(rank)), r))
    return [Doc(r) for r in rels]


# ---------------------------------------------------------------- links

def rel_link(from_doc, target):
    """Relative link from one document (path from ROOT) to a target 'path#anchor' (path from ROOT)."""
    path, _, anchor = target.partition("#")
    if path == from_doc:
        return f"#{anchor}" if anchor else posixpath.basename(path)
    rel = posixpath.relpath(path, posixpath.dirname(from_doc) or ".")
    return f"{rel}#{anchor}" if anchor else rel


def rebase_links(md, src_doc, dst_doc):
    """Rewrite the relative links in a snippet taken from src_doc so that they work from dst_doc."""
    def fix(m):
        text, href = m.group(1), m.group(2)
        if re.match(r"^[a-z]+:", href):
            return m.group(0)
        path, _, anchor = href.partition("#")
        target = src_doc if not path else posixpath.normpath(posixpath.join(posixpath.dirname(src_doc), path))
        return f"[{text}]({rel_link(dst_doc, target + ('#' + anchor if anchor else ''))})"
    return re.sub(r"\[([^\]]+)\]\(([^)\s]+)\)", fix, md)


def view_title(title):
    """Heading text without a trailing package name in parentheses, for compact index links."""
    text = plain_text(title)
    short = re.sub(r" \([^()]*\)$", "", text)
    return short if short else text


def entry_link(entry, from_doc, full=False):
    text = plain_text(entry["title"]) if full else view_title(entry["title"])
    return f"[{text}]({rel_link(from_doc, entry['doc'].rel + '#' + entry['slug'])})"


# ---------------------------------------------------------------- checks

class Report:
    def __init__(self):
        self.errors = []

    def error(self, where, msg):
        self.errors.append(f"{where}: {msg}")


def check(docs, tax, report):
    vocab = {
        "family": {f["id"] for f in tax["family"]},
        "step": {s["id"] for s in tax["step"]},
        "noise": {n["id"] for n in tax["noise"]},
        "runs": {r["id"] for r in tax["runs"]},
        "status": {s["id"] for s in tax["status"]},
    }
    newest = max((d.meta.get("last_verified", "") for d in docs), default="")[:7]
    if newest > tax["as_of"]:
        report.error("tools/taxonomy.toml", f"as_of {tax['as_of']} is older than the newest last_verified ({newest}); update as_of")
    for doc in docs:
        for key in REQUIRED_FRONT_MATTER:
            if not doc.meta.get(key):
                report.error(doc.rel, f"front matter lacks '{key}'")
        if doc.meta.get("last_verified") and not re.match(r"^\d{4}-\d{2}-\d{2}$", doc.meta["last_verified"]):
            report.error(doc.rel, "last_verified must be YYYY-MM-DD")
        for anchor, msg in doc.problems:
            report.error(f"{doc.rel}#{anchor}" if anchor else doc.rel, msg)
        for e in doc.entries:
            where = f"{doc.rel}#{e['slug']}"
            for key in e["tags"]:
                if key not in TAG_KEYS:
                    report.error(where, f"unknown tag '{key}'")
            for key in REQUIRED_TAGS:
                if not e["tags"].get(key):
                    report.error(where, f"Tags lacks '{key}'")
            for key in SINGLE_TAGS:
                if len(e["tags"].get(key, [])) > 1:
                    report.error(where, f"'{key}' takes one value")
            for key, allowed in vocab.items():
                for v in e["tags"].get(key, []):
                    if v not in allowed:
                        report.error(where, f"{key} value '{v}' is not in taxonomy.toml")
            for v in e["tags"].get("updated", []):
                if not re.match(r"^\d{4}-(0[1-9]|1[0-2])$", v):
                    report.error(where, f"updated '{v}' must be YYYY-MM")
                elif v > tax["as_of"]:
                    report.error(where, f"updated '{v}' is later than as_of ({tax['as_of']})")
            if e["tags"].get("updated") and not {"Recent changes", "Current version"} & set(e["fields"]) \
                    and not month_named(e["text"], e["tags"]["updated"][0]):
                report.error(where, "has an 'updated' month, but neither a Recent changes or Current version field nor its text names that month")
            if "Recent changes" in e["fields"] and not e["tags"].get("updated"):
                report.error(where, "has Recent changes but no 'updated' month, so New and changing leaves it out")
            for f in REQUIRED_FIELDS:
                if f not in e["fields"]:
                    report.error(where, f"lacks the '{f}' field")
            known = [f for f in e["fields"] if f in FIELD_ORDER]
            for f in e["fields"]:
                if f not in FIELD_ORDER:
                    report.error(where, f"unknown field '{f}'")
            if known != sorted(known, key=FIELD_ORDER.index) or len(set(known)) != len(known):
                report.error(where, f"fields out of order: {', '.join(known)}")
    check_links(docs, report)
    check_tasks(docs, tax, report)
    check_repeats(docs, report)


def month_named(text, ym):
    """True if the text names the month of 'YYYY-MM', for example 'September 2026'."""
    y, m = ym.split("-")
    return f"{MONTHS[int(m) - 1]} {y}" in plain_text(text)


def check_taxonomy(tax, report):
    """Problems that would stop the views from being built."""
    if not re.fullmatch(r"\d{4}-(0[1-9]|1[0-2])", str(tax.get("as_of", ""))):
        report.error("tools/taxonomy.toml", "as_of must be YYYY-MM")
    for table, keys in TAXONOMY_KEYS.items():
        for n, row in enumerate(tax.get(table, []), 1):
            missing = [k for k in keys if not row.get(k)]
            if missing:
                report.error("tools/taxonomy.toml", f"[[{table}]] number {n} ({row.get('id', row.get('text', '?'))}) lacks {', '.join(missing)}")
        for value, count in Counter(row.get("id") for row in tax.get(table, []) if "id" in row).items():
            if count > 1:
                report.error("tools/taxonomy.toml", f"[[{table}]] id '{value}' is defined {count} times")


def check_order(docs, order, report):
    rels = {d.rel for d in docs}
    for rel in order:
        if rel not in rels:
            report.error("tools/taxonomy.toml", f"documents.order lists '{rel}', which does not exist")
    for rel in sorted(rels - set(order)):
        report.error(rel, "is missing from documents.order in tools/taxonomy.toml")


def all_md_files():
    return [p.relative_to(ROOT).as_posix() for p in ROOT.rglob("*.md")]


def check_links(docs, report):
    by_rel = {d.rel: d for d in docs}
    files = all_md_files()
    anchors = {rel: (by_rel.get(rel) or Doc(rel)).anchors for rel in files}
    for rel in files:
        for n, line in enumerate(strip_code(read(rel)).split("\n"), 1):
            line = CODE_SPAN.sub(" ", line)  # a [x](y) inside inline code is not a link
            for m in LINK.finditer(line):
                href = m.group(1)
                if re.match(r"^[A-Za-z][\w+.-]*:", href):
                    continue
                path, _, anchor = href.partition("#")
                target = rel if not path else posixpath.normpath(posixpath.join(posixpath.dirname(rel), path))
                if not (ROOT / target).exists():
                    report.error(f"{rel}:{n}", f"link to missing file '{href}'")
                elif anchor and target.endswith(".md") and anchor not in anchors.get(target, set()):
                    report.error(f"{rel}:{n}", f"link to missing anchor '{href}'")


def resolve_target(docs, target):
    """Find the heading a task link points to: returns (title, doc_rel, slug) or None."""
    path, _, anchor = target.partition("#")
    for doc in docs:
        if doc.rel == path:
            if not anchor:
                return doc.title, doc.rel, ""
            for lvl, title, slug, _ in doc.headings:
                if slug == anchor:
                    return title, doc.rel, slug
    return None


def check_tasks(docs, tax, report):
    for task in tax.get("task", []):
        for target in task["links"]:
            if not resolve_target(docs, target):
                report.error("tools/taxonomy.toml", f"task '{task['text']}' links to '{target}', which does not exist")


def sentences(text):
    text = strip_generated(strip_code(text))
    _, text = split_front_matter(text)
    out = []
    for line in text.split("\n"):
        s = line.strip()
        if not s or s.startswith(("#", "|", "<!--", "- **Tags:**")) or re.fullmatch(r"[-*] \[[^\]]+\]\([^)]+\)\.?", s):
            continue
        s = re.sub(r"^([-*]|\d+\.) +", "", s)
        s = re.sub(r"^\*\*[^*]+:\*\* *", "", s)
        for part in re.split(r"(?<=[.!?])\s+(?=[A-Z`(\[*_])", s):  # split before plain_text removes the backticks
            norm = re.sub(r"\s+", " ", plain_text(part).strip().lower())
            if len(norm) >= MIN_SENTENCE:
                out.append(norm)
    return out


def check_repeats(docs, report):
    seen = {}
    for rel in [d.rel for d in docs] + [README, "CONTRIBUTING.md"]:
        if not (ROOT / rel).exists():
            continue
        for s in sentences(read(rel)):
            if s in seen and seen[s] != rel:
                report.error(rel, f"repeats a sentence from {seen[s]}: \"{s[:90]}...\"")
            elif s in seen:
                report.error(rel, f"repeats a sentence within the document: \"{s[:90]}...\"")
            else:
                seen[s] = rel


# ---------------------------------------------------------------- views

def links_by_doc(entries, from_doc):
    """Bullets 'Doc: entry · entry', grouped by document in document order."""
    groups = defaultdict(list)
    order = []
    for e in entries:
        if e["doc"].rel not in groups:
            order.append(e["doc"])
        groups[e["doc"].rel].append(e)
    lines = []
    for doc in order:
        items = " · ".join(entry_link(e, from_doc) for e in groups[doc.rel])
        lines.append(f"- **{doc.short}:** {items}")
    return lines


def runs_note(entry):
    """A short marker for entries that need a paid plan, derived from the runs tag."""
    return " (Premium plans)" if entry and "premium" in entry["tags"].get("runs", []) else ""


def render_views(docs, tax):
    entries = [e for d in docs for e in d.entries]
    by_target = {f"{e['doc'].rel}#{e['slug']}": e for e in entries}
    steps = tax["step"]
    sec = {}

    # I want to...
    out = ["## I want to…", "", "Tasks grouped by what you are doing. Each link names the entry or section that answers it.", ""]
    groups = []
    for task in tax.get("task", []):
        g = task.get("group", "Other tasks")
        if g not in groups:
            groups.append(g)
    for g in groups:
        out += [f"### {g}", "", "| Task | Go to |", "|---|---|"]
        for task in [t for t in tax.get("task", []) if t.get("group", "Other tasks") == g]:
            targets = []
            for target in task["links"]:
                found = resolve_target(docs, target)
                title = plain_text(found[0]) if found else target
                targets.append(f"[{view_title(title)}]({rel_link(README, target)}){runs_note(by_target.get(target))}")
            out.append(f"| {task['text']} | {' · '.join(targets)} |")
        out.append("")
    sec["tasks"] = out

    # By algorithm family
    families = [f for f in tax["family"] if f["id"] != "general"]
    out = ["## By algorithm family", "",
           "Each family lists the tools specific to it, grouped by workflow step. Tools that every family uses, such as the transpiler, "
           "the primitives and QPU budgeting, are listed under [By workflow step](#by-workflow-step).", "",
           "Families: " + " · ".join(f"[{f['title']}](#{gh_slug(f['title'])})" for f in families), ""]
    for f in families:
        members = [e for e in entries if f["id"] in e["tags"].get("family", [])]
        out += [f"### {f['title']}", "", f["description"], ""]
        if not members:
            out += ["No entries yet.", ""]
            continue
        for s in steps:
            in_step = [e for e in members if s["id"] in e["tags"].get("step", [])]
            if in_step:
                out.append(f"- **{s['short']}:** " + " · ".join(entry_link(e, README) for e in in_step))
        out.append("")
    sec["families"] = out

    # By workflow step
    out = ["## By workflow step", "",
           "The four steps of a Qiskit pattern (map, optimize, execute, post-process), plus scaling out, and developing and debugging.", ""]
    for s in steps:
        members = [e for e in entries if s["id"] in e["tags"].get("step", [])]
        out += [f"### {s['title']}", "", s["description"], ""] + links_by_doc(members, README) + [""]
    sec["steps"] = out

    # By noise-management layer
    out = ["## By noise-management layer", "",
           "Grouped as in IBM's [overview of noise management techniques](https://quantum.cloud.ibm.com/docs/en/guides/error-mitigation-overview), "
           "with error detection and noise characterization added.", ""]
    for n in tax["noise"]:
        members = [e for e in entries if n["id"] in e["tags"].get("noise", [])]
        out += [f"### {n['title']}", "", n["description"], ""] + links_by_doc(members, README) + [""]
    sec["noise"] = out

    # New and changing
    as_of = tax["as_of"]
    y, m = map(int, as_of.split("-"))
    first = (y * 12 + m - 1) - (tax.get("new_window_months", 6) - 1)
    start = f"{first // 12:04d}-{first % 12 + 1:02d}"
    recent = [e for e in entries if e["tags"].get("updated") and e["tags"]["updated"][0] >= start]
    out = ["## New and changing", "",
           f"Entries whose newest listed change is from {MONTHS[int(start[5:]) - 1]} {start[:4]} or later "
           f"(the cheat sheet was checked in {MONTHS[m - 1]} {y}). Each entry says what changed and when.", ""]
    by_month = defaultdict(list)
    for e in recent:
        by_month[e["tags"]["updated"][0]].append(e)
    for month in sorted(by_month, reverse=True):
        out.append(f"- **{MONTHS[int(month[5:]) - 1]} {month[:4]}**")
        out += [f"  - {entry_link(e, README)} ({e['doc'].short})" for e in by_month[month]]
    out += ["", "### Not yet stable", "",
            "Entries whose status is not stable. Check their interfaces against the installed version before you rely on them.", ""]
    for st in tax["status"]:
        if st["id"] == "stable":
            continue
        members = [e for e in entries if st["id"] in e["tags"].get("status", [])]
        if members:
            out.append(f"- **{st['title']}:** " + " · ".join(entry_link(e, README) for e in members))
    out.append("")
    sec["new"] = out

    # By tool type
    out = ["## By tool type", "",
           "One reference document per tool type, each owned by one team; the guides hold what spans tools. "
           "[Document owners](#document-owners) lists who keeps each document current.", ""]
    for group, prefix in (("Reference documents", "reference/"), ("Guides", "guides/")):
        out += [f"**{group}**", ""]
        for doc in [d for d in docs if d.rel.startswith(prefix)]:
            if prefix == "reference/":
                items = " · ".join(entry_link(e, README) for e in doc.entries)
            else:
                items = " · ".join(f"[{view_title(t)}]({rel_link(README, doc.rel + '#' + s)})" for t, s, _, _ in doc.sections(2))
            out.append(f"- **[{doc.title}]({doc.rel}):** {items}")
        out.append("")
    sec["tools"] = out

    # All entries
    out = ["## All entries", "",
           "Every entry with its tags. Tag values are defined in [taxonomy.toml](tools/taxonomy.toml).", "",
           "Runs on: " + "; ".join(f"**{r['id']}** {r['description'][0].lower()}{r['description'][1:]}" for r in tax["runs"]) + ".", "",
           "| Entry | Document | Families | Step | Noise | Runs on | Status | Updated |", "|---|---|---|---|---|---|---|---|"]
    fam_short = {f["id"]: f["short"] for f in tax["family"]}
    step_short = {s["id"]: s["short"] for s in steps}
    noise_short = {n["id"]: n["short"] for n in tax["noise"]}
    for e in entries:
        tg = e["tags"]
        out.append("| " + " | ".join([
            entry_link(e, README, full=True), e["doc"].short,
            ", ".join(fam_short.get(v, v) for v in tg.get("family", [])),
            ", ".join(step_short.get(v, v) for v in tg.get("step", [])),
            ", ".join(noise_short.get(v, v) for v in tg.get("noise", [])),
            ", ".join(tg.get("runs", [])), ", ".join(tg.get("status", [])), ", ".join(tg.get("updated", [])),
        ]) + " |")
    out.append("")
    sec["all"] = out

    # Document owners
    out = ["## Document owners", "",
           "Proposed owners until the teams confirm them. [Maintaining this cheat sheet](CONTRIBUTING.md) explains how to update a document.", "",
           "| Document | Owner | Last verified |", "|---|---|---|"]
    for doc in docs:
        out.append(f"| [{doc.title}]({doc.rel}) | {doc.meta.get('owner', '')} | {doc.meta.get('last_verified', '')} |")
    sec["owners"] = out

    order = ["tasks", "families", "all", "steps", "noise", "new", "tools", "owners"]  # task index and All entries come early, within an agent's first read
    return "\n".join(line for name in order for line in sec[name]).rstrip() + "\n"


def render_versions(docs):
    out = ["| Package | Version checked | Notes | Document |", "|---|---|---|---|"]
    for doc in docs:
        for package, version, notes in doc.packages:
            notes = rebase_links(notes, doc.rel, VERSIONS_DOC)
            out.append(f"| {package} | {version} | {notes} | [{doc.short}]({rel_link(VERSIONS_DOC, doc.rel)}) |")
    return "\n".join(out) + "\n"


def replace_block(text, name, content):
    pattern = re.compile(rf"(<!-- BEGIN GENERATED: {name} -->\n).*?(<!-- END GENERATED: {name} -->)", re.S)
    if not pattern.search(text):
        raise SystemExit(f"generated block '{name}' not found")
    return pattern.sub(lambda m: m.group(1) + content + m.group(2), text)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--check", action="store_true", help="change nothing; fail on any error or stale generated section")
    args = ap.parse_args()
    tax = tomllib.loads(TAXONOMY.read_text(encoding="utf-8"))
    order = tax.get("documents", {}).get("order", [])
    docs = load_docs(order)
    report = Report()
    check_taxonomy(tax, report)
    if report.errors:  # the views cannot be built from a broken taxonomy
        print(f"{len(report.errors)} problem(s):\n  " + "\n  ".join(report.errors))
        sys.exit(1)

    outputs = {}
    readme = (ROOT / README).read_text(encoding="utf-8")
    outputs[README] = replace_block(readme, "views", render_views(docs, tax))
    vdoc = (ROOT / VERSIONS_DOC).read_text(encoding="utf-8")
    outputs[VERSIONS_DOC] = replace_block(vdoc, "versions", render_versions(docs))

    stale = [rel for rel, text in outputs.items() if (ROOT / rel).read_text(encoding="utf-8") != text]
    if args.check:
        for rel in stale:
            diff = difflib.unified_diff((ROOT / rel).read_text(encoding="utf-8").splitlines(), outputs[rel].splitlines(), n=0, lineterm="")
            report.error(rel, "generated section is out of date; run python3 tools/build_cheatsheet.py\n      "
                         + "\n      ".join(list(diff)[2:10]))
    else:
        for rel in stale:
            (ROOT / rel).write_text(outputs[rel], encoding="utf-8")
        docs = load_docs(order)
    check_order(docs, order, report)
    check(docs, tax, report)

    n_entries = sum(len(d.entries) for d in docs)
    if report.errors:
        print(f"{len(report.errors)} problem(s):")
        for e in report.errors:
            print("  " + e)
        sys.exit(1)
    action = "checked" if args.check else ("rebuilt " + ", ".join(stale) if stale else "nothing to rebuild")
    print(f"OK: {len(docs)} documents, {n_entries} entries, {len(tax.get('task', []))} tasks; {action}.")


if __name__ == "__main__":
    main()
