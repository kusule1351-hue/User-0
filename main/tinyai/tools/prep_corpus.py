"""Clean the Gutenberg texts in data/raw into one lowercase character stream (data/corpus.txt).

Vocabulary (36 symbols): a-z, space, and . , ; : ? ! ' -
"""
import glob, re, sys, os

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
raw_files = sorted(glob.glob(os.path.join(ROOT, "data", "raw", "pg*.txt")))
parts = []
for f in raw_files:
    t = open(f, encoding="utf-8-sig", errors="replace").read().replace("\r\n", "\n")
    a = re.search(r"\*\*\* ?START OF (THE|THIS) PROJECT GUTENBERG EBOOK.*?\*\*\*", t)
    b = re.search(r"\*\*\* ?END OF (THE|THIS) PROJECT GUTENBERG EBOOK", t)
    if not a or not b:
        sys.exit("markers missing in " + f)
    t = t[a.end():b.start()]
    t = re.sub(r"\[[^\]]*\]", " ", t)                         # footnote marks, editor notes
    keep = []
    for line in t.split("\n"):
        s = line.strip()
        if not s: keep.append(""); continue
        letters = [c for c in s if c.isalpha()]
        if not letters: continue                               # numbering, rules
        if len(s) < 60 and s == s.upper(): continue            # ALL-CAPS headings
        if re.fullmatch(r"[IVXLC]+\.?", s): continue           # roman numerals
        if re.match(r"(?i)(chapter|book|part|section|preface|footnotes?|contents|index|note)\b", s) and len(s) < 60: continue
        keep.append(s)
    t = " ".join(keep)
    t = t.lower()
    t = t.replace("’", "'").replace("‘", "'").replace("—", " - ").replace("–", "-").replace("_", "")
    t = re.sub(r"[^a-z .,;:?!'\-]", " ", t)
    t = re.sub(r"(?<![a-z])-+|-+(?![a-z])", " ", t)            # stray dashes
    t = re.sub(r" +", " ", t).strip()
    parts.append(t)
    print(f"{os.path.basename(f)}: {len(t):>9,} chars")
out = " ".join(parts)
open(os.path.join(ROOT, "data", "corpus.txt"), "w", encoding="ascii").write(out)
print(f"total {len(out):,} chars, {len(set(out))} symbols: {''.join(sorted(set(out)))!r}")
