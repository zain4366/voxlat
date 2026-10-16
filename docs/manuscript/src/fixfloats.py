"""Template layout: find full-width floats that LibreOffice pushes to the next page and give them a page break."""
import json, re, subprocess, sys
from pathlib import Path
HERE = Path(__file__).parent
sys.path.insert(0, str(HERE)); import build
pdf = HERE / "render" / "VoxLat_manuscript_template_layout.pdf"
n = int(re.search(r"Pages:\s+(\d+)", subprocess.run(["pdfinfo", str(pdf)], capture_output=True, text=True).stdout).group(1))
pages = [re.sub(r"\s+", " ", subprocess.run(["pdftotext", "-f", str(k), "-l", str(k), str(pdf), "-"], capture_output=True, text=True).stdout) for k in range(1, n + 1)]
src = "\n\n".join((HERE / f).read_text(encoding="utf-8") for f in build.BODY)
blocks = build.parse_source(src); build.number_labels(blocks)
bf = HERE / "out" / "breaks_template.json"
breaks = set(json.loads(bf.read_text())) if bf.exists() else set()
prev = None
added = []
for b in blocks:
    if b["type"] == "para":
        prev = b["text"]
    if b["type"] in ("fig", "table") and b["width"] == "full" and prev and b["label"] not in breaks:
        words = re.sub(r"\$[^$]*\$|\[@[^\]]*\]|@\w+:\w+|[*`]", " ", prev).split()
        tail = " ".join(words[-4:])
        tp = next((k for k, t in enumerate(pages) if tail in t), None)
        key = ("Fig. %d. " if b["type"] == "fig" else "Table %d. ") % b["num"]
        cp = next((k for k, t in enumerate(pages) if key in t and (tp is None or k >= tp)), None)
        if tp is not None and cp is not None and cp > tp:
            breaks.add(b["label"]); added.append((b["label"], tp + 1, cp + 1)); break   # one at a time: layout shifts
    if b["type"] in ("fig", "table"):
        prev = None
bf.write_text(json.dumps(sorted(breaks)))
print("added", added, "total", sorted(breaks))
