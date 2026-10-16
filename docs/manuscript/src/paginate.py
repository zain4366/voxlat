"""Second pass: render with LibreOffice, read the page of every heading, store for the TOC."""
import json, re, subprocess, sys
from pathlib import Path
HERE = Path(__file__).parent
review = "--review" in sys.argv
name = "VoxLat_manuscript_review" if review else "VoxLat_manuscript_template_layout"
sys.path.insert(0, str(HERE)); import build  # noqa
subprocess.run(["soffice", "--headless", "--convert-to", "pdf", "--outdir", str(HERE / "render"), str(HERE / "out" / f"{name}.docx")], check=True, capture_output=True)
pdf = HERE / "render" / f"{name}.pdf"
n = int(re.search(r"Pages:\s+(\d+)", subprocess.run(["pdfinfo", str(pdf)], capture_output=True, text=True).stdout).group(1))
pages = [subprocess.run(["pdftotext", "-f", str(k), "-l", str(k), str(pdf), "-"], capture_output=True, text=True).stdout for k in range(1, n + 1)]
src = "\n\n".join((HERE / f).read_text(encoding="utf-8") for f in build.BODY)
blocks = build.parse_source(src); build.number_labels(blocks)
heads = ["NOMENCLATURE"] + [((b.get("num", "") + " " + b["text"]).strip() if b["numbered"] else b["text"]) if b["type"] == "h1" else b["num"] + " " + b["text"] for b in blocks if b["type"] in ("h1", "h2")] + ["REFERENCES"]
norm = lambda t: re.sub(r"\s+", " ", t).strip()
out, start = {}, 0
for h in heads:
    hn = norm(h)
    for k in range(start, n):
        lines = [norm(re.sub(r"^\s*\d{1,4}\s+(?=\S)", "", l)) for l in pages[k].split("\n")]
        joined = [lines[i] + " " + lines[i + 1] for i in range(len(lines) - 1)]
        if hn in lines or hn in joined or any(l == hn or l.endswith(" " + hn) for l in lines):
            out[h] = k + 1; start = k; break
    else:
        print("not found:", h)
(HERE / "out" / ("pages_review.json" if review else "pages_template.json")).write_text(json.dumps(out, indent=0))
print(len(out), "of", len(heads), "headings located;", n, "pages")
