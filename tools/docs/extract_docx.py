"""Extract plain text from a .docx file using stdlib only (no python-docx required)."""
import argparse
import sys
import zipfile
import xml.etree.ElementTree as ET
from pathlib import Path

W = "{http://schemas.openxmlformats.org/wordprocessingml/2006/main}"


def extract_text(docx_path):
    with zipfile.ZipFile(docx_path) as z:
        with z.open("word/document.xml") as f:
            tree = ET.parse(f)
    lines = []
    for para in tree.getroot().iter(f"{W}p"):
        parts = [t.text for t in para.iter(f"{W}t") if t.text]
        lines.append("".join(parts).strip())
    return "\n".join(lines)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("input", help="Path to .docx file")
    p.add_argument("-o", "--output", help="Output .txt path (default: stdout)")
    args = p.parse_args()

    src = Path(args.input)
    if not src.exists():
        print(f"FAIL: cannot find {src}", file=sys.stderr)
        sys.exit(1)

    text = extract_text(src)
    if args.output:
        out = Path(args.output)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(text, encoding="utf-8")
        print(f"saved {out} ({len(text)} chars)")
    else:
        print(text)


if __name__ == "__main__":
    main()
