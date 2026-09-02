# -*- coding: utf-8 -*-
"""Export the built deck to PDF with the installed PowerPoint (the portal wants PDF).

    python topdf.py MuleShield_AI_SIH26184.pptx

PowerPoint's PDF exporter drops some hyperlink annotations even though the
hyperlinks are present and correctly related in the .pptx -- two of the eight
references came out as plain blue text with nothing to click. The links are fine
inside PowerPoint; only the export loses them. So after exporting we read the
PDF back, find every URL that is rendered as visible text, and add a link
annotation over any that has none.
"""

import re
import sys
import pathlib

import pymupdf
import win32com.client

URL_RE = re.compile(r"https?://[^\s,;)\]]+")


def heal_links(pdf_path: pathlib.Path) -> tuple[int, int]:
    """Give every URL drawn on the page a clickable rectangle. Returns (total, added)."""
    doc = pymupdf.open(pdf_path)
    added = 0
    total = 0
    for page in doc:
        covered = [l["from"] for l in page.get_links() if l.get("uri")]
        for url in dict.fromkeys(URL_RE.findall(page.get_text())):
            rects = page.search_for(url)
            if not rects:
                continue
            total += 1
            # A URL can be split across several rects when PowerPoint breaks the
            # run (at a hyphen, say); one covered rect means the link is live.
            if any(r.intersects(c) for r in rects for c in covered):
                continue
            for r in rects:
                page.insert_link({"kind": pymupdf.LINK_URI, "from": r, "uri": url})
            added += 1
    if added:
        doc.saveIncr()
    doc.close()
    return total, added


def main() -> None:
    src = pathlib.Path(sys.argv[1]).resolve()
    dst = src.with_suffix(".pdf")

    app = win32com.client.Dispatch("PowerPoint.Application")
    pres = app.Presentations.Open(str(src), WithWindow=False)
    pres.SaveAs(str(dst), 32)          # 32 = ppSaveAsPDF
    pres.Close()
    app.Quit()

    total, added = heal_links(dst)
    note = f", repaired {added} dropped by the exporter" if added else ""
    print(f"  {dst.name}  ({dst.stat().st_size // 1024} KB, {total} URLs clickable{note})")


if __name__ == "__main__":
    main()
