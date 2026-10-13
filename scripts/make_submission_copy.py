#!/usr/bin/env python3
"""Make the submission copy of the manuscript from the accepted-changes version.

    python scripts/make_submission_copy.py

Reads  paper/514_Manuscript_revised_clean.docx
Writes paper/514_Manuscript_submission_copy.docx

Changes (each asserted to match exactly once):
  * three wording fixes from the audit (docs/MANUSCRIPT_AUDIT.md, round 2 "smaller points" and the
    deployment-architecture wording);
  * all review comments removed (comment ranges, references and the comments parts);
  * media files no longer referenced by the document dropped.
Nothing else in the document is touched: tables, figures, references and styles are copied byte for byte.
"""
import re
import sys
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
SRC = ROOT / "paper" / "514_Manuscript_revised_clean.docx"
DST = ROOT / "paper" / "514_Manuscript_submission_copy.docx"

EDITS = [
    ("As a design recommendation (not evaluated here), from a clinical utility perspective,",
     "As a design recommendation (not evaluated here),"),
    ("SHAP-based explanations help clinicians see which variables drive each prediction, but they describe model behaviour, not causes.",
     "SHAP-based explanations can show which variables drive a prediction, but they describe model behaviour, not causes."),
    ("Applies normalization, imputation, and feature computation in real time.",
     "Applies normalization, imputation, and feature computation hourly."),
    # the stated cause of the k = 10 effect was untested; the normalised-sweep control shows it is sparse reporting
    ("but k = 10 raised false alarms from 0.61 to 2.69 per 100 region-days, because the pooled count then varies with the number of sites that pass suppression.",
     "but k = 10 raised false alarms from 0.61 to 2.69 per 100 region-days. With about eight admissions per site per day, "
     "k = 10 suppresses two thirds of site-days, so a region has no report on a third of days and on average only one of "
     "its three sites reporting; the regional series then rests on few, intermittent reports. Using the viral count per "
     "reporting site instead of the sum did not remove the effect (2.84)."),
]


def main():
    zin = zipfile.ZipFile(SRC)
    doc = zin.read("word/document.xml").decode("utf8")

    for old, new in EDITS:
        n = doc.count(old)
        assert n == 1, f"expected exactly one match for {old[:50]!r}, found {n}"
        doc = doc.replace(old, new)

    # comments: remove range markers and the reference runs
    doc = re.sub(r"<w:commentRangeStart[^>]*/>|<w:commentRangeEnd[^>]*/>", "", doc)
    doc = re.sub(r"<w:r(?:\s[^>]*)?>(?:(?!</w:r>).)*?<w:commentReference[^>]*/>(?:(?!</w:r>).)*?</w:r>", "", doc, flags=re.S)
    assert "commentReference" not in doc and "commentRange" not in doc

    # relationships and content types
    rels = zin.read("word/_rels/document.xml.rels").decode("utf8")
    rels = re.sub(r"<Relationship [^>]*Target=\"comments[^\"]*\"[^>]*/>", "", rels)
    used_media = set(re.findall(r"r:embed=\"([^\"]+)\"", doc)) | set(re.findall(r"r:id=\"([^\"]+)\"", doc))
    def keep_rel(m):
        rid = re.search(r'Id="([^"]+)"', m.group(0)).group(1)
        tgt = re.search(r'Target="([^"]+)"', m.group(0)).group(1)
        return m.group(0) if (not tgt.startswith("media/") or rid in used_media) else ""
    rels = re.sub(r"<Relationship [^>]*/>", keep_rel, rels)
    kept_media = {"word/" + t for t in re.findall(r'Target="(media/[^"]+)"', rels)}

    ct = zin.read("[Content_Types].xml").decode("utf8")
    ct = re.sub(r"<Override [^>]*PartName=\"/word/comments[^\"]*\"[^>]*/>", "", ct)

    dropped = []
    with zipfile.ZipFile(DST, "w", zipfile.ZIP_DEFLATED) as zout:
        for item in zin.infolist():
            name = item.filename
            if name.startswith("word/comments"):
                dropped.append(name)
                continue
            if name.startswith("word/media/") and name not in kept_media and not name.endswith("/"):
                dropped.append(name)
                continue
            data = zin.read(name)
            if name == "word/document.xml":
                data = doc.encode("utf8")
            elif name == "word/_rels/document.xml.rels":
                data = rels.encode("utf8")
            elif name == "[Content_Types].xml":
                data = ct.encode("utf8")
            zout.writestr(item, data)
    print(f"wrote {DST.relative_to(ROOT)} ({DST.stat().st_size / 1024:.0f} KB; source {SRC.stat().st_size / 1024:.0f} KB)")
    print("dropped parts:", ", ".join(sorted(dropped)))


if __name__ == "__main__":
    sys.exit(main())
