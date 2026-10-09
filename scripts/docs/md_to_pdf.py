import os
import sys
from datetime import datetime
from fpdf import FPDF


class MD2PDF(FPDF):
    def header(self):
        self.set_font("Helvetica", "B", 14)
        self.cell(0, 10, "Sepsis Digital Twin - Project Overview", ln=1, align="C")
        self.set_draw_color(200, 200, 200)
        self.line(10, self.get_y(), 200, self.get_y())
        self.ln(4)

    def footer(self):
        self.set_y(-15)
        self.set_font("Helvetica", size=8)
        self.set_text_color(120, 120, 120)
        self.cell(0, 8, f"Generated {datetime.now().strftime('%Y-%m-%d %H:%M')}", align="L")
        self.cell(0, 8, f"Page {self.page_no()}", align="R")


def render_markdown(pdf: MD2PDF, path: str):
    with open(path, "r", encoding="utf-8") as f:
        lines = f.read().splitlines()

    pdf.set_auto_page_break(auto=True, margin=15)
    pdf.add_page()
    pdf.set_font("Helvetica", size=10)

    in_code = False
    for raw in lines:
        line = raw.rstrip()
        if line.strip().startswith("```"):
            in_code = not in_code
            if in_code:
                pdf.set_font("Courier", size=9)
            else:
                pdf.set_font("Helvetica", size=10)
            continue

        if in_code:
            pdf.multi_cell(190, 5, line)
            continue

        # Headings
        if line.startswith("### "):
            pdf.set_font("Helvetica", "B", 11)
            pdf.multi_cell(190, 6, line[4:])
            pdf.set_font("Helvetica", size=10)
            continue
        if line.startswith("## "):
            pdf.set_font("Helvetica", "B", 12)
            pdf.ln(2)
            pdf.multi_cell(190, 7, line[3:])
            pdf.ln(1)
            pdf.set_font("Helvetica", size=10)
            continue
        if line.startswith("# "):
            pdf.set_font("Helvetica", "B", 14)
            pdf.ln(2)
            pdf.multi_cell(190, 8, line[2:])
            pdf.ln(2)
            pdf.set_font("Helvetica", size=10)
            continue

        # Bullets
        if line.strip().startswith("- ") or line.strip().startswith("* "):
            content = line.strip()[2:]
            pdf.multi_cell(190, 5, f"- {content}")
            continue

        # Normal text or blank
        if line.strip() == "":
            pdf.ln(2)
        else:
            pdf.multi_cell(190, 5, line)


def main():
    root = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
    md_path = os.path.join(root, "outputs", "docs", "Project_Overview.md")
    out_path = os.path.join(root, "outputs", "docs", "Project_Overview.pdf")
    if not os.path.exists(md_path):
        print(f"Markdown not found: {md_path}")
        sys.exit(1)
    pdf = MD2PDF()
    render_markdown(pdf, md_path)
    pdf.output(out_path)
    print(f"PDF written to: {out_path}")


if __name__ == "__main__":
    main()


