#!/usr/bin/env python3
"""
Convert markdown file to PDF
"""
import sys
import os
from pathlib import Path

try:
    import markdown
    from weasyprint import HTML, CSS
    from weasyprint.text.fonts import FontConfiguration
except ImportError:
    print("Installing required packages...")
    os.system("pip3 install markdown weasyprint --quiet")
    import markdown
    from weasyprint import HTML, CSS
    from weasyprint.text.fonts import FontConfiguration

def markdown_to_pdf(md_file, pdf_file=None):
    """Convert markdown file to PDF"""
    if pdf_file is None:
        pdf_file = md_file.replace('.md', '.pdf')
    
    # Read markdown file
    with open(md_file, 'r', encoding='utf-8') as f:
        md_content = f.read()
    
    # Convert markdown to HTML
    html_content = markdown.markdown(md_content, extensions=['extra', 'codehilite'])
    
    # Add CSS styling
    html_with_style = f"""
    <!DOCTYPE html>
    <html>
    <head>
        <meta charset="UTF-8">
        <style>
            @page {{
                size: A4;
                margin: 2cm;
            }}
            body {{
                font-family: 'Helvetica', 'Arial', sans-serif;
                font-size: 11pt;
                line-height: 1.6;
                color: #333;
            }}
            h1 {{
                color: #2c3e50;
                border-bottom: 3px solid #3498db;
                padding-bottom: 10px;
                margin-top: 30px;
            }}
            h2 {{
                color: #34495e;
                border-bottom: 2px solid #95a5a6;
                padding-bottom: 5px;
                margin-top: 25px;
            }}
            h3 {{
                color: #555;
                margin-top: 20px;
            }}
            code {{
                background-color: #f4f4f4;
                padding: 2px 6px;
                border-radius: 3px;
                font-family: 'Courier New', monospace;
                font-size: 10pt;
            }}
            pre {{
                background-color: #f4f4f4;
                padding: 15px;
                border-radius: 5px;
                overflow-x: auto;
                border-left: 4px solid #3498db;
            }}
            blockquote {{
                border-left: 4px solid #95a5a6;
                padding-left: 15px;
                margin-left: 0;
                color: #666;
                font-style: italic;
            }}
            table {{
                border-collapse: collapse;
                width: 100%;
                margin: 15px 0;
            }}
            th, td {{
                border: 1px solid #ddd;
                padding: 8px;
                text-align: left;
            }}
            th {{
                background-color: #3498db;
                color: white;
            }}
            tr:nth-child(even) {{
                background-color: #f2f2f2;
            }}
            ul, ol {{
                margin: 10px 0;
                padding-left: 30px;
            }}
            li {{
                margin: 5px 0;
            }}
            hr {{
                border: none;
                border-top: 2px solid #ecf0f1;
                margin: 30px 0;
            }}
            strong {{
                color: #2c3e50;
            }}
        </style>
    </head>
    <body>
        {html_content}
    </body>
    </html>
    """
    
    # Convert HTML to PDF
    try:
        HTML(string=html_with_style).write_pdf(pdf_file)
        print(f"✅ Successfully converted {md_file} to {pdf_file}")
        return pdf_file
    except Exception as e:
        print(f"❌ Error converting to PDF: {e}")
        print("\nTrying alternative method...")
        # Alternative: use markdown2pdf if available
        try:
            import markdown2pdf
            markdown2pdf.convert(md_file, pdf_file)
            print(f"✅ Successfully converted using markdown2pdf")
            return pdf_file
        except:
            print("❌ Both methods failed. Please install weasyprint:")
            print("   pip3 install weasyprint")
            return None

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python3 convert_md_to_pdf.py <markdown_file.md> [output.pdf]")
        sys.exit(1)
    
    md_file = sys.argv[1]
    pdf_file = sys.argv[2] if len(sys.argv) > 2 else None
    
    if not os.path.exists(md_file):
        print(f"❌ File not found: {md_file}")
        sys.exit(1)
    
    result = markdown_to_pdf(md_file, pdf_file)
    if result:
        print(f"📄 PDF saved to: {result}")


