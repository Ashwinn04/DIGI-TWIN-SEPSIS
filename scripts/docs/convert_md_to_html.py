#!/usr/bin/env python3
"""
Convert markdown file to HTML (can be printed to PDF from browser)
"""
import sys
import os

try:
    import markdown
except ImportError:
    print("Installing markdown package...")
    os.system("pip3 install markdown --quiet")
    import markdown

def markdown_to_html(md_file, html_file=None):
    """Convert markdown file to styled HTML"""
    if html_file is None:
        html_file = md_file.replace('.md', '.html')
    
    # Read markdown file
    with open(md_file, 'r', encoding='utf-8') as f:
        md_content = f.read()
    
    # Convert markdown to HTML
    html_content = markdown.markdown(md_content, extensions=['extra', 'codehilite'])
    
    # Add CSS styling
    html_with_style = f"""<!DOCTYPE html>
<html>
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Ashwin Presentation Script</title>
    <style>
        @media print {{
            @page {{
                size: A4;
                margin: 2cm;
            }}
            body {{
                font-size: 10pt;
            }}
        }}
        body {{
            font-family: 'Helvetica', 'Arial', sans-serif;
            font-size: 11pt;
            line-height: 1.6;
            color: #333;
            max-width: 900px;
            margin: 0 auto;
            padding: 20px;
            background: white;
        }}
        h1 {{
            color: #2c3e50;
            border-bottom: 3px solid #3498db;
            padding-bottom: 10px;
            margin-top: 30px;
            page-break-after: avoid;
        }}
        h2 {{
            color: #34495e;
            border-bottom: 2px solid #95a5a6;
            padding-bottom: 5px;
            margin-top: 25px;
            page-break-after: avoid;
        }}
        h3 {{
            color: #555;
            margin-top: 20px;
            page-break-after: avoid;
        }}
        h4 {{
            color: #666;
            margin-top: 15px;
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
            page-break-inside: avoid;
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
            page-break-inside: avoid;
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
            page-break-after: avoid;
        }}
        strong {{
            color: #2c3e50;
            font-weight: bold;
        }}
        p {{
            margin: 10px 0;
        }}
        .print-instructions {{
            background-color: #e8f4f8;
            border: 2px solid #3498db;
            padding: 15px;
            border-radius: 5px;
            margin-bottom: 20px;
        }}
    </style>
</head>
<body>
    <div class="print-instructions">
        <strong>📄 To convert to PDF:</strong> Press Ctrl+P (or Cmd+P on Mac) and select "Save as PDF" as the destination.
    </div>
    {html_content}
</body>
</html>"""
    
    # Write HTML file
    with open(html_file, 'w', encoding='utf-8') as f:
        f.write(html_with_style)
    
    print(f"✅ Successfully converted {md_file} to {html_file}")
    print(f"📄 Open {html_file} in your browser and press Ctrl+P (Cmd+P on Mac) to save as PDF")
    return html_file

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python3 scripts/docs/convert_md_to_html.py <markdown_file.md> [output.html]")
        sys.exit(1)
    
    md_file = sys.argv[1]
    html_file = sys.argv[2] if len(sys.argv) > 2 else None
    
    if not os.path.exists(md_file):
        print(f"❌ File not found: {md_file}")
        sys.exit(1)
    
    result = markdown_to_html(md_file, html_file)


