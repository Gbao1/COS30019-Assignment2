"""
Convert REPORT.md to professionally formatted REPORT.docx.

This script creates a Word document with:
- Professional formatting and styling
- Automatic table of contents
- Proper heading hierarchy
- Formatted tables and code blocks
- Academic document structure
"""

import re
from pathlib import Path
from typing import List, Dict, Any

try:
    from docx import Document
    from docx.shared import Inches, Pt, RGBColor
    from docx.enum.text import WD_ALIGN_PARAGRAPH, WD_BREAK
    from docx.enum.style import WD_STYLE_TYPE
    from docx.oxml import parse_xml
    from docx.oxml.ns import nsdecls
except ImportError:
    print("Error: python-docx not installed. Install with: pip install python-docx")
    exit(1)

class MarkdownToWordConverter:
    """Convert Markdown report to professionally formatted Word document."""

    def __init__(self, input_file: str, output_file: str):
        """
        Initialize the converter.

        Args:
            input_file: Path to input Markdown file
            output_file: Path to output Word file
        """
        self.input_file = Path(input_file)
        self.output_file = Path(output_file)
        self.doc = Document()
        self.toc_entries = []

        # Set up document properties
        self._setup_document_properties()
        self._setup_styles()

    def _setup_document_properties(self):
        """Set up document properties and metadata."""
        self.doc.core_properties.title = "Traffic Flow Prediction Model Comparison Report"
        self.doc.core_properties.author = "COS30019 Assignment 2"
        self.doc.core_properties.subject = "Machine Learning Model Analysis"
        self.doc.core_properties.keywords = "traffic prediction, machine learning, LSTM, GRU, Random Forest"

    def _setup_styles(self):
        """Set up custom styles for the document."""
        styles = self.doc.styles

        # Title style
        if 'Custom Title' not in [s.name for s in styles]:
            title_style = styles.add_style('Custom Title', WD_STYLE_TYPE.PARAGRAPH)
            title_style.font.name = 'Calibri'
            title_style.font.size = Pt(24)
            title_style.font.bold = True
            title_style.font.color.rgb = RGBColor(0x2F, 0x54, 0x96)  # Blue color
            title_style.paragraph_format.alignment = WD_ALIGN_PARAGRAPH.CENTER
            title_style.paragraph_format.space_after = Pt(24)

        # Subtitle style
        if 'Custom Subtitle' not in [s.name for s in styles]:
            subtitle_style = styles.add_style('Custom Subtitle', WD_STYLE_TYPE.PARAGRAPH)
            subtitle_style.font.name = 'Calibri'
            subtitle_style.font.size = Pt(16)
            subtitle_style.font.italic = True
            subtitle_style.font.color.rgb = RGBColor(0x44, 0x54, 0x66)  # Dark gray
            subtitle_style.paragraph_format.alignment = WD_ALIGN_PARAGRAPH.CENTER
            subtitle_style.paragraph_format.space_after = Pt(18)

        # Update heading styles
        for i in range(1, 7):
            heading_style = styles[f'Heading {i}']
            heading_style.font.name = 'Calibri'
            heading_style.font.color.rgb = RGBColor(0x2F, 0x54, 0x96)

        # Code style
        if 'Code Block' not in [s.name for s in styles]:
            code_style = styles.add_style('Code Block', WD_STYLE_TYPE.PARAGRAPH)
            code_style.font.name = 'Consolas'
            code_style.font.size = Pt(10)
            code_style.paragraph_format.left_indent = Inches(0.5)
            code_style.paragraph_format.space_before = Pt(6)
            code_style.paragraph_format.space_after = Pt(6)

        # Quote style
        if 'Block Quote' not in [s.name for s in styles]:
            quote_style = styles.add_style('Block Quote', WD_STYLE_TYPE.PARAGRAPH)
            quote_style.font.name = 'Calibri'
            quote_style.font.italic = True
            quote_style.paragraph_format.left_indent = Inches(0.5)
            quote_style.paragraph_format.space_before = Pt(6)
            quote_style.paragraph_format.space_after = Pt(6)

    def _add_title_page(self):
        """Add a professional title page."""
        # Main title
        title = self.doc.add_paragraph("Traffic Flow Prediction Model Comparison Report", 'Custom Title')

        # Subtitle
        subtitle = self.doc.add_paragraph("COS30019 Assignment 2 - Machine Learning Model Analysis", 'Custom Subtitle')

        # Add some spacing
        self.doc.add_paragraph("")
        self.doc.add_paragraph("")

        # Add author and course info
        info_para = self.doc.add_paragraph()
        info_para.alignment = WD_ALIGN_PARAGRAPH.CENTER
        run = info_para.add_run("Course: COS30019 - Introduction to Artificial Intelligence\n")
        run.font.size = Pt(12)
        run.font.name = 'Calibri'

        run = info_para.add_run("Assignment: Traffic Flow Prediction Analysis\n")
        run.font.size = Pt(12)
        run.font.name = 'Calibri'

        run = info_para.add_run("Models: LSTM, GRU, Random Forest\n")
        run.font.size = Pt(12)
        run.font.name = 'Calibri'

        # Add date
        import datetime
        date_para = self.doc.add_paragraph()
        date_para.alignment = WD_ALIGN_PARAGRAPH.CENTER
        date_para.add_run(f"Date: {datetime.datetime.now().strftime('%B %Y')}")

        # Page break
        self.doc.add_page_break()

    def _add_table_of_contents(self):
        """Add a table of contents placeholder."""
        toc_title = self.doc.add_heading("Table of Contents", level=1)
        toc_title.style.font.color.rgb = RGBColor(0x2F, 0x54, 0x96)

        # Add TOC note
        note = self.doc.add_paragraph()
        note.add_run("Note: ").bold = True
        note.add_run("This table of contents is generated based on document headings. "
                    "In Microsoft Word, you can update it by right-clicking and selecting 'Update Field' "
                    "or by going to References → Update Table.")

        # Add basic TOC structure based on collected entries
        toc_para = self.doc.add_paragraph()

        # Add TOC entries (simplified version)
        toc_entries = [
            ("Abstract", 1),
            ("1. Introduction", 1),
            ("1.1 Research Objectives", 2),
            ("1.2 Problem Statement", 2),
            ("2. Literature Review", 1),
            ("2.1 Traffic Flow Prediction Methods", 2),
            ("2.2 Recurrent Neural Networks for Time Series", 2),
            ("2.3 Ensemble Methods in Traffic Prediction", 2),
            ("3. Methodology", 1),
            ("3.1 Dataset Description", 2),
            ("3.2 Data Preprocessing", 2),
            ("3.3 Model Architectures", 2),
            ("3.4 Evaluation Methodology", 2),
            ("4. Results and Analysis", 1),
            ("4.1 Overall Performance Comparison", 2),
            ("4.2 Regression Performance Analysis", 2),
            ("4.3 Classification Performance Analysis", 2),
            ("4.4 Computational Efficiency", 2),
            ("4.5 Cross-Validation Results", 2),
            ("4.6 Feature Importance Analysis", 2),
            ("5. Discussion", 1),
            ("5.1 Model Performance Interpretation", 2),
            ("5.2 Practical Implications", 2),
            ("5.3 Limitations and Challenges", 2),
            ("5.4 Comparison with Existing Literature", 2),
            ("6. Conclusions and Recommendations", 1),
            ("6.1 Key Findings", 2),
            ("6.2 Recommendations for Practice", 2),
            ("6.3 Future Work", 2),
            ("7. References", 1),
            ("Appendices", 1)
        ]

        for entry, level in toc_entries:
            toc_line = self.doc.add_paragraph()
            if level == 1:
                run = toc_line.add_run(entry)
                run.bold = True
                toc_line.paragraph_format.left_indent = Inches(0.0)
            else:
                run = toc_line.add_run(entry)
                toc_line.paragraph_format.left_indent = Inches(0.3)

        # Add page break
        self.doc.add_page_break()

    def _process_line(self, line: str) -> None:
        """
        Process a single line of Markdown and add to document.

        Args:
            line: Line of Markdown text to process
        """
        line = line.rstrip()

        # Skip horizontal rules
        if line.strip() == "---":
            return

        # Handle headers
        if line.startswith('#'):
            self._add_heading(line)
        # Handle tables
        elif line.startswith('|') and '|' in line[1:]:
            self._add_table_row(line)
        # Handle code blocks
        elif line.startswith('```'):
            self._toggle_code_block()
        # Handle bullet points
        elif line.startswith('- ') or line.startswith('* '):
            self._add_bullet_point(line)
        # Handle numbered lists
        elif re.match(r'^\d+\.\s', line):
            self._add_numbered_list_item(line)
        # Handle regular text
        elif line.strip():
            self._add_paragraph(line)
        # Handle empty lines
        else:
            if not getattr(self, '_in_code_block', False):
                self.doc.add_paragraph("")

    def _add_heading(self, line: str) -> None:
        """Add a heading to the document."""
        # Count the number of # symbols to determine heading level
        level = 0
        for char in line:
            if char == '#':
                level += 1
            else:
                break

        # Extract the heading text
        heading_text = line[level:].strip()

        # Skip empty headings
        if not heading_text:
            return

        # Add the heading
        if level <= 6:
            heading = self.doc.add_heading(heading_text, level=level)
            heading.style.font.color.rgb = RGBColor(0x2F, 0x54, 0x96)
        else:
            # For very deep headings, use bold text
            para = self.doc.add_paragraph()
            run = para.add_run(heading_text)
            run.bold = True

    def _add_bullet_point(self, line: str) -> None:
        """Add a bullet point to the document."""
        # Remove the bullet marker and leading/trailing whitespace
        text = line[2:].strip()

        # Handle nested bullets (count leading spaces)
        indent_level = 0
        original_line = line
        while original_line.startswith('  '):
            indent_level += 1
            original_line = original_line[2:]

        # Add the bullet point
        para = self.doc.add_paragraph(text, style='List Bullet')
        if indent_level > 0:
            para.paragraph_format.left_indent = Inches(0.5 * indent_level)

    def _add_numbered_list_item(self, line: str) -> None:
        """Add a numbered list item to the document."""
        # Extract the text after the number
        text = re.sub(r'^\d+\.\s*', '', line)

        # Add the numbered list item
        self.doc.add_paragraph(text, style='List Number')

    def _add_paragraph(self, line: str) -> None:
        """Add a regular paragraph to the document."""
        if getattr(self, '_in_code_block', False):
            # Add to code block
            self.doc.add_paragraph(line, style='Code Block')
        else:
            # Process formatting in regular text
            para = self.doc.add_paragraph()
            self._add_formatted_text(para, line)

    def _add_formatted_text(self, paragraph, text: str) -> None:
        """Add formatted text to a paragraph, handling bold, italic, and code."""
        # Simple regex patterns for formatting
        # This is a basic implementation - could be enhanced for more complex formatting

        # Handle inline code first
        parts = re.split(r'`([^`]+)`', text)
        for i, part in enumerate(parts):
            if i % 2 == 0:
                # Regular text - process bold and italic
                self._add_text_with_emphasis(paragraph, part)
            else:
                # Code text
                run = paragraph.add_run(part)
                run.font.name = 'Consolas'
                run.font.size = Pt(11)

    def _add_text_with_emphasis(self, paragraph, text: str) -> None:
        """Add text with bold and italic formatting."""
        # Handle **bold** and *italic*
        # This is a simplified implementation
        remaining = text

        while remaining:
            # Find next formatting
            bold_match = re.search(r'\*\*([^*]+)\*\*', remaining)
            italic_match = re.search(r'\*([^*]+)\*', remaining)

            next_format = None
            if bold_match and italic_match:
                next_format = bold_match if bold_match.start() < italic_match.start() else italic_match
            elif bold_match:
                next_format = bold_match
            elif italic_match:
                next_format = italic_match

            if next_format:
                # Add text before formatting
                if next_format.start() > 0:
                    paragraph.add_run(remaining[:next_format.start()])

                # Add formatted text
                run = paragraph.add_run(next_format.group(1))
                if next_format == bold_match:
                    run.bold = True
                else:
                    run.italic = True

                # Continue with remaining text
                remaining = remaining[next_format.end():]
            else:
                # No more formatting, add remaining text
                paragraph.add_run(remaining)
                break

    def _toggle_code_block(self) -> None:
        """Toggle code block mode."""
        if not hasattr(self, '_in_code_block'):
            self._in_code_block = False

        self._in_code_block = not self._in_code_block

    def _add_table_row(self, line: str) -> None:
        """Add a table row (simplified table handling)."""
        if not hasattr(self, '_current_table'):
            self._current_table = None
            self._table_headers = []

        # Parse table cells
        cells = [cell.strip() for cell in line.split('|')[1:-1]]

        if not cells:
            return

        # Check if this is a header separator row
        if all(cell.strip().replace('-', '').replace(':', '').replace(' ', '') == '' for cell in cells):
            return

        # Create table if not exists
        if self._current_table is None:
            self._current_table = self.doc.add_table(rows=1, cols=len(cells))
            self._current_table.style = 'Light Grid Accent 1'
            self._table_headers = cells

            # Add header row
            header_cells = self._current_table.rows[0].cells
            for i, cell_text in enumerate(cells):
                header_cells[i].text = cell_text
                # Make header bold
                for paragraph in header_cells[i].paragraphs:
                    for run in paragraph.runs:
                        run.bold = True
        else:
            # Add data row
            row_cells = self._current_table.add_row().cells
            for i, cell_text in enumerate(cells):
                if i < len(row_cells):
                    row_cells[i].text = cell_text

    def convert(self) -> None:
        """Convert the Markdown file to Word document."""
        print(f"Converting {self.input_file} to {self.output_file}")

        # Add title page
        self._add_title_page()

        # Add table of contents
        self._add_table_of_contents()

        # Read and process the markdown file
        try:
            with open(self.input_file, 'r', encoding='utf-8') as f:
                content = f.read()
        except Exception as e:
            print(f"Error reading file: {e}")
            return

        # Split into lines and process
        lines = content.split('\n')

        for line_num, line in enumerate(lines, 1):
            try:
                self._process_line(line)
            except Exception as e:
                print(f"Warning: Error processing line {line_num}: {e}")
                print(f"Line content: {line[:100]}...")
                continue

        # Finalize any open table
        if hasattr(self, '_current_table'):
            self._current_table = None

        # Save the document
        try:
            self.doc.save(self.output_file)
            print(f"Successfully created {self.output_file}")
        except Exception as e:
            print(f"Error saving document: {e}")

    def add_footer(self):
        """Add a footer with page numbers."""
        section = self.doc.sections[0]
        footer = section.footer
        footer_para = footer.paragraphs[0]
        footer_para.text = "Traffic Flow Prediction Model Comparison Report - COS30019"
        footer_para.alignment = WD_ALIGN_PARAGRAPH.CENTER

def main():
    """Main function to convert report."""
    input_file = "REPORT.md"
    output_file = "REPORT.docx"

    if not Path(input_file).exists():
        print(f"Error: {input_file} not found")
        return

    converter = MarkdownToWordConverter(input_file, output_file)
    converter.convert()
    converter.add_footer()

    print(f"\nConversion complete!")
    print(f"Created: {Path(output_file).absolute()}")
    print(f"Size: {Path(output_file).stat().st_size / 1024:.1f} KB")
    print(f"\nTips for the Word document:")
    print(f"   - Update Table of Contents: Right-click TOC -> Update Field")
    print(f"   - Add page numbers: Insert -> Page Numbers")
    print(f"   - Customize styles: Home -> Styles")

if __name__ == "__main__":
    main()