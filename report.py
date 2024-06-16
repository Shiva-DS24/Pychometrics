from reportlab.lib.pagesizes import A4
from reportlab.platypus import Paragraph, SimpleDocTemplate, Spacer, Table, TableStyle, PageTemplate, Frame
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.lib.units import inch, mm
from reportlab.lib import colors
from reportlab.pdfgen import canvas

class FooterCanvas(canvas.Canvas):
    def __init__(self, *args, **kwargs):
        canvas.Canvas.__init__(self, *args, **kwargs)
        self._saved_page_states = []

    def showPage(self):
        self._saved_page_states.append(dict(self.__dict__))
        self._startPage()

    def save(self):
        num_pages = len(self._saved_page_states)
        for state in self._saved_page_states:
            self.__dict__.update(state)
            self.draw_footer(num_pages)
            canvas.Canvas.showPage(self)
        canvas.Canvas.save(self)

    def draw_footer(self, page_count):
        self.setFont("Helvetica", 10)
        page_number_text = f"Page {self._pageNumber} of {page_count}"
        self.drawRightString(200 * mm, 15 * mm, page_number_text)
        footer_text = "Generated by Pychometrics Python Library"
        self.drawString(10 * mm, 15 * mm, footer_text)

def generate_pdf_report(marks_df, total_max_marks, item_analysis, stats):
    # Create a PDF document
    doc = SimpleDocTemplate("report.pdf", pagesize=A4)
    elements = []

    # Styles
    styles = getSampleStyleSheet()
    title_style = styles['Title']
    subtitle_style = styles['Heading2']
    heading1_style = styles['Heading1']
    heading2_style = styles['Heading2']
    normal_style = styles['Normal']
    bullet_style = styles['Bullet']

    # Title
    title = Paragraph("Psychometric Analysis of Assessments", title_style)
    elements.append(title)
    elements.append(Spacer(1, 0.2 * inch))

    # Subtitle with blue hyperlink
    subtitle = Paragraph('Created by <a href="https://example.com" color="blue">Pychometrics Python Library</a>', subtitle_style)
    elements.append(subtitle)
    elements.append(Spacer(1, 0.5 * inch))

    # Heading1
    heading1 = Paragraph("1. Assessment Analysis", heading1_style)
    elements.append(heading1)
    elements.append(Spacer(1, 0.2 * inch))

    # Number of students and questions
    num_students = len(marks_df)
    num_questions = len(marks_df.columns) - 2  # Exclude 'Total' and 'Total_Percentage'
    student_info = Paragraph(f"Number of students: {num_students}", normal_style)
    question_info = Paragraph(f"Number of questions: {num_questions}", normal_style)
    elements.append(student_info)
    elements.append(question_info)
    elements.append(Spacer(1, 0.5 * inch))

    # Centered text
    centered_text = Paragraph("Outcomes(%)", heading2_style)
    centered_text.style.alignment = 1  # Center alignment
    elements.append(centered_text)
    elements.append(Spacer(1, 0.2 * inch))

    # Bullet points
    bullets = [
        f"• Average: {stats['mean']}",
        f"• Median: {stats['median']}",
        f"• Standard Deviation: {stats['std_dev']}",
        f"• Skewness: {stats['skewness']}",
        f"• Kurtosis: {stats['kurt']}",
        f"• Cronbach Alpha: {stats['alpha']}",
        f"• Stand Error: {stats['sem']}"
    ]
    for bullet in bullets:
        elements.append(Paragraph(bullet, bullet_style))
    elements.append(Spacer(1, 0.5 * inch))

    # Heading2
    heading2 = Paragraph("2. Item Analysis", heading1_style)
    elements.append(heading2)
    elements.append(Spacer(1, 0.2 * inch))

    # Item analysis table with index as the first column
    item_analysis_with_index = item_analysis.reset_index()
    data = [item_analysis_with_index.columns.tolist()] + item_analysis_with_index.values.tolist()
    table = Table(data)
    table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), colors.grey),
        ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
        ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
        ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
        ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
        ('BACKGROUND', (0, 1), (-1, -1), colors.beige),
        ('GRID', (0, 0), (-1, -1), 1, colors.black)
    ]))
    elements.append(table)
    elements.append(Spacer(1, 0.5 * inch))

    # FI Ranges table
    fi_data = [
        ['FI Ranges', 'Descriptor'],
        ['Less than 40', 'Very hard'],
        ['40 - 50', 'Hard'],
        ['50 - 60', 'Appropriate for the average student'],
        ['60 - 70', 'Fairly Easy'],
        ['70 - 80', 'Easy'],
        ['More than 80', 'Very Easy']
    ]
    fi_table = Table(fi_data)
    fi_table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), colors.grey),
        ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
        ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
        ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
        ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
        ('BACKGROUND', (0, 1), (-1, -1), colors.beige),
        ('GRID', (0, 0), (-1, -1), 1, colors.black)
    ]))
    elements.append(Paragraph("FI Ranges and Descriptors", heading2_style))
    elements.append(fi_table)
    elements.append(Spacer(1, 0.5 * inch))

    # DI Ranges table
    di_data = [
        ['DI Ranges', 'Descriptor'],
        ['50% and above', 'Very Good'],
        ['40% - 50%', 'Adequate'],
        ['20% - 39%', 'Weak'],
        ['Below 20%', 'Very Weak'],
        ['Negative', 'Probably Invalid']
    ]
    di_table = Table(di_data)
    di_table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), colors.grey),
        ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
        ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
        ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
        ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
        ('BACKGROUND', (0, 1), (-1, -1), colors.beige),
        ('GRID', (0, 0), (-1, -1), 1, colors.black)
    ]))
    elements.append(Paragraph("DI Ranges and Descriptors", heading2_style))
    elements.append(di_table)
    elements.append(Spacer(1, 0.5 * inch))

    # Build the PDF
    doc.build(elements, canvasmaker=FooterCanvas)