import io
import os
from datetime import datetime
from reportlab.lib.pagesizes import letter
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib import colors
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont


def _register_font():
    font_path = "Vazir.ttf"
    if os.path.exists(font_path):
        try:
            pdfmetrics.registerFont(TTFont('Vazir', font_path))
            return 'Vazir'
        except Exception:
            pass
    return 'Helvetica'


FONT_NAME = _register_font()

try:
    import arabic_reshaper
    from bidi.algorithm import get_display
    HAS_FA_SUPPORT = (FONT_NAME == 'Vazir')
except ImportError:
    HAS_FA_SUPPORT = False


def _reshape(text):
    if not HAS_FA_SUPPORT or not text:
        return str(text)
    try:
        reshaped = arabic_reshaper.reshape(str(text))
        return get_display(reshaped)
    except Exception:
        return str(text)


def generate_unified_report(ceph_results, bolton_3d=None):
    buffer = io.BytesIO()
    doc = SimpleDocTemplate(
        buffer, pagesize=letter,
        rightMargin=36, leftMargin=36, topMargin=36, bottomMargin=36
    )

    styles = getSampleStyleSheet()

    title_style = ParagraphStyle(
        'Title', parent=styles['Heading1'],
        fontName=FONT_NAME, fontSize=18, leading=22,
        textColor=colors.HexColor('#1E3A8A'), alignment=1, spaceAfter=10
    )
    section_style = ParagraphStyle(
        'Section', parent=styles['Heading2'],
        fontName=FONT_NAME, fontSize=13, leading=16,
        textColor=colors.HexColor('#1E40AF'),
        spaceBefore=12, spaceAfter=6, alignment=2
    )
    normal_style = ParagraphStyle(
        'Normal', parent=styles['Normal'],
        fontName=FONT_NAME, fontSize=10, leading=14, alignment=2
    )
    header_cell = ParagraphStyle(
        'HeaderCell', parent=styles['Normal'],
        fontName=FONT_NAME, fontSize=10, leading=13,
        textColor=colors.white, alignment=1
    )
    body_cell = ParagraphStyle(
        'BodyCell', parent=styles['Normal'],
        fontName=FONT_NAME, fontSize=9, leading=12, alignment=1
    )

    elements = []

    elements.append(Paragraph(
        _reshape("گزارش جامع ارتودنسی - Aariz Precision"),
        title_style
    ))
    elements.append(Paragraph(
        "Aariz Comprehensive Orthodontic Report",
        ParagraphStyle('Sub', parent=normal_style, alignment=1, textColor=colors.grey)
    ))
    elements.append(Spacer(1, 10))

    patient = ceph_results.get('patient_info', {})
    info_data = [
        [Paragraph(_reshape(f"جنسیت: {patient.get('gender', 'نامشخص')}"), body_cell),
         Paragraph(f"Date: {patient.get('date', datetime.now().strftime('%Y-%m-%d'))}", body_cell)],
        [Paragraph(_reshape(f"تشخیص: {patient.get('diagnosis', 'نامشخص')}"), body_cell),
         Paragraph(f"Pixel Size: {patient.get('pixel_size', 'N/A')} mm/px", body_cell)]
    ]
    info_table = Table(info_data, colWidths=[270, 270])
    info_table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, -1), colors.HexColor('#F3F4F6')),
        ('PADDING', (0, 0), (-1, -1), 6),
        ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
        ('GRID', (0, 0), (-1, -1), 0.5, colors.HexColor('#D1D5DB')),
    ]))
    elements.append(info_table)
    elements.append(Spacer(1, 12))

    elements.append(Paragraph(_reshape("۱. تحلیل سفالومتری (۲D)"), section_style))

    m = ceph_results.get('measurements', {})
    norms_table_data = [
        [Paragraph(_reshape("مقدار"), header_cell),
         Paragraph(_reshape("نرمال"), header_cell),
         Paragraph(_reshape("پارامتر"), header_cell)],
        [Paragraph(f"{m.get('SNA', 'N/A')}°", body_cell),
         Paragraph("82.0°", body_cell),
         Paragraph("SNA (Maxilla)", body_cell)],
        [Paragraph(f"{m.get('SNB', 'N/A')}°", body_cell),
         Paragraph("80.0°", body_cell),
         Paragraph("SNB (Mandible)", body_cell)],
        [Paragraph(f"{m.get('ANB', 'N/A')}°", body_cell),
         Paragraph("2.0°", body_cell),
         Paragraph("ANB (Skeletal Relation)", body_cell)],
        [Paragraph(f"{m.get('Wits', 'N/A')} mm", body_cell),
         Paragraph("0.0 mm", body_cell),
         Paragraph("Wits Appraisal", body_cell)],
        [Paragraph(f"{m.get('FMA', 'N/A')}°", body_cell),
         Paragraph("25.0°", body_cell),
         Paragraph("FMA (Vertical Angle)", body_cell)],
        [Paragraph(f"{m.get('Co_A', 'N/A')} mm", body_cell),
         Paragraph("90.0 mm", body_cell),
         Paragraph("Co-A (Maxilla Length)", body_cell)],
        [Paragraph(f"{m.get('Co_Gn', 'N/A')} mm", body_cell),
         Paragraph("115.0 mm", body_cell),
         Paragraph("Co-Gn (Mandible Length)", body_cell)],
    ]

    norms_table = Table(norms_table_data, colWidths=[100, 100, 340])
    norms_table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#1E40AF')),
        ('GRID', (0, 0), (-1, -1), 0.5, colors.HexColor('#D1D5DB')),
        ('ROWBACKGROUNDS', (0, 1), (-1, -1), [colors.white, colors.HexColor('#F9FAFB')]),
        ('PADDING', (0, 0), (-1, -1), 5),
        ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
    ]))
    elements.append(norms_table)
    elements.append(Spacer(1, 10))

    interp = ceph_results.get('interpretations', {})
    if interp:
        elements.append(Paragraph(
            _reshape("تفسیر بالینی:"),
            ParagraphStyle('SubSec', parent=normal_style, fontSize=11,
                           textColor=colors.HexColor('#1E40AF'))
        ))
        for cat, desc in interp.items():
            elements.append(Paragraph(f"• <b>{_reshape(cat)}</b>", normal_style))
            elements.append(Paragraph(
                _reshape(desc),
                ParagraphStyle('Indent', parent=normal_style, leftIndent=15,
                               textColor=colors.HexColor('#374151'))
            ))
            elements.append(Spacer(1, 4))

    plan = ceph_results.get('treatment_plan', [])
    if plan:
        elements.append(Spacer(1, 6))
        elements.append(Paragraph(
            _reshape("طرح درمان پیشنهادی:"),
            ParagraphStyle('SubSec2', parent=normal_style, fontSize=11,
                           textColor=colors.HexColor('#1E40AF'))
        ))
        for idx, item in enumerate(plan, 1):
            title = item.get('title', '')
            desc = item.get('desc', '')
            elements.append(Paragraph(f"{idx}. <b>{_reshape(title)}</b>", normal_style))
            elements.append(Paragraph(
                _reshape(desc),
                ParagraphStyle('Indent2', parent=normal_style, leftIndent=15,
                               textColor=colors.HexColor('#374151'))
            ))
            elements.append(Spacer(1, 4))

    elements.append(Spacer(1, 15))
    elements.append(Paragraph(_reshape("۲. تحلیل اسکن داخل دهانی (۳D)"), section_style))

    if bolton_3d:
        bolton_data = [
            [Paragraph(_reshape("وضعیت"), header_cell),
             Paragraph(_reshape("مقدار"), header_cell),
             Paragraph(_reshape("نرمال"), header_cell),
             Paragraph(_reshape("پارامتر"), header_cell)],
            [Paragraph(_reshape("Overall"), body_cell),
             Paragraph(f"{bolton_3d.get('overall_ratio', 'N/A')}%", body_cell),
             Paragraph("91.3%", body_cell),
             Paragraph("Overall Bolton Ratio", body_cell)],
            [Paragraph(_reshape("Anterior"), body_cell),
             Paragraph(f"{bolton_3d.get('ant_ratio', 'N/A')}%", body_cell),
             Paragraph("77.2%", body_cell),
             Paragraph("Anterior Bolton Ratio", body_cell)],
        ]
        bolton_table = Table(bolton_data, colWidths=[120, 100, 100, 220])
        bolton_table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#1E40AF')),
            ('GRID', (0, 0), (-1, -1), 0.5, colors.HexColor('#D1D5DB')),
            ('ROWBACKGROUNDS', (0, 1), (-1, -1), [colors.white, colors.HexColor('#F9FAFB')]),
            ('PADDING', (0, 0), (-1, -1), 5),
            ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
        ]))
        elements.append(bolton_table)
        elements.append(Spacer(1, 8))

        widths_data = [
            [Paragraph(f"{bolton_3d.get('u_tot', 'N/A')} mm", body_cell),
             Paragraph(f"{bolton_3d.get('u_ant', 'N/A')} mm", body_cell),
             Paragraph(_reshape("فک بالا"), body_cell)],
            [Paragraph(f"{bolton_3d.get('l_tot', 'N/A')} mm", body_cell),
             Paragraph(f"{bolton_3d.get('l_ant', 'N/A')} mm", body_cell),
             Paragraph(_reshape("فک پایین"), body_cell)],
        ]
        widths_table = Table(widths_data, colWidths=[150, 150, 240])
        widths_table.setStyle(TableStyle([
            ('GRID', (0, 0), (-1, -1), 0.5, colors.HexColor('#D1D5DB')),
            ('PADDING', (0, 0), (-1, -1), 5),
            ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
        ]))
        elements.append(Paragraph(
            _reshape("عرض دندان‌های اندازه‌گیری‌شده (Total / Anterior):"),
            normal_style
        ))
        elements.append(widths_table)

        elements.append(Spacer(1, 8))
        overall = bolton_3d.get('overall_ratio', 0)
        ant = bolton_3d.get('ant_ratio', 0)
        bolton_interp = ""
        if overall > 92.5:
            bolton_interp += _reshape("• Overall Bolton بالاتر از نرمال: اضافه حجم دندانی در فک پایین.<br/>")
        elif overall < 90.0:
            bolton_interp += _reshape("• Overall Bolton پایین‌تر از نرمال: اضافه حجم دندانی در فک بالا.<br/>")
        else:
            bolton_interp += _reshape("• Overall Bolton در محدوده نرمال.<br/>")

        if ant > 78.5:
            bolton_interp += _reshape("• Anterior Bolton بالاتر از نرمال: اضافه حجم دندان‌های قدامی فک پایین.<br/>")
        elif ant < 75.5:
            bolton_interp += _reshape("• Anterior Bolton پایین‌تر از نرمال: اضافه حجم دندان‌های قدامی فک بالا.<br/>")
        else:
            bolton_interp += _reshape("• Anterior Bolton در محدوده نرمال.<br/>")

        if bolton_interp:
            elements.append(Paragraph(
                _reshape("تفسیر Bolton:"),
                ParagraphStyle('SubSec3', parent=normal_style, fontSize=11,
                               textColor=colors.HexColor('#1E40AF'))
            ))
            elements.append(Paragraph(bolton_interp, normal_style))
    else:
        elements.append(Paragraph(
            _reshape(
                "⚠ توجه: نتایج ۳D (نسبت‌های بولتون و Space Analysis) در دسترس نیست. "
                "لطفاً ابتدا در اپلیکیشن ۳D اسکن‌ها را آپلود کرده و داده‌ها را وارد نمایید."
            ),
            ParagraphStyle('Note', parent=normal_style,
                           textColor=colors.HexColor('#B45309'))
        ))

    elements.append(Spacer(1, 20))
    elements.append(Paragraph(
        f"Generated by Aariz Precision Station — {datetime.now().strftime('%Y-%m-%d %H:%M')}",
        ParagraphStyle('Footer', parent=normal_style, fontSize=8,
                       alignment=1, textColor=colors.grey)
    ))

    doc.build(elements)
    buffer.seek(0)
    return buffer.getvalue()
