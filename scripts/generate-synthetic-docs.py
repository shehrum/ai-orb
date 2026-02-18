#!/usr/bin/env python3
"""
Generate 3 synthetic legal PDF documents using reportlab.

Documents:
1. commercial-lease-100-bishopsgate.pdf  (~6-8 pages)
2. title-report-lot-7.pdf               (~4-5 pages)
3. environmental-assessment-manchester.pdf (~5-6 pages)
"""

import os
from datetime import date

from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import mm, cm
from reportlab.lib.enums import TA_CENTER, TA_JUSTIFY, TA_LEFT, TA_RIGHT
from reportlab.lib.colors import black, HexColor
from reportlab.platypus import (
    SimpleDocTemplate,
    Paragraph,
    Spacer,
    PageBreak,
    Table,
    TableStyle,
    Frame,
    PageTemplate,
    BaseDocTemplate,
)
from reportlab.lib import colors

OUTPUT_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "synthetic-docs"
)
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ---------------------------------------------------------------------------
# Shared styles
# ---------------------------------------------------------------------------

_base = getSampleStyleSheet()

STYLES = {
    "title": ParagraphStyle(
        "DocTitle",
        parent=_base["Title"],
        fontName="Times-Bold",
        fontSize=28,
        leading=34,
        alignment=TA_CENTER,
        spaceAfter=12,
    ),
    "subtitle": ParagraphStyle(
        "DocSubtitle",
        parent=_base["Title"],
        fontName="Times-Roman",
        fontSize=16,
        leading=20,
        alignment=TA_CENTER,
        spaceAfter=6,
    ),
    "heading1": ParagraphStyle(
        "Heading1Legal",
        fontName="Helvetica-Bold",
        fontSize=14,
        leading=18,
        spaceBefore=18,
        spaceAfter=8,
        textColor=black,
    ),
    "heading2": ParagraphStyle(
        "Heading2Legal",
        fontName="Helvetica-Bold",
        fontSize=12,
        leading=15,
        spaceBefore=12,
        spaceAfter=6,
        leftIndent=12,
        textColor=black,
    ),
    "heading3": ParagraphStyle(
        "Heading3Legal",
        fontName="Helvetica-Bold",
        fontSize=11,
        leading=14,
        spaceBefore=8,
        spaceAfter=4,
        leftIndent=24,
        textColor=black,
    ),
    "body": ParagraphStyle(
        "BodyLegal",
        fontName="Times-Roman",
        fontSize=11,
        leading=14,
        alignment=TA_JUSTIFY,
        spaceBefore=2,
        spaceAfter=6,
        leftIndent=12,
    ),
    "body_indent": ParagraphStyle(
        "BodyLegalIndent",
        fontName="Times-Roman",
        fontSize=11,
        leading=14,
        alignment=TA_JUSTIFY,
        spaceBefore=2,
        spaceAfter=6,
        leftIndent=24,
    ),
    "body_indent2": ParagraphStyle(
        "BodyLegalIndent2",
        fontName="Times-Roman",
        fontSize=11,
        leading=14,
        alignment=TA_JUSTIFY,
        spaceBefore=2,
        spaceAfter=6,
        leftIndent=36,
    ),
    "center": ParagraphStyle(
        "CenterText",
        fontName="Times-Roman",
        fontSize=11,
        leading=14,
        alignment=TA_CENTER,
        spaceAfter=6,
    ),
    "footer": ParagraphStyle(
        "FooterStyle",
        fontName="Helvetica",
        fontSize=8,
        leading=10,
        alignment=TA_CENTER,
        textColor=HexColor("#666666"),
    ),
    "header_right": ParagraphStyle(
        "HeaderRight",
        fontName="Helvetica",
        fontSize=8,
        leading=10,
        alignment=TA_RIGHT,
        textColor=HexColor("#666666"),
    ),
    "table_header": ParagraphStyle(
        "TableHeader",
        fontName="Helvetica-Bold",
        fontSize=10,
        leading=13,
        alignment=TA_LEFT,
    ),
    "table_body": ParagraphStyle(
        "TableBody",
        fontName="Times-Roman",
        fontSize=10,
        leading=13,
        alignment=TA_LEFT,
    ),
    "small_bold": ParagraphStyle(
        "SmallBold",
        fontName="Times-Bold",
        fontSize=10,
        leading=13,
        alignment=TA_LEFT,
        spaceAfter=4,
    ),
}


def _header_footer(canvas, doc, title_text, confidential=False):
    """Draw header and footer on every page."""
    canvas.saveState()
    # Header line
    canvas.setStrokeColor(HexColor("#333333"))
    canvas.setLineWidth(0.5)
    canvas.line(20 * mm, A4[1] - 18 * mm, A4[0] - 20 * mm, A4[1] - 18 * mm)
    canvas.setFont("Helvetica", 8)
    canvas.setFillColor(HexColor("#666666"))
    canvas.drawString(20 * mm, A4[1] - 16 * mm, title_text)
    if confidential:
        canvas.drawRightString(A4[0] - 20 * mm, A4[1] - 16 * mm, "PRIVATE & CONFIDENTIAL")
    # Footer line
    canvas.line(20 * mm, 18 * mm, A4[0] - 20 * mm, 18 * mm)
    canvas.drawCentredString(A4[0] / 2, 12 * mm, f"Page {doc.page}")
    canvas.restoreState()


# ===========================================================================
# DOCUMENT 1 — Commercial Lease
# ===========================================================================


def _build_lease():
    path = os.path.join(OUTPUT_DIR, "commercial-lease-100-bishopsgate.pdf")
    doc = SimpleDocTemplate(
        path,
        pagesize=A4,
        topMargin=28 * mm,
        bottomMargin=25 * mm,
        leftMargin=25 * mm,
        rightMargin=25 * mm,
    )

    def _on_page(canvas, doc):
        _header_footer(canvas, doc, "Commercial Lease — 100 Bishopsgate", True)

    story = []
    S = STYLES

    # --- Title page ---
    story.append(Spacer(1, 80 * mm))
    story.append(Paragraph("LEASE", S["title"]))
    story.append(Spacer(1, 10 * mm))
    story.append(Paragraph("relating to", S["center"]))
    story.append(Spacer(1, 4 * mm))
    story.append(Paragraph("<b>Floors 8-10, 100 Bishopsgate, London EC2M 1GT</b>", S["center"]))
    story.append(Spacer(1, 12 * mm))
    story.append(Paragraph("between", S["center"]))
    story.append(Spacer(1, 4 * mm))
    story.append(Paragraph("<b>BISHOPSGATE PROPERTY HOLDINGS LIMITED</b>", S["center"]))
    story.append(Paragraph("(Landlord)", S["center"]))
    story.append(Spacer(1, 6 * mm))
    story.append(Paragraph("and", S["center"]))
    story.append(Spacer(1, 6 * mm))
    story.append(Paragraph("<b>MERIDIAN CONSULTING GROUP LLP</b>", S["center"]))
    story.append(Paragraph("(Tenant)", S["center"]))
    story.append(Spacer(1, 16 * mm))
    story.append(Paragraph("Dated 1 January 2024", S["center"]))
    story.append(Spacer(1, 20 * mm))
    story.append(
        Paragraph(
            "Prepared by Whitfield & Partners LLP, Solicitors<br/>14 Bedford Row, London WC1R 4ED",
            S["center"],
        )
    )
    story.append(PageBreak())

    # --- Section 1 — Definitions and Interpretation ---
    story.append(Paragraph("Section 1 — Definitions and Interpretation", S["heading1"]))
    story.append(
        Paragraph(
            "1.1 In this Lease, unless the context otherwise requires, the following expressions shall have the meanings set out below:",
            S["body"],
        )
    )
    defs = [
        (
            '"the Landlord"',
            "means Bishopsgate Property Holdings Limited (Company No. 05198234), whose registered office is at 1 Poultry, London EC2R 8EJ, and includes its successors in title and assigns;",
        ),
        (
            '"the Tenant"',
            "means Meridian Consulting Group LLP (Registration No. OC412987), whose registered office is at 25 Chancery Lane, London WC2A 1PL, and includes its successors in title and permitted assigns;",
        ),
        (
            '"the Premises"',
            "means the office accommodation comprising Floors 8, 9 and 10 of the building known as 100 Bishopsgate, London EC2M 1GT, as more particularly delineated on the plans annexed hereto and edged in red, together with all fixtures and fittings (other than tenant's fixtures) and the Landlord's furniture therein;",
        ),
        (
            '"the Building"',
            "means the building known as 100 Bishopsgate, London EC2M 1GT, including all common parts, service areas, structural elements, and the curtilage thereof;",
        ),
        (
            '"the Term"',
            "means a term of fifteen (15) years commencing on the Term Commencement Date;",
        ),
        (
            '"the Term Commencement Date"',
            "means 1 January 2024;",
        ),
        (
            '"the Initial Rent"',
            "means Eight Hundred and Fifty Thousand Pounds (£850,000) per annum exclusive of Value Added Tax;",
        ),
        (
            '"the Review Dates"',
            "means the fifth (5th) and tenth (10th) anniversaries of the Term Commencement Date, being 1 January 2029 and 1 January 2034 respectively;",
        ),
        (
            '"the Permitted Use"',
            "means use falling within Class E(g)(i) of the Town and Country Planning (Use Classes) Order 1987 (as amended), being use as offices;",
        ),
        (
            '"the Insured Risks"',
            "means fire, lightning, explosion, storm, tempest, flood, earthquake, impact by aircraft and articles dropped therefrom, riot, civil commotion, malicious damage, subsidence, burst pipes, and such other risks as the Landlord may from time to time reasonably determine;",
        ),
    ]
    for term, defn in defs:
        story.append(Paragraph(f"<b>{term}</b> {defn}", S["body_indent"]))

    story.append(
        Paragraph(
            "1.2 In this Lease, references to statutes or statutory provisions include any statute or statutory provision which amends, extends, consolidates or replaces the same, and shall include any orders, regulations, instruments or other subordinate legislation made under the relevant statute or statutory provision.",
            S["body"],
        )
    )
    story.append(
        Paragraph(
            "1.3 Words importing the singular number include the plural number and vice versa. Words importing one gender include all genders. References to persons include bodies corporate, unincorporated associations and partnerships.",
            S["body"],
        )
    )
    story.append(
        Paragraph(
            "1.4 The headings in this Lease are for convenience only and shall not affect the interpretation of any provision of this Lease.",
            S["body"],
        )
    )

    # --- Section 2 — Demise ---
    story.append(Paragraph("Section 2 — Demise", S["heading1"]))
    story.append(
        Paragraph(
            "2.1 In consideration of the rents herein reserved and the covenants on the part of the Tenant herein contained, the Landlord hereby demises unto the Tenant ALL THAT the Premises TOGETHER WITH the rights set out in Schedule 1 hereto but EXCEPTING AND RESERVING unto the Landlord the rights set out in Schedule 2 hereto TO HOLD the same unto the Tenant for the Term, YIELDING AND PAYING therefor during the Term:",
            S["body"],
        )
    )
    story.append(
        Paragraph(
            "(a) the yearly rent specified in clause 3 of this Lease, or such revised rent as may become payable pursuant to clause 3.2; and",
            S["body_indent"],
        )
    )
    story.append(
        Paragraph(
            "(b) by way of further additional rent, all other sums payable by the Tenant under this Lease.",
            S["body_indent"],
        )
    )
    story.append(
        Paragraph(
            "2.2 The Premises comprise the entirety of Floors 8, 9 and 10 of the Building, having an aggregate net internal area of approximately 32,500 square feet (3,019 square metres), as measured in accordance with the RICS Code of Measuring Practice (6th Edition).",
            S["body"],
        )
    )
    story.append(
        Paragraph(
            "2.3 The Term shall be for a period of fifteen (15) years commencing on 1 January 2024 and expiring on 31 December 2038, subject to earlier determination in accordance with the provisions of this Lease.",
            S["body"],
        )
    )

    # --- Section 3 — Rent ---
    story.append(Paragraph("Section 3 — Rent", S["heading1"]))
    story.append(
        Paragraph(
            "3.1 The Tenant shall pay to the Landlord the Initial Rent of Eight Hundred and Fifty Thousand Pounds (£850,000) per annum, without any deduction, counterclaim or set-off whatsoever, by equal quarterly payments in advance on the usual quarter days (being 25 March, 24 June, 29 September and 25 December in each year), the first such payment (or a due proportion thereof in respect of the period from the Term Commencement Date to the next quarter day) to be made on the Term Commencement Date.",
            S["body"],
        )
    )
    story.append(Paragraph("3.2 Rent Review Mechanism", S["heading2"]))
    story.append(
        Paragraph(
            '3.2.1 The rent payable under this Lease shall be reviewed on each Review Date in accordance with the provisions of this clause 3.2. With effect from each Review Date, the rent payable shall be the higher of (a) the rent payable immediately before the relevant Review Date and (b) the open market rent of the Premises as at the relevant Review Date, determined in accordance with clauses 3.2.2 to 3.2.5 below (the <b>"Reviewed Rent"</b>).',
            S["body_indent"],
        )
    )
    story.append(
        Paragraph(
            "3.2.2 The open market rent shall be such rent as might reasonably be expected to be obtained on a letting of the Premises in the open market at the relevant Review Date by a willing landlord to a willing tenant, on the terms of this Lease (other than the amount of rent but including provisions for rent review), for a term equal to the residue of the Term unexpired at the Review Date or fifteen (15) years, whichever is the longer.",
            S["body_indent"],
        )
    )
    story.append(
        Paragraph(
            "3.2.3 In determining the open market rent, the following assumptions shall be made: (a) that the Premises are available to let on the open market without a fine or premium; (b) that the Premises are fit for immediate occupation and use; (c) that no work has been carried out to the Premises by the Tenant or any undertenant which has diminished the rental value of the Premises; and (d) that all covenants on the part of the Landlord and the Tenant have been fully observed and performed.",
            S["body_indent"],
        )
    )
    story.append(
        Paragraph(
            "3.2.4 In determining the open market rent, the following matters shall be disregarded: (a) any effect on rent of the fact that the Tenant or any undertenant has been in occupation of the Premises; (b) any goodwill attached to the Premises by reason of the carrying on thereat of the business of the Tenant or any undertenant; (c) any improvement to the Premises carried out by and at the expense of the Tenant during the Term otherwise than in pursuance of an obligation to the Landlord.",
            S["body_indent"],
        )
    )
    story.append(
        Paragraph(
            "3.2.5 In default of agreement between the Landlord and the Tenant as to the Reviewed Rent by the relevant Review Date, the Reviewed Rent shall be determined by an independent surveyor acting as an expert (and not as an arbitrator) appointed by agreement between the parties or, in default of agreement, by the President for the time being of the Royal Institution of Chartered Surveyors on the application of either party. The rent review shall be upward only.",
            S["body_indent"],
        )
    )
    story.append(
        Paragraph(
            "3.3 If and so long as the Reviewed Rent has not been ascertained by the relevant Review Date, the Tenant shall continue to pay rent at the rate payable immediately before that Review Date. Upon the Reviewed Rent being ascertained, the Tenant shall pay to the Landlord within fourteen (14) days any shortfall, together with interest thereon at 2% above the base rate of Barclays Bank PLC from the relevant Review Date to the date of payment.",
            S["body"],
        )
    )

    # --- Section 4 — Repair and Maintenance ---
    story.append(Paragraph("Section 4 — Repair and Maintenance", S["heading1"]))
    story.append(Paragraph("4.1 Tenant's Obligations", S["heading2"]))
    story.append(
        Paragraph(
            "4.1.1 The Tenant shall keep the Premises and every part thereof (including all additions and alterations thereto) in good and substantial repair and condition throughout the Term, fair wear and tear excepted, and shall yield up the Premises in such repair and condition at the expiration or sooner determination of the Term.",
            S["body_indent"],
        )
    )
    story.append(
        Paragraph(
            "4.1.2 The Tenant shall keep the interior of the Premises (including the internal surfaces of all external and load-bearing walls, floors, ceilings, columns, doors, windows and window frames) properly decorated to a standard consistent with that of a first-class office building in the City of London, and shall redecorate the same not less frequently than once in every five (5) years of the Term and in the last year of the Term (howsoever determined).",
            S["body_indent"],
        )
    )
    story.append(
        Paragraph(
            "4.1.3 The Tenant shall not commit or permit any waste, spoil or destruction in or upon the Premises.",
            S["body_indent"],
        )
    )
    story.append(Paragraph("4.2 Landlord's Obligations", S["heading2"]))
    story.append(
        Paragraph(
            "4.2.1 The Landlord shall keep in good and substantial repair the structure and exterior of the Building (including the roof, foundations, external walls and common parts) and all conducting media serving the Building.",
            S["body_indent"],
        )
    )
    story.append(
        Paragraph(
            "4.2.2 The Landlord shall maintain, repair and as necessary renew all plant, machinery and equipment serving the Building, including the lifts, air conditioning system, fire detection and suppression systems, and common area lighting.",
            S["body_indent"],
        )
    )
    story.append(
        Paragraph(
            "4.2.3 The cost of the Landlord's obligations under this clause 4.2 shall be recoverable from the Tenant as part of the service charge in accordance with Schedule 3 hereto, the Tenant's proportion being 18.7% (representing the ratio of the net internal area of the Premises to the total lettable area of the Building).",
            S["body_indent"],
        )
    )
    story.append(
        Paragraph(
            "4.3 A schedule of condition recording the state and condition of the Premises as at the Term Commencement Date has been prepared by Carter Mitchell Surveyors LLP and is annexed to this Lease at Schedule 4. The Tenant's repairing obligations under clause 4.1 shall not require the Tenant to put the Premises in any better state or condition than that evidenced by the schedule of condition.",
            S["body"],
        )
    )

    # --- Section 5 — Insurance ---
    story.append(Paragraph("Section 5 — Insurance", S["heading1"]))
    story.append(
        Paragraph(
            "5.1 The Landlord shall insure the Building (including the Premises) against the Insured Risks for their full reinstatement value (including professional fees, demolition and site clearance costs, and Value Added Tax) and against loss of rent for a period of not less than three (3) years.",
            S["body"],
        )
    )
    story.append(
        Paragraph(
            "5.2 The Landlord shall use reasonable endeavours to procure that the insurer waives all rights of subrogation against the Tenant in respect of the Insured Risks.",
            S["body"],
        )
    )
    story.append(
        Paragraph(
            "5.3 The Tenant shall reimburse to the Landlord on demand a fair proportion (being the Tenant's proportion as defined in clause 4.2.3) of the premium paid or payable by the Landlord for such insurance and any excess applicable under the policy. Such reimbursement shall constitute additional rent payable under this Lease.",
            S["body"],
        )
    )
    story.append(
        Paragraph(
            "5.4 The Tenant shall not do or permit anything to be done on the Premises which may render void or voidable any policy of insurance on the Building or which may cause any increased premium to become payable in respect thereof.",
            S["body"],
        )
    )

    # --- Section 6 — Permitted Use ---
    story.append(Paragraph("Section 6 — Permitted Use", S["heading1"]))
    story.append(
        Paragraph(
            "6.1 The Tenant shall use the Premises for the Permitted Use only and for no other purpose whatsoever. The Permitted Use shall be limited to use falling within Class E(g)(i) of the Town and Country Planning (Use Classes) Order 1987 (as amended), being offices for the conduct of business operations that may be carried out in a residential area without detriment to its amenity.",
            S["body"],
        )
    )
    story.append(
        Paragraph(
            "6.2 The Tenant shall not use the Premises or any part thereof for any illegal or immoral purpose, for any sale by auction, or for any purpose which in the reasonable opinion of the Landlord may be or become a nuisance, annoyance or disturbance to the Landlord or the other tenants or occupiers of the Building or the neighbourhood.",
            S["body"],
        )
    )

    # --- Section 7 — Alienation ---
    story.append(Paragraph("Section 7 — Alienation", S["heading1"]))
    story.append(Paragraph("7.1 Assignment", S["heading2"]))
    story.append(
        Paragraph(
            "7.1.1 The Tenant shall not assign the whole or any part of this Lease without the prior written consent of the Landlord, such consent not to be unreasonably withheld or delayed. The Landlord may impose reasonable conditions on any such consent, including (without limitation) a requirement for an authorised guarantee agreement from the outgoing Tenant.",
            S["body_indent"],
        )
    )
    story.append(
        Paragraph(
            "7.1.2 The Landlord shall not be obliged to give consent to an assignment unless the proposed assignee has first demonstrated to the Landlord's reasonable satisfaction that: (a) it is of sufficient financial standing to enable it to comply with the Tenant's covenants in this Lease; and (b) it is a respectable and responsible person or entity.",
            S["body_indent"],
        )
    )
    story.append(Paragraph("7.2 Subletting of Whole", S["heading2"]))
    story.append(
        Paragraph(
            "7.2.1 The Tenant shall not sublet the whole of the Premises without the prior written consent of the Landlord, such consent not to be unreasonably withheld or delayed, provided that any subletting shall be at a rent not less than the higher of the rent payable under this Lease and the then open market rent of the Premises.",
            S["body_indent"],
        )
    )
    story.append(
        Paragraph(
            "7.2.2 Any subletting shall be by way of a sub-lease which contains provisions for upward only rent review at intervals of not more than five (5) years, and which otherwise contains provisions consistent with the terms of this Lease.",
            S["body_indent"],
        )
    )
    story.append(Paragraph("7.3 Subletting of Part", S["heading2"]))
    story.append(
        Paragraph(
            "7.3.1 The Tenant shall not sublet any part of the Premises (as distinct from the whole).",
            S["body_indent"],
        )
    )

    # --- Section 8 — Break Clause ---
    story.append(Paragraph("Section 8 — Break Clause", S["heading1"]))
    story.append(Paragraph("8.1 Tenant's Break Rights", S["heading2"]))
    story.append(
        Paragraph(
            '8.1.1 The Tenant may determine this Lease on the fifth (5th) anniversary of the Term Commencement Date (being 1 January 2029) or the tenth (10th) anniversary of the Term Commencement Date (being 1 January 2034) (each a <b>"Break Date"</b>) by serving written notice on the Landlord in accordance with clause 8.2.',
            S["body_indent"],
        )
    )
    story.append(Paragraph("8.2 Notice Requirements", S["heading2"]))
    story.append(
        Paragraph(
            '8.2.1 The Tenant must give the Landlord not less than twelve (12) months\' prior written notice of its intention to exercise a break right (a <b>"Break Notice"</b>). A Break Notice, once given, shall be irrevocable. Any Break Notice must be served in accordance with Section 196 of the Law of Property Act 1925 (as amended).',
            S["body_indent"],
        )
    )
    story.append(Paragraph("8.3 Conditions for Exercise of Break", S["heading2"]))
    story.append(
        Paragraph(
            "8.3.1 The Tenant's right to determine this Lease on a Break Date shall be conditional upon the following conditions being satisfied on or before the relevant Break Date:",
            S["body_indent"],
        )
    )
    story.append(
        Paragraph(
            "(a) there is no material breach by the Tenant of any of the Tenant's covenants in this Lease which remains unremedied at the Break Date;",
            S["body_indent2"],
        )
    )
    story.append(
        Paragraph(
            "(b) the Tenant gives vacant possession of the whole of the Premises to the Landlord on or before the Break Date, free from all occupiers, subtenants and other persons claiming through or under the Tenant;",
            S["body_indent2"],
        )
    )
    story.append(
        Paragraph(
            '(c) the Tenant pays to the Landlord on or before the Break Date a sum equivalent to six (6) months\' rent at the rate payable immediately before the Break Date (the <b>"Break Premium"</b>), such payment to be made by way of cleared funds.',
            S["body_indent2"],
        )
    )
    story.append(
        Paragraph(
            "8.3.2 If any of the conditions in clause 8.3.1 are not satisfied, the Break Notice shall be of no effect and this Lease shall continue in full force and effect.",
            S["body_indent"],
        )
    )

    # --- Section 9 — Indemnity ---
    story.append(Paragraph("Section 9 — Indemnity", S["heading1"]))
    story.append(Paragraph("9.1 General Indemnity", S["heading2"]))
    story.append(
        Paragraph(
            "9.1.1 The Tenant shall indemnify and keep indemnified the Landlord against all actions, proceedings, claims, demands, losses, costs, expenses, damages and liability (including professional fees and Value Added Tax thereon) arising directly or indirectly out of: (a) any act, omission or negligence of the Tenant or any person at the Premises with the express or implied authority of the Tenant; (b) any breach or non-observance by the Tenant of the covenants, conditions and other provisions of this Lease; or (c) the exercise or purported exercise by the Tenant of any of the rights granted to the Tenant under this Lease.",
            S["body_indent"],
        )
    )
    story.append(Paragraph("9.2 Environmental Indemnity", S["heading2"]))
    story.append(
        Paragraph(
            "9.2.1 The Tenant shall indemnify and keep indemnified the Landlord against all claims, costs and liabilities arising from or in connection with any contamination of or damage to the environment (including the Premises, the Building, the air, soil, groundwater, surface water and any living organism) caused by or attributable to the acts, omissions or operations of the Tenant or any person at the Premises with the express or implied authority of the Tenant.",
            S["body_indent"],
        )
    )
    story.append(
        Paragraph(
            "9.2.2 Without prejudice to the generality of clause 9.2.1, the Tenant's environmental indemnity shall include liability under Part IIA of the Environmental Protection Act 1990, the Water Resources Act 1991, and any other environmental legislation in force from time to time.",
            S["body_indent"],
        )
    )

    # --- Section 10 — Dispute Resolution ---
    story.append(Paragraph("Section 10 — Dispute Resolution", S["heading1"]))
    story.append(Paragraph("10.1 Mediation", S["heading2"]))
    story.append(
        Paragraph(
            '10.1.1 In the event of any dispute or difference arising out of or in connection with this Lease, including any question regarding its existence, validity or termination (a <b>"Dispute"</b>), the parties shall first attempt to resolve the Dispute by mediation in accordance with the Centre for Effective Dispute Resolution (CEDR) Model Mediation Procedure. To initiate mediation, a party must give written notice to the other party requesting mediation (a <b>"Mediation Notice"</b>).',
            S["body_indent"],
        )
    )
    story.append(
        Paragraph(
            "10.1.2 The mediation shall commence within twenty-eight (28) days of the Mediation Notice. If the Dispute is not resolved within fifty-six (56) days of the Mediation Notice (or such longer period as the parties may agree in writing), either party may refer the Dispute to arbitration in accordance with clause 10.2.",
            S["body_indent"],
        )
    )
    story.append(Paragraph("10.2 Arbitration", S["heading2"]))
    story.append(
        Paragraph(
            "10.2.1 Any Dispute not resolved by mediation in accordance with clause 10.1 shall be referred to and finally resolved by arbitration under the Arbitration Act 1996. The arbitral tribunal shall consist of a single arbitrator to be appointed by agreement between the parties or, in default of agreement within fourteen (14) days, by the President for the time being of the Chartered Institute of Arbitrators.",
            S["body_indent"],
        )
    )
    story.append(
        Paragraph(
            "10.2.2 The seat of the arbitration shall be London, England. The language of the arbitration shall be English. The arbitrator's award shall be final and binding on the parties.",
            S["body_indent"],
        )
    )
    story.append(
        Paragraph(
            "10.2.3 Nothing in this clause 10 shall prevent either party from applying to any court of competent jurisdiction for injunctive or other interim relief.",
            S["body_indent"],
        )
    )

    # --- Execution ---
    story.append(Spacer(1, 14 * mm))
    story.append(
        Paragraph(
            "IN WITNESS WHEREOF the parties hereto have executed this Lease as a deed on the date first above written.",
            S["body"],
        )
    )
    story.append(Spacer(1, 12 * mm))
    story.append(
        Paragraph(
            "EXECUTED as a DEED by<br/>"
            "<b>BISHOPSGATE PROPERTY HOLDINGS LIMITED</b><br/>"
            "acting by a director and the company secretary:",
            S["body"],
        )
    )
    story.append(Spacer(1, 8 * mm))
    story.append(Paragraph("Director: ____________________________", S["body"]))
    story.append(Paragraph("Secretary: ____________________________", S["body"]))
    story.append(Spacer(1, 10 * mm))
    story.append(
        Paragraph(
            "EXECUTED as a DEED by<br/>"
            "<b>MERIDIAN CONSULTING GROUP LLP</b><br/>"
            "acting by two members:",
            S["body"],
        )
    )
    story.append(Spacer(1, 8 * mm))
    story.append(Paragraph("Member: ____________________________", S["body"]))
    story.append(Paragraph("Member: ____________________________", S["body"]))

    doc.build(story, onFirstPage=_on_page, onLaterPages=_on_page)
    print(f"  Created: {path}")


# ===========================================================================
# DOCUMENT 2 — Title Report
# ===========================================================================


def _build_title_report():
    path = os.path.join(OUTPUT_DIR, "title-report-lot-7.pdf")
    doc = SimpleDocTemplate(
        path,
        pagesize=A4,
        topMargin=28 * mm,
        bottomMargin=25 * mm,
        leftMargin=25 * mm,
        rightMargin=25 * mm,
    )

    def _on_page(canvas, doc):
        _header_footer(canvas, doc, "Official Title Report — LN782451")

    story = []
    S = STYLES

    # --- Header ---
    story.append(Spacer(1, 8 * mm))
    story.append(Paragraph("OFFICIAL TITLE REPORT", S["title"]))
    story.append(Spacer(1, 4 * mm))

    # -- Horizontal rule --
    story.append(
        Table(
            [[""]],
            colWidths=[A4[0] - 50 * mm],
            rowHeights=[0.5 * mm],
            style=TableStyle([("LINEABOVE", (0, 0), (-1, -1), 1, colors.black)]),
        )
    )
    story.append(Spacer(1, 6 * mm))

    # --- Core details table ---
    data = [
        [
            Paragraph("<b>Title Number:</b>", S["table_header"]),
            Paragraph("LN782451", S["table_body"]),
        ],
        [
            Paragraph("<b>Edition Date:</b>", S["table_header"]),
            Paragraph("22 November 2023", S["table_body"]),
        ],
        [
            Paragraph("<b>Title Type:</b>", S["table_header"]),
            Paragraph("Absolute Freehold", S["table_body"]),
        ],
    ]
    t = Table(data, colWidths=[55 * mm, 100 * mm])
    t.setStyle(
        TableStyle(
            [
                ("VALIGN", (0, 0), (-1, -1), "TOP"),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 6),
            ]
        )
    )
    story.append(t)
    story.append(Spacer(1, 6 * mm))

    # --- Property Description ---
    story.append(Paragraph("Property Description", S["heading1"]))
    story.append(
        Paragraph(
            "The land and buildings known as Lot 7, Victoria Park Estate, 42-48 Victoria Park Road, London E9 7HD, as shown edged with red on the title plan filed at the Land Registry.",
            S["body"],
        )
    )
    story.append(
        Paragraph(
            "The property comprises a freehold parcel of land with an approximate area of 0.34 hectares (0.84 acres). The site is roughly rectangular in shape and is bounded to the north by Victoria Park Road, to the east by residential properties on Cadogan Terrace, to the south by the rear gardens of properties on Wick Road, and to the west by an access road serving the Victoria Park Estate.",
            S["body"],
        )
    )
    story.append(
        Paragraph(
            "The property currently contains a two-storey detached commercial building of approximately 1,200 square metres gross internal area, together with associated hardstanding and landscaped areas.",
            S["body"],
        )
    )

    # --- Registered Owner ---
    story.append(Paragraph("Registered Owner", S["heading1"]))
    story.append(
        Paragraph(
            "<b>Victoria Park Developments Ltd</b> (Company No. 08234571)<br/>"
            "Registered office: 17 Hackney Road, London E2 7NX",
            S["body"],
        )
    )
    story.append(
        Paragraph(
            "Registered as proprietor on <b>14 March 2019</b>.",
            S["body"],
        )
    )

    # --- Price Paid ---
    story.append(Paragraph("Price Paid / Value Stated", S["heading1"]))
    story.append(
        Paragraph(
            "The price stated on the transfer dated 14 March 2019 was <b>£4,250,000</b> (Four Million Two Hundred and Fifty Thousand Pounds).",
            S["body"],
        )
    )

    # --- Charges Register ---
    story.append(Paragraph("Charges Register", S["heading1"]))
    charges_data = [
        [
            Paragraph("<b>Entry</b>", S["table_header"]),
            Paragraph("<b>Date</b>", S["table_header"]),
            Paragraph("<b>Details</b>", S["table_header"]),
        ],
        [
            Paragraph("1", S["table_body"]),
            Paragraph("15 March 2019", S["table_body"]),
            Paragraph(
                "LEGAL CHARGE dated 15 March 2019 in favour of <b>Barclays Bank PLC</b> "
                "(Company No. 01026167), registered office at 1 Churchill Place, London E14 5HP. "
                "The charge secures further advances. NOTE: Copy of charge filed under reference LN782451/1.",
                S["table_body"],
            ),
        ],
        [
            Paragraph("2", S["table_body"]),
            Paragraph("1 June 1952", S["table_body"]),
            Paragraph(
                "RESTRICTIVE COVENANT dated 1 June 1952 contained in a conveyance of the land in this title "
                "by the London County Council to Herbert William Marsh. See details under Restrictive Covenants below.",
                S["table_body"],
            ),
        ],
    ]
    ct = Table(charges_data, colWidths=[15 * mm, 30 * mm, 110 * mm])
    ct.setStyle(
        TableStyle(
            [
                ("VALIGN", (0, 0), (-1, -1), "TOP"),
                ("GRID", (0, 0), (-1, -1), 0.5, colors.grey),
                ("BACKGROUND", (0, 0), (-1, 0), HexColor("#E8E8E8")),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 6),
                ("TOPPADDING", (0, 0), (-1, -1), 4),
            ]
        )
    )
    story.append(ct)

    # --- Restrictive Covenants ---
    story.append(Paragraph("Restrictive Covenants", S["heading1"]))
    story.append(
        Paragraph(
            "The following restrictive covenants are contained in the conveyance dated 1 June 1952 and are binding on the registered proprietor and successors in title:",
            S["body"],
        )
    )
    story.append(
        Paragraph(
            "(i) <b>Covenant not to use for industrial purposes:</b> The property shall not at any time be used for any industrial, manufacturing or heavy commercial purpose, nor for any purpose which may be or become a nuisance, annoyance or disturbance to the owners or occupiers of neighbouring properties.",
            S["body_indent"],
        )
    )
    story.append(
        Paragraph(
            "(ii) <b>Covenant to maintain boundary fencing:</b> The proprietor shall at all times maintain in good repair boundary fencing to a minimum height of 1.8 metres (6 feet) along all boundaries of the property which adjoin residential properties.",
            S["body_indent"],
        )
    )
    story.append(
        Paragraph(
            "(iii) <b>Covenant not to build above 4 storeys:</b> No building or structure shall be erected on the property which exceeds four (4) storeys in height, measured from the existing ground level at the date of the conveyance.",
            S["body_indent"],
        )
    )

    # --- Easements ---
    story.append(Paragraph("Easements", S["heading1"]))
    story.append(
        Paragraph(
            "(a) <b>Right of way:</b> The property has the benefit of a right of way at all times and for all purposes with or without vehicles over the access road to the north of the property (coloured brown on the title plan) leading to Victoria Park Road.",
            S["body_indent"],
        )
    )
    story.append(
        Paragraph(
            "(b) <b>Drainage easement:</b> There exists a drainage easement for a shared sewer running in a north-easterly to south-westerly direction across the property (shown by a broken blue line on the title plan). The registered proprietor is obliged to permit access for maintenance and repair of the sewer by the relevant statutory undertaker and by the owners of adjoining properties having the benefit of the easement.",
            S["body_indent"],
        )
    )

    # --- Noted Entries ---
    story.append(Paragraph("Noted Entries", S["heading1"]))
    story.append(
        Paragraph(
            "(a) <b>Planning permission B/2023/4521:</b> Planning permission was granted by the London Borough of Hackney on 8 September 2023 for the demolition of the existing commercial building and the erection of a residential development comprising 48 residential units (12 x 1-bed, 24 x 2-bed, 12 x 3-bed) in two blocks of 3 and 4 storeys respectively, together with associated landscaping, car parking (24 spaces) and cycle storage.",
            S["body_indent"],
        )
    )
    story.append(
        Paragraph(
            "(b) <b>Section 106 Agreement:</b> A Section 106 Agreement (Town and Country Planning Act 1990) dated 12 October 2023 between the London Borough of Hackney (1), Victoria Park Developments Ltd (2), and Barclays Bank PLC (3) has been registered against the title. Key obligations include: provision of 35% affordable housing (17 units), contribution of £185,000 towards local highway improvements, contribution of £75,000 towards public open space, and a commitment to enter into a local employment agreement during the construction phase.",
            S["body_indent"],
        )
    )

    # --- Cautions and Restrictions ---
    story.append(Paragraph("Cautions and Restrictions", S["heading1"]))
    story.append(
        Paragraph(
            "There are no cautions, restrictions on disposition, or inhibitions registered against this title.",
            S["body"],
        )
    )

    # --- Certification ---
    story.append(Spacer(1, 10 * mm))
    story.append(
        Table(
            [[""]],
            colWidths=[A4[0] - 50 * mm],
            rowHeights=[0.5 * mm],
            style=TableStyle([("LINEABOVE", (0, 0), (-1, -1), 1, colors.black)]),
        )
    )
    story.append(Spacer(1, 4 * mm))
    story.append(
        Paragraph(
            "This report reflects the entries on the register as at 12:00 on 22 November 2023. "
            "It is not a copy of the register and may not be relied upon for the purposes of Section 67 of the Land Registration Act 2002.",
            S["body"],
        )
    )
    story.append(Spacer(1, 6 * mm))
    story.append(
        Paragraph(
            "Issued by: <b>HM Land Registry</b><br/>"
            "Trafalgar House, 1 Bedford Park, Croydon CR0 2AQ",
            S["body"],
        )
    )

    doc.build(story, onFirstPage=_on_page, onLaterPages=_on_page)
    print(f"  Created: {path}")


# ===========================================================================
# DOCUMENT 3 — Environmental Assessment
# ===========================================================================


def _build_environmental():
    path = os.path.join(OUTPUT_DIR, "environmental-assessment-manchester.pdf")
    doc = SimpleDocTemplate(
        path,
        pagesize=A4,
        topMargin=28 * mm,
        bottomMargin=25 * mm,
        leftMargin=25 * mm,
        rightMargin=25 * mm,
    )

    def _on_page(canvas, doc):
        _header_footer(
            canvas, doc, "Phase I Environmental Site Assessment — 15-21 Deansgate, Manchester"
        )

    story = []
    S = STYLES

    # --- Title page ---
    story.append(Spacer(1, 50 * mm))
    story.append(Paragraph("PHASE I ENVIRONMENTAL SITE ASSESSMENT", S["title"]))
    story.append(Spacer(1, 10 * mm))
    story.append(Paragraph("<b>15-21 Deansgate, Manchester M3 4FN</b>", S["center"]))
    story.append(Spacer(1, 14 * mm))
    story.append(Paragraph("Prepared for:", S["center"]))
    story.append(Paragraph("<b>Manchester Property Holdings Ltd</b>", S["center"]))
    story.append(Spacer(1, 8 * mm))
    story.append(Paragraph("Prepared by:", S["center"]))
    story.append(
        Paragraph(
            "<b>Greenfield Environmental Consultants Ltd</b><br/>"
            "Quay House, Quay Street, Manchester M3 3JE",
            S["center"],
        )
    )
    story.append(Spacer(1, 12 * mm))
    story.append(Paragraph("Report Reference: GEC/2024/0142", S["center"]))
    story.append(Paragraph("Date: 15 January 2024", S["center"]))
    story.append(Spacer(1, 20 * mm))
    story.append(
        Paragraph(
            "Classification: CONFIDENTIAL",
            ParagraphStyle(
                "ConfStyle",
                fontName="Helvetica-Bold",
                fontSize=11,
                alignment=TA_CENTER,
                textColor=HexColor("#CC0000"),
            ),
        )
    )
    story.append(PageBreak())

    # --- 1. Executive Summary ---
    story.append(Paragraph("1. Executive Summary", S["heading1"]))
    story.append(
        Paragraph(
            'Greenfield Environmental Consultants Ltd ("GEC") was commissioned by Manchester Property Holdings Ltd (the "Client") to undertake a Phase I Environmental Site Assessment of the property situated at 15-21 Deansgate, Manchester M3 4FN (the "Site"). The assessment was carried out in accordance with BS 10175:2011+A2:2017 (Investigation of Potentially Contaminated Sites — Code of Practice) and CIRIA C552 (Contaminated Land Risk Assessment: A Guide to Good Practice).',
            S["body"],
        )
    )
    story.append(
        Paragraph(
            "The Site comprises an area of approximately 0.28 hectares (0.69 acres) and is currently occupied by a vacant four-storey commercial building of brick construction dating from approximately 1920. Historical use of the Site includes textile warehousing (1920-1975), printing works (1975-2002), and general office accommodation (2002-2019). The Site has been vacant since 2019.",
            S["body"],
        )
    )
    story.append(
        Paragraph(
            "Based on the findings of this assessment, the overall risk classification for the Site is assessed as <b>LOW to MODERATE</b>. The principal areas of concern are: (i) potential soil and groundwater contamination associated with the former printing works use (solvents and inks); and (ii) an underground storage tank of unknown status identified from historical records. A Phase II intrusive investigation is recommended to further characterise these risks.",
            S["body"],
        )
    )

    # --- 2. Site Description ---
    story.append(Paragraph("2. Site Description", S["heading1"]))
    story.append(
        Paragraph(
            "2.1 The Site is located at 15-21 Deansgate, Manchester M3 4FN, in the city centre of Manchester, within the administrative boundary of Manchester City Council. The national grid reference for the approximate centre of the Site is SJ 8365 9812.",
            S["body"],
        )
    )
    story.append(
        Paragraph(
            "2.2 The Site has an approximate area of 0.28 hectares (0.69 acres) and is roughly rectangular in plan. It is bounded to the north by Deansgate, to the east by a six-storey office building, to the south by a service yard and rear elevations of properties fronting St Mary's Street, and to the west by a narrow pedestrian alleyway and a multi-storey car park.",
            S["body"],
        )
    )
    story.append(
        Paragraph(
            "2.3 The Site is currently occupied by a single four-storey building of load-bearing brick construction with a flat roof. The building dates from approximately 1920, with internal alterations carried out at various times (most recently in 2002). The gross internal area is approximately 2,400 square metres. The building has been vacant and secured since 2019.",
            S["body"],
        )
    )
    story.append(
        Paragraph(
            "2.4 The historical uses of the Site are summarised as follows:",
            S["body"],
        )
    )
    uses = [
        "1920 - 1975: Textile warehouse (storage and distribution of cotton and wool goods)",
        "1975 - 2002: Printing works (commercial printing, lithographic and offset processes)",
        "2002 - 2019: General office accommodation (occupied by various tenants)",
        "2019 - present: Vacant",
    ]
    for u in uses:
        story.append(Paragraph(f"&bull; {u}", S["body_indent"]))

    # --- 3. Environmental Setting ---
    story.append(Paragraph("3. Environmental Setting", S["heading1"]))
    story.append(
        Paragraph(
            "3.1 <b>Geology:</b> The published geological mapping (British Geological Survey 1:50,000 Sheet 85, Manchester) indicates that the Site is underlain by superficial deposits of glacial till (boulder clay) to a depth of approximately 5-8 metres, overlying bedrock of the Sherwood Sandstone Group (Chester Pebble Beds Formation). The glacial till typically comprises stiff to very stiff brown sandy clay with occasional gravel and cobbles.",
            S["body"],
        )
    )
    story.append(
        Paragraph(
            "3.2 <b>Hydrogeology:</b> The Sherwood Sandstone Group is classified by the Environment Agency as a Principal Aquifer, representing a regionally significant groundwater resource. The overlying glacial till is classified as a Secondary (Undifferentiated) Aquifer, which may locally provide some protection to the underlying principal aquifer. Groundwater flow direction is anticipated to be generally westward towards the River Irwell.",
            S["body"],
        )
    )
    story.append(
        Paragraph(
            "3.3 <b>Hydrology:</b> The nearest significant surface watercourse is the River Irwell, located approximately 180 metres to the north-west of the Site. The River Irwell at this location is classified as a Main River under the jurisdiction of the Environment Agency. The Bridgewater Canal is located approximately 350 metres to the south of the Site.",
            S["body"],
        )
    )
    story.append(
        Paragraph(
            "3.4 <b>Flood risk:</b> The Environment Agency Flood Map for Planning indicates that the Site is located within Flood Zone 2, meaning that there is a medium probability of flooding (between 1 in 100 and 1 in 1,000 annual probability of river flooding). The Site is not within an area benefiting from flood defences.",
            S["body"],
        )
    )

    # --- 4. Regulatory Review ---
    story.append(Paragraph("4. Regulatory Review", S["heading1"]))
    story.append(Paragraph("4.1 Contaminated Land Register", S["heading2"]))
    story.append(
        Paragraph(
            "4.1.1 A search of Manchester City Council's Contaminated Land Register (maintained under Part IIA of the Environmental Protection Act 1990) confirms that the Site has not been determined as contaminated land. No remediation notices, charging notices or appeals relating to the Site were identified.",
            S["body_indent"],
        )
    )
    story.append(Paragraph("4.2 Historical Pollution Incidents", S["heading2"]))
    story.append(
        Paragraph(
            "4.2.1 A review of Environment Agency records identified two historical pollution incidents within a 250-metre radius of the Site:",
            S["body_indent"],
        )
    )
    story.append(
        Paragraph(
            "(a) <b>Fuel spillage (1987):</b> A release of approximately 500 litres of diesel fuel from a road tanker on Deansgate, approximately 120 metres north-east of the Site. The incident was responded to by the then National Rivers Authority and the spillage was contained and cleaned up. The case was closed in 1988 with no further action required.",
            S["body_indent2"],
        )
    )
    story.append(
        Paragraph(
            "(b) <b>Solvent discharge (1993):</b> An unauthorised discharge of chlorinated solvents to surface water drainage from a dry cleaning premises on Bridge Street, approximately 200 metres north-west of the Site. Remedial action was taken by the operator under the supervision of the Environment Agency. Subsequent monitoring confirmed that groundwater quality had returned to acceptable levels. The case was closed in 1997.",
            S["body_indent2"],
        )
    )
    story.append(Paragraph("4.3 Air Quality", S["heading2"]))
    story.append(
        Paragraph(
            "4.3.1 The Site is located within the Greater Manchester Air Quality Management Area (AQMA), declared in 2016 for exceedances of annual mean nitrogen dioxide (NO2) concentrations. The AQMA encompasses the majority of the city centre and major transport corridors. Manchester City Council has published an Air Quality Action Plan (updated 2022) setting out measures to achieve compliance with national air quality objectives.",
            S["body_indent"],
        )
    )
    story.append(Paragraph("4.4 Environmental Permits", S["heading2"]))
    story.append(
        Paragraph(
            "4.4.1 A search of the Environment Agency's public register of environmental permits confirms that there are no current or historical environmental permits (including waste management licences, integrated pollution control authorisations, or pollution prevention and control permits) associated with the Site.",
            S["body_indent"],
        )
    )

    # --- 5. Historical Use Review ---
    story.append(Paragraph("5. Historical Use Review", S["heading1"]))
    story.append(Paragraph("5.1 Ordnance Survey Map Regression", S["heading2"]))
    story.append(
        Paragraph(
            "5.1.1 A review of historical Ordnance Survey maps from 1890 to the present day confirms that the Site and surrounding area have been in continuous commercial and mixed-use development since at least the late 19th century. Key observations from the map regression are:",
            S["body_indent"],
        )
    )
    map_entries = [
        "1890 (1:2,500): Site shown occupied by commercial buildings. Surrounding area comprises dense urban development, predominantly warehouses and workshops.",
        "1922 (1:2,500): Current building footprint visible. Annotated as 'Warehouse'. No significant changes to surrounding area.",
        "1955 (1:2,500): No significant changes to Site. Former gasworks shown 400m to the south (now redeveloped).",
        "1977 (1:1,250): Building annotated as 'Works'. Loading bay visible to the rear (south).",
        "1995 (1:1,250): No change to building footprint. Adjacent multi-storey car park to the west now shown.",
        "2010 (1:1,250): No change to building footprint. Significant new development in surrounding area.",
    ]
    for me in map_entries:
        story.append(Paragraph(f"&bull; {me}", S["body_indent2"]))

    story.append(Paragraph("5.2 Former Printing Works (1975-2002)", S["heading2"]))
    story.append(
        Paragraph(
            "5.2.1 The use of the Site as a printing works between 1975 and 2002 represents the primary contamination concern identified in this assessment. Commercial printing operations of this era typically involved the use of a range of potentially contaminative substances, including:",
            S["body_indent"],
        )
    )
    contaminants = [
        "Petroleum-based inks and pigments (containing heavy metals including lead, chromium, cadmium and barium)",
        "Organic solvents used in cleaning and degreasing processes (including toluene, xylene, methyl ethyl ketone, and isopropanol)",
        "Chlorinated solvents (potentially including trichloroethylene and perchloroethylene)",
        "Photographic developing chemicals (silver, hydroquinone)",
        "Printing plate chemicals (aluminium, zinc, phosphoric acid)",
    ]
    for c in contaminants:
        story.append(Paragraph(f"&bull; {c}", S["body_indent2"]))
    story.append(
        Paragraph(
            "5.2.2 Spillages and leaks of these substances during the 27-year period of printing operations may have resulted in contamination of the underlying soil and, potentially, groundwater.",
            S["body_indent"],
        )
    )

    story.append(Paragraph("5.3 Underground Storage Tank", S["heading2"]))
    story.append(
        Paragraph(
            "5.3.1 A building survey report dated June 1985 (prepared by Hargreaves & Mitchell Chartered Surveyors in connection with a proposed extension to the building) makes reference to an underground storage tank located in the rear yard area to the south of the building. The report describes the tank as being used for the storage of heating oil and having an estimated capacity of 5,000 litres.",
            S["body_indent"],
        )
    )
    story.append(
        Paragraph(
            "5.3.2 No records have been identified to confirm whether the underground storage tank was decommissioned, removed or filled in situ. The current status of the tank is therefore unknown. Underground storage tanks, particularly those of this age, present a significant risk of leakage and soil and groundwater contamination from stored petroleum products.",
            S["body_indent"],
        )
    )

    # --- 6. Findings and Recommendations ---
    story.append(Paragraph("6. Findings and Recommendations", S["heading1"]))
    story.append(
        Paragraph(
            "6.1 Risk of Soil Contamination from Former Printing Works",
            S["heading2"],
        )
    )
    story.append(
        Paragraph(
            "6.1.1 There is a <b>moderate risk</b> that soil beneath and adjacent to the building has been contaminated by substances associated with the former printing works use (1975-2002). The risk is assessed as moderate rather than high because: (a) the building has a solid ground floor construction which may have limited direct infiltration; (b) the glacial till beneath the Site provides some attenuation and retardation of contaminant migration; and (c) there are no known specific incidents of significant spillage.",
            S["body_indent"],
        )
    )
    story.append(
        Paragraph(
            "6.1.2 The potential contaminants of concern include volatile organic compounds (VOCs), semi-volatile organic compounds (SVOCs), heavy metals, and total petroleum hydrocarbons (TPH). These contaminants may present risks to human health (particularly via inhalation of vapours in enclosed spaces) and to controlled waters (via migration through the glacial till to the underlying principal aquifer).",
            S["body_indent"],
        )
    )

    story.append(Paragraph("6.2 Underground Storage Tank Investigation", S["heading2"]))
    story.append(
        Paragraph(
            "6.2.1 The underground storage tank identified in the 1985 survey report requires further investigation to determine its current status (in situ or removed), its contents (if any), and whether leakage has occurred. If the tank remains in situ, it is likely to require decommissioning and removal in accordance with current best practice (CIRIA C736, Containment Systems for the Prevention of Pollution).",
            S["body_indent"],
        )
    )

    story.append(
        Paragraph(
            "6.3 Recommendation: Phase II Intrusive Investigation",
            S["heading2"],
        )
    )
    story.append(
        Paragraph(
            "6.3.1 A Phase II intrusive investigation is recommended to characterise the nature and extent of any contamination at the Site. The investigation should include the following scope as a minimum:",
            S["body_indent"],
        )
    )
    scope = [
        "Soil sampling from a minimum of six (6) boreholes drilled to a depth of at least 6 metres (to penetrate through the glacial till and into the upper Sherwood Sandstone). Boreholes should be targeted on areas of greatest potential contamination, including the former printing works area, the loading bay, and the vicinity of the underground storage tank.",
        "Installation of two (2) groundwater monitoring wells (one up-gradient and one down-gradient of the Site) to enable assessment of groundwater quality and determination of groundwater flow direction and gradient.",
        "Chemical analysis of soil and groundwater samples for a comprehensive suite of determinands including: heavy metals, VOCs, SVOCs, TPH (CWG banded), polycyclic aromatic hydrocarbons (PAHs), polychlorinated biphenyls (PCBs), and pH.",
        "Ground gas monitoring (minimum 3 rounds over 6 weeks) to assess risks from methane, carbon dioxide and volatile organic compounds.",
        "Geophysical survey (ground penetrating radar) in the rear yard area to locate the underground storage tank.",
    ]
    for i, s in enumerate(scope, 1):
        story.append(Paragraph(f"({chr(96 + i)}) {s}", S["body_indent2"]))

    story.append(Paragraph("6.4 Estimated Cost of Phase II Investigation", S["heading2"]))
    story.append(
        Paragraph(
            "6.4.1 The estimated cost of the Phase II intrusive investigation as described in Section 6.3 above is in the range of <b>£15,000 to £25,000</b> (exclusive of VAT). This estimate is based on current market rates and includes: site investigation works (drilling, sampling, well installation), laboratory analysis, ground gas monitoring, geophysical survey, and the preparation of a Phase II investigation report with quantitative risk assessment.",
            S["body_indent"],
        )
    )
    story.append(
        Paragraph(
            "6.4.2 The estimate does not include any costs associated with further assessment or remediation which may be required depending on the findings of the Phase II investigation.",
            S["body_indent"],
        )
    )

    story.append(Paragraph("6.5 Estimated Remediation Costs", S["heading2"]))
    story.append(
        Paragraph(
            "6.5.1 In the event that the Phase II investigation confirms the presence of significant contamination, remediation works are likely to be required prior to any redevelopment of the Site. Based on the nature and scale of potential contamination identified in this Phase I assessment, remediation costs are estimated in the range of <b>£50,000 to £200,000</b> (exclusive of VAT), depending on the extent and severity of contamination.",
            S["body_indent"],
        )
    )
    story.append(
        Paragraph(
            "6.5.2 The lower end of this range (£50,000) would apply if contamination is limited to localised hotspots (e.g., around the underground storage tank) and can be addressed by targeted excavation and off-site disposal. The upper end (£200,000) would apply if more widespread contamination is identified requiring extensive soil remediation (e.g., in situ treatment or large-scale excavation) and/or groundwater remediation.",
            S["body_indent"],
        )
    )
    story.append(
        Paragraph(
            "6.5.3 These cost estimates are indicative only and should be refined following the completion of the Phase II intrusive investigation.",
            S["body_indent"],
        )
    )

    # --- Certification ---
    story.append(Spacer(1, 10 * mm))
    story.append(
        Table(
            [[""]],
            colWidths=[A4[0] - 50 * mm],
            rowHeights=[0.5 * mm],
            style=TableStyle([("LINEABOVE", (0, 0), (-1, -1), 1, colors.black)]),
        )
    )
    story.append(Spacer(1, 6 * mm))
    story.append(
        Paragraph(
            "This report has been prepared by Greenfield Environmental Consultants Ltd for the sole use of Manchester Property Holdings Ltd "
            "in connection with the proposed acquisition and redevelopment of the Site. No liability is accepted to any third party for the "
            "whole or any part of the contents of this report.",
            S["body"],
        )
    )
    story.append(Spacer(1, 8 * mm))
    story.append(
        Paragraph(
            "<b>Prepared by:</b> Dr. Sarah J. Whitmore, BSc (Hons), MSc, PhD, CEnv, CSci<br/>"
            "Principal Environmental Consultant<br/>"
            "Greenfield Environmental Consultants Ltd",
            S["body"],
        )
    )
    story.append(Spacer(1, 6 * mm))
    story.append(
        Paragraph(
            "<b>Reviewed by:</b> Mr. James R. Thornton, BSc (Hons), MSc, FGS, SiLC<br/>"
            "Technical Director<br/>"
            "Greenfield Environmental Consultants Ltd",
            S["body"],
        )
    )
    story.append(Spacer(1, 8 * mm))
    story.append(Paragraph("Date: 15 January 2024", S["body"]))

    doc.build(story, onFirstPage=_on_page, onLaterPages=_on_page)
    print(f"  Created: {path}")


# ===========================================================================
# Main
# ===========================================================================

if __name__ == "__main__":
    print("Generating sample legal documents...")
    _build_lease()
    _build_title_report()
    _build_environmental()
    print(f"\nAll documents saved to: {OUTPUT_DIR}")
