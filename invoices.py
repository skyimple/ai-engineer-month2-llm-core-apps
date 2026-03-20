INVOICES = [
    # Invoice 1: English, simple format
    """INVOICE

Invoice Number: INV-2024-001
Date: 2024-03-15
Due Date: 2024-04-15

Items:
- Widget A x 10 @ $25.00 each = $250.00
- Gadget B x 5 @ $50.00 each = $250.00
- Service C x 2 @ $100.00 each = $200.00

Total Amount: $700.00
""",

    # Invoice 2: Chinese format
    """发票

发票号码: INV-2024-002
开票日期: 2024年3月20日
到期日期: 2024年4月20日

商品明细:
- 产品X  数量: 5  单价: 100.00  金额: 500.00
- 产品Y  数量: 3  单价: 200.00  金额: 600.00
- 服务Z  数量: 1  单价: 300.00  金额: 300.00

价税合计: 1400.00
""",

    # Invoice 3: Table format
    """INVOICE

Invoice #: INV-2024-003
Issue Date: March 25, 2024
Payment Due: April 25, 2024

Description          Qty    Unit Price    Total
-------------------------------------------------
Item Alpha             8       15.00      120.00
Item Beta             12       22.50      270.00
Item Gamma             4       45.00      180.00
-------------------------------------------------
                                    TOTAL: $570.00
""",

    # Invoice 4: Mixed Chinese-English
    """INVOICE / 发票

Invoice No: INV-2024-004
Date: 2024-03-28
Due: 2024-04-28

Items:
1. Server License    Qty: 2    Price: 500.00    Subtotal: 1000.00
2. Support Plan       Qty: 1    Price: 250.00    Subtotal: 250.00
3. Training Course    Qty: 5    Price: 80.00     Subtotal: 400.00

Total Amount / 总计: 1650.00
""",

    # Invoice 5: Complex format with discount
    """========================================
         INVOICE
========================================
Invoice Number: INV-2024-005
Date: 2024-03-30
Due Date: April 30, 2024

----------------------------------------
Item Description    Qty    Rate    Amount
----------------------------------------
Product A            20    10.00   200.00
Product B            10    25.00   250.00
Product C             5    40.00   200.00
----------------------------------------
Subtotal: 650.00
Discount: 50.00
----------------------------------------
TOTAL DUE: 600.00
========================================
"""
]
