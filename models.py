from pydantic import BaseModel
from datetime import date
from typing import List


class Item(BaseModel):
    description: str
    quantity: float
    unit_price: float
    total: float


class Invoice(BaseModel):
    invoice_number: str
    total_amount: float
    items: List[Item]
    due_date: date | None = None
