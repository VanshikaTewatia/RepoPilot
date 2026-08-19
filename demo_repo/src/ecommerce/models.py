"""Domain models for e-commerce demo application."""

from dataclasses import dataclass, field
from typing import List, Optional


@dataclass
class Item:
    """Product catalog item."""
    id: str
    name: str
    unit_price: float
    stock_quantity: int


@dataclass
class Customer:
    """Customer account representation."""
    id: str
    name: str
    email: str
    loyalty_points: int = 0
    is_vip: bool = False


@dataclass
class OrderLine:
    """Individual line item in an order."""
    item: Item
    quantity: int
    line_total: float = 0.0


@dataclass
class Order:
    """Customer purchase order."""
    id: str
    customer: Customer
    lines: List[OrderLine] = field(default_factory=list)
    status: str = "PENDING"  # PENDING, PAID, SHIPPED, CANCELLED
    subtotal: float = 0.0
    discount_amount: float = 0.0
    tax_amount: float = 0.0
    total_amount: float = 0.0
