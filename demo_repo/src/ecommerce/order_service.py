"""Order processing and business logic service."""

from typing import List
from ecommerce.models import Customer, Item, Order, OrderLine


class OrderService:
    """Handles order creation, discount calculations, and cancellations."""

    TAX_RATE: float = 0.08  # 8% sales tax
    VIP_DISCOUNT_PERCENT: float = 0.15  # 15% discount for VIPs

    def create_order(self, order_id: str, customer: Customer, items_with_qty: List[tuple[Item, int]]) -> Order:
        """Construct an order from catalog items and quantities, validating stock."""
        lines: List[OrderLine] = []
        for item, qty in items_with_qty:
            if qty <= 0:
                raise ValueError(f"Quantity must be greater than zero for item {item.id}")
            if item.stock_quantity < qty:
                raise ValueError(f"Insufficient stock for item {item.name}. Available: {item.stock_quantity}, Requested: {qty}")

            # Reserve stock
            item.stock_quantity -= qty
            line_total = round(item.unit_price * qty, 2)
            lines.append(OrderLine(item=item, quantity=qty, line_total=line_total))

        order = Order(id=order_id, customer=customer, lines=lines)
        return self.calculate_order_total(order)

    def calculate_order_total(self, order: Order) -> Order:
        """Calculate subtotal, VIP discounts, tax, and final amount."""
        subtotal = sum(line.line_total for line in order.lines)
        order.subtotal = round(subtotal, 2)

        if order.customer.is_vip:
            order.discount_amount = round(order.subtotal * self.VIP_DISCOUNT_PERCENT, 2)
        else:
            order.discount_amount = 0.0

        taxable_amount = max(0.0, order.subtotal - order.discount_amount)
        order.tax_amount = round(taxable_amount * self.TAX_RATE, 2)
        order.total_amount = round(taxable_amount + order.tax_amount, 2)
        return order

    def cancel_order(self, order: Order) -> bool:
        """Cancel an order and return reserved items to inventory.

        BUG 2 (Deliberate): Does not check if order is already CANCELLED, causing
        inventory to be duplicate-restored on repeated cancellation.
        """
        # BUG: Missing idempotency check for order.status == 'CANCELLED'
        # Should raise ValueError if already CANCELLED.

        for line in order.lines:
            line.item.stock_quantity += line.quantity

        order.status = "CANCELLED"
        return True
