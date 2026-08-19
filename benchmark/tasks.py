"""Evaluation benchmark tasks definition for RepoPilot."""

from dataclasses import dataclass
from typing import Dict, List, Optional


@dataclass
class BenchmarkTask:
    """Represents an evaluation benchmark task."""

    id: str
    title: str
    description: str
    target_file: str
    expected_patch: str
    start_line: int
    end_line: int
    test_file: str


BENCHMARK_TASKS: List[BenchmarkTask] = [
    BenchmarkTask(
        id="task-1-vip-discount",
        title="Fix VIP discount to apply to order subtotal",
        description="In OrderService.calculate_order_total, VIP discount should be calculated as 15% of the total order subtotal rather than the first item price.",
        target_file="src/ecommerce/order_service.py",
        expected_patch="""        if order.customer.is_vip:
            order.discount_amount = round(order.subtotal * self.VIP_DISCOUNT_PERCENT, 2)
        else:
            order.discount_amount = 0.0
""",
        start_line=39,
        end_line=45,
        test_file="tests/test_order_service.py::test_vip_customer_discount_across_entire_subtotal",
    ),
    BenchmarkTask(
        id="task-2-cancel-idempotency",
        title="Enforce cancel_order idempotency",
        description="In OrderService.cancel_order, check if order.status is already 'CANCELLED' and raise ValueError('Order is already cancelled') to prevent inventory double restoration.",
        target_file="src/ecommerce/order_service.py",
        expected_patch="""        if order.status == "CANCELLED":
            raise ValueError("Order is already cancelled")
""",
        start_line=59,
        end_line=60,
        test_file="tests/test_order_service.py::test_cancel_order_idempotency_raises_error",
    ),
    BenchmarkTask(
        id="task-3-payment-expiry-range",
        title="Validate card expiry month range",
        description="In PaymentValidator.validate_expiry, ensure month is strictly between 1 and 12 inclusive.",
        target_file="src/ecommerce/payment_validator.py",
        expected_patch="""        if month < 1 or month > 12:
            return False
""",
        start_line=24,
        end_line=25,
        test_file="tests/test_payment_validator.py::test_card_expiry_validation",
    ),
    BenchmarkTask(
        id="task-4-stock-reservation",
        title="Verify item stock reservation during order creation",
        description="In OrderService.create_order, ensure quantity must be positive and stock is correctly decremented.",
        target_file="src/ecommerce/order_service.py",
        expected_patch="""        for item, qty in items_with_qty:
            if qty <= 0:
                raise ValueError(f"Quantity must be greater than zero for item {item.id}")
""",
        start_line=15,
        end_line=17,
        test_file="tests/test_order_service.py::test_standard_customer_order",
    ),
    BenchmarkTask(
        id="task-5-tax-computation",
        title="Verify tax calculation on discounted amount",
        description="In OrderService.calculate_order_total, ensure taxable amount cannot be negative.",
        target_file="src/ecommerce/order_service.py",
        expected_patch="""        taxable_amount = max(0.0, order.subtotal - order.discount_amount)
        order.tax_amount = round(taxable_amount * self.TAX_RATE, 2)
        order.total_amount = round(taxable_amount + order.tax_amount, 2)
""",
        start_line=47,
        end_line=49,
        test_file="tests/test_order_service.py",
    ),
]
