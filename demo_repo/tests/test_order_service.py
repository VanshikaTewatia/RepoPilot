"""Unit tests for OrderService."""

import pytest
from ecommerce.models import Customer, Item
from ecommerce.order_service import OrderService


def test_standard_customer_order():
    """Test order creation for non-VIP customer with tax."""
    service = OrderService()
    cust = Customer(id="c1", name="Alice", email="alice@test.com", is_vip=False)
    item1 = Item(id="i1", name="Keyboard", unit_price=100.0, stock_quantity=10)
    item2 = Item(id="i2", name="Mouse", unit_price=50.0, stock_quantity=5)

    order = service.create_order("ord_1", cust, [(item1, 1), (item2, 2)])

    assert order.subtotal == 200.0
    assert order.discount_amount == 0.0
    assert order.tax_amount == 16.0  # 8% of 200
    assert order.total_amount == 216.0


def test_vip_customer_discount_across_entire_subtotal():
    """Test VIP customer receives 15% discount across entire order subtotal."""
    service = OrderService()
    vip_cust = Customer(id="c2", name="Bob VIP", email="bob@test.com", is_vip=True)
    item1 = Item(id="i1", name="Monitor", unit_price=200.0, stock_quantity=10)
    item2 = Item(id="i2", name="Cable", unit_price=20.0, stock_quantity=20)

    # 1 Monitor ($200) + 2 Cables ($40) = Subtotal $240
    order = service.create_order("ord_2", vip_cust, [(item1, 1), (item2, 2)])

    assert order.subtotal == 240.0
    # Expected discount: 15% of $240 = $36.00
    # (Buggy version only discounted $200 * 0.15 = $30.00)
    assert order.discount_amount == 36.0
    assert order.tax_amount == round((240.0 - 36.0) * 0.08, 2)  # $16.32
    assert order.total_amount == 204.0 + 16.32


def test_cancel_order_idempotency_raises_error():
    """Test that cancelling an already cancelled order raises ValueError."""
    service = OrderService()
    cust = Customer(id="c3", name="Charlie", email="charlie@test.com")
    item = Item(id="i3", name="Headphones", unit_price=50.0, stock_quantity=10)

    order = service.create_order("ord_3", cust, [(item, 2)])
    assert item.stock_quantity == 8

    # First cancellation succeeds
    service.cancel_order(order)
    assert order.status == "CANCELLED"
    assert item.stock_quantity == 10

    # Second cancellation must raise ValueError and not double-restore stock
    with pytest.raises(ValueError, match="already cancelled"):
        service.cancel_order(order)

    assert item.stock_quantity == 10
