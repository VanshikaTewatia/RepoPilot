"""Unit tests for PaymentValidator."""

from ecommerce.payment_validator import PaymentValidator


def test_card_expiry_validation():
    """Test valid and expired cards."""
    # Future dates are valid
    assert PaymentValidator.validate_expiry(month=12, year=2030) is True
    assert PaymentValidator.validate_expiry(month=5, year=35) is True

    # Past dates are invalid
    assert PaymentValidator.validate_expiry(month=1, year=2020) is False
    assert PaymentValidator.validate_expiry(month=13, year=2030) is False
    assert PaymentValidator.validate_expiry(month=0, year=2030) is False
