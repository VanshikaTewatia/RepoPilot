"""Payment information validator."""

from datetime import datetime


class PaymentValidator:
    """Validates credit cards and billing inputs."""

    @staticmethod
    def validate_card_number(card_number: str) -> bool:
        """Validate card using Luhn algorithm."""
        clean = card_number.replace(" ", "").replace("-", "")
        if not clean.isdigit() or len(clean) < 13 or len(clean) > 19:
            return False

        total = 0
        reverse_digits = [int(d) for d in clean[::-1]]
        for idx, digit in enumerate(reverse_digits):
            if idx % 2 == 1:
                doubled = digit * 2
                total += doubled - 9 if doubled > 9 else doubled
            else:
                total += digit
        return total % 10 == 0

    @staticmethod
    def validate_expiry(month: int, year: int) -> bool:
        """Validate credit card expiration date against current year/month."""
        if month < 1 or month > 12:
            return False

        now = datetime.now()
        # Accept 2-digit or 4-digit years
        full_year = 2000 + year if year < 100 else year

        if full_year < now.year:
            return False
        if full_year == now.year and month < now.month:
            return False
        return True
