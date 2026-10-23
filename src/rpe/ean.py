"""EAN-13 / ISBN-13 check digits."""

from __future__ import annotations


def check_digit(first12: str) -> int:
    s = sum(int(d) * (3 if i % 2 else 1) for i, d in enumerate(first12))
    return (10 - s % 10) % 10


def make(prefix: str, number: int) -> str:
    body = f"{prefix}{number:0{12 - len(prefix)}d}"
    return body + str(check_digit(body))


def valid(code: object) -> bool:
    if not isinstance(code, str) or len(code) != 13 or not code.isdigit():
        return False
    return check_digit(code[:12]) == int(code[12])
