from calculator.utils import (
    add,
    calculate_discount,
    multiply,
    divide,
)


def test_add():
    assert add(2, 3) == 5


def test_discount():
    assert calculate_discount(100, 10) == 90


def test_multiply():
    assert multiply(4, 5) == 20


def test_divide():
    assert divide(10, 2) == 5


def test_divide_by_zero():
    try:
        divide(10, 0)
        assert False
    except ValueError:
        assert True