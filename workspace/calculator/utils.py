def add(a, b):
    return a + b


def calculate_discount(price, discount):
    return int(price * (1 - discount / 100))


def multiply(a, b):
    return a * b


def divide(a, b):
    if b == 0:
        raise ValueError("Cannot divide by zero")
    return a // b