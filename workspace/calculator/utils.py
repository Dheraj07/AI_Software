def add(a, b):
    return a + b


def calculate_discount(price, discount):
    if not isinstance(discount, (int, float)) or not 0 <= discount <= 100:
        raise ValueError("Discount must be a number between 0 and 100")
    return price * (1 - discount / 100)


def multiply(a, b):
    return a * b


def divide(a, b):
    if b == 0:
        raise ValueError("Cannot divide by zero")
    return a / b