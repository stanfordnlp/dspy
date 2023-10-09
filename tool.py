def calculator(expr):
    return eval(expr)

def prefix_calculator(expr):
    """
    Evaluate a prefix arithmetic expression.
    """
    stack = []
    for token in reversed(expr.split()):
        if token.isdigit():
            stack.append(int(token))
        else:
            op1 = stack.pop()
            op2 = stack.pop()
            if token == '+':
                stack.append(op1 + op2)
            elif token == '-':
                stack.append(op1 - op2)
            elif token == '*':
                stack.append(op1 * op2)
            elif token == '/':
                stack.append(op1 / op2)
    return stack.pop()


def test_calculator():
    expr1 = "1 + 2"
    assert calculator(expr1) == 3, f"Expected 3, got {calculator(expr1)}"
    expr2 = "1 + 2 * 3"
    assert calculator(expr2) == 7, f"Expected 7, got {calculator(expr2)}"
    expr3 = "(1 + 2) * 3"
    assert calculator(expr3) == 9, f"Expected 9, got {calculator(expr3)}"

def test_prefix_calculator():
    expr1 = "+ 1 2"
    assert prefix_calculator(expr1) == 3, f"Expected 3, got {prefix_calculator(expr1)}"
    expr2 = "+ 1 * 2 3"
    assert prefix_calculator(expr2) == 7, f"Expected 7, got {prefix_calculator(expr2)}"
    expr3 = "* + 1 2 3"
    assert prefix_calculator(expr3) == 9, f"Expected 9, got {prefix_calculator(expr3)}"

if __name__ == '__main__':
    test_calculator()
    test_prefix_calculator()
    print("All tests passed!")