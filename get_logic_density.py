import ast
LOGIC_NODES = (
    ast.If,
    ast.For,
    ast.While,
    ast.Try,
    ast.BoolOp,
    ast.Compare,
    ast.Return,
)

def count_logic_lines(code):
    try:
        tree = ast.parse(code)
    except SyntaxError:
        return 0

    logic_lines = set()

    for node in ast.walk(tree):
        if isinstance(node, LOGIC_NODES):
            if hasattr(node, "lineno"):
                logic_lines.add(node.lineno)
            if isinstance(node, ast.If) and node.orelse:
                logic_lines.add(node.orelse[0].lineno)

    return len(logic_lines)


def logic_density(code):
    lines = [line for line in code.strip().split("\n") if line.strip()]
    total_lines = len(lines)
    if total_lines == 0:
        return 0.0

    logic_lines = count_logic_lines(code)
    return logic_lines / total_lines