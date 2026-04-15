import ast
class DepthVisitor(ast.NodeVisitor):
    def __init__(self):
        self.current_depth = 0
        self.max_depth = 0

    def _enter(self):
        self.current_depth += 1
        self.max_depth = max(self.max_depth, self.current_depth)

    def _exit(self):
        self.current_depth -= 1

    def visit_If(self, node):
        self._enter()
        self.generic_visit(node)
        self._exit()

    def visit_For(self, node):
        self._enter()
        self.generic_visit(node)
        self._exit()

    def visit_While(self, node):
        self._enter()
        self.generic_visit(node)
        self._exit()

    def visit_Try(self, node):
        self._enter()
        self.generic_visit(node)
        self._exit()


def max_nesting_depth(code):
    try:
        tree = ast.parse(code)
    except SyntaxError:
        return 0
    visitor = DepthVisitor()
    visitor.visit(tree)
    return visitor.max_depth