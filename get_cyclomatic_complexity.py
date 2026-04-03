
from radon.complexity import cc_visit

def cyclomatic_complexity_total(code):
    try:
        blocks = cc_visit(code)
        return sum(block.complexity for block in blocks)
    except Exception:
        return 0