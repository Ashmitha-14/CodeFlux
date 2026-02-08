
import ast
import networkx as nx

class CodeAnalyzer(ast.NodeVisitor):
    def __init__(self):
        self.stats = {
            "num_functions": 0,
            "num_classes": 0,
            "max_depth": 0,
            "complexity": 0,  # Cyclomatic complexity approximation
            "lines_of_code": 0
        }
        self.current_depth = 0

    def visit(self, node):
        self.current_depth += 1
        self.stats["max_depth"] = max(self.stats["max_depth"], self.current_depth)
        super().visit(node)
        self.current_depth -= 1

    def visit_FunctionDef(self, node):
        self.stats["num_functions"] += 1
        self.stats["complexity"] += 1  # Base complexity
        self.generic_visit(node)

    def visit_ClassDef(self, node):
        self.stats["num_classes"] += 1
        self.generic_visit(node)

    def visit_If(self, node):
        self.stats["complexity"] += 1
        self.generic_visit(node)

    def visit_For(self, node):
        self.stats["complexity"] += 1
        self.generic_visit(node)

    def visit_While(self, node):
        self.stats["complexity"] += 1
        self.generic_visit(node)

def parse_code(code_str):
    try:
        tree = ast.parse(code_str)
        analyzer = CodeAnalyzer()
        analyzer.visit(tree)
        analyzer.stats["lines_of_code"] = len(code_str.splitlines())
        return {
            "ast": tree,
            "stats": analyzer.stats
        }
    except SyntaxError as e:
        return {"error": str(e)}
