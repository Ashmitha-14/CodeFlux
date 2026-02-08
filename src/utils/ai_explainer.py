
def generate_insights(stats):
    """
    Generates natural language explanations for code risk based on AST stats.
    inputs: stats dict {'complexity': int, 'max_depth': int, ...}
    returns: list of strings (insights)
    """
    insights = []
    
    # 1. Complexity Analysis
    cc = stats.get('complexity', 0)
    if cc > 10:
        insights.append(f"🔴 **High Cyclomatic Complexity ({cc})**: The code has many branching paths (if/for/while), making it hard to test and maintain.")
    elif cc > 5:
        insights.append(f"🟠 **Moderate Complexity ({cc})**: Consider refactoring large functions into smaller helpers.")
        
    # 2. Nesting Analysis
    depth = stats.get('max_depth', 0)
    if depth > 4:
        insights.append(f"🔴 **Deep Nesting ({depth} levels)**: Logic is buried deep inside loops/conditions. This increases cognitive load and bug risk.")
        
    # 3. Function/Class Analysis
    num_funcs = stats.get('num_functions', 0)
    lines = stats.get('lines_of_code', 0)
    
    if lines > 200 and num_funcs < 3:
        insights.append(f"🟠 **Monolithic Code**: File is long ({lines} loc) but has few functions. Consider breaking it down.")
        
    if not insights:
        insights.append("🟢 **Clean Code**: No structural risk factors detected.")
        
    return insights
