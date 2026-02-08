
def factorial(n):
    # Recursive function - moderate complexity
    if n <= 1:
        return 1
    else:
        return n * factorial(n-1)

def compute_series(n):
    # Linear loop with a condition
    result = 0
    for i in range(n):
        if i % 2 == 0:
            result += i * 2
        else:
            result += i
    return result

class DataProcessor:
    def __init__(self, data):
        self.data = data

    def filter_data(self, threshold):
        # List comprehension - pythonic and generally clean
        return [x for x in self.data if x > threshold]
