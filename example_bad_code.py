
def confusing_function(x, y):
    # This function has deep nesting and high complexity
    result = 0
    if x > 0:
        for i in range(10):
            if y < 5:
                while result < 100:
                    result += x * y
                    if result % 2 == 0:
                        print("Even")
                    else:
                        for j in range(5):
                            result += j
            else:
                result -= 1
    return result

class ComplexClass:
    def __init__(self):
        self.data = []

    def process(self):
        # Cyclomatic complexity here is high
        for item in self.data:
            if item:
                if item > 10:
                    if item < 20:
                        print("Range A")
                    elif item < 30:
                        print("Range B")
                else:
                    print("Low")
            else:
                print("None")
