class XOR:
    def __init__(self):
        self.name = "XOR"

    def result(self, x1, x2):
        return (x1 and not x2) or (not x1 and x2)


# file can be extended with more logic operators to make the network learn them