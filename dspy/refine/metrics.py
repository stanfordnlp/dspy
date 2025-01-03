import inspect


class Metric:
    def __init__(self, name, description, fn):
        self.name = name
        self.description = description
        self.fn = fn

    def __call__(self, inputs, outputs):
        return self.fn(inputs, outputs)

    def __repr__(self):
        if self.description:
            return f"Metric name: {self.name}\nMetric description: {self.description}\n"
        else:
            return f"Metric name: {self.name}\nMetric function: {inspect.getsource(self.fn)}\n"


class BoolMetric(Metric):
    def __init__(self, name, description, fn):
        super().__init__(name, description, fn)
        self.type_description = "This is a bool metric, true if the metric looks good, false otherwise."

    def __repr__(self):
        return f"{super().__repr__()}\nMetric type:{self.type_description}"


class FloatMetric(Metric):
    def __init__(self, name, description, fn):
        super().__init__(name, description, fn)
        self.type_description = "This is a float metric, the higher the value the better."

    def __repr__(self):
        return f"{super().__repr__()}\nMetric type:{self.type_description}"
