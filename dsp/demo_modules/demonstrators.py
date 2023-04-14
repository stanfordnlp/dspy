import dsp

def default_demonstrate(x, program, train, annotator, num_samples=7, num_demos=5):
    demos = dsp.sample(train, k=num_samples)
    x.demos = dsp.annotate_with_program(annotator)(demos, program, k=num_demos, return_all=True)
    return x

def add_demos(program, train, annotator, demonstrator=default_demonstrate):
    def progam_with_demonstrate(x: dsp.Example) -> dsp.Example:
        x = demonstrator(x, program, train, annotator)
        x = program(x)
        return x
    return progam_with_demonstrate
