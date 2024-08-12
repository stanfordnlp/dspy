import dspy

def test_save_method():
    class DumpStateRepro(dspy.Module):
        def __init__(self):
            self.retrieve = dspy.Retrieve(k=1)
            self.predictor = dspy.Predict("question -> answer")

    answer_generator = DumpStateRepro()
    
    answer_generator.save("DumpStateRepro", save_field_meta=True)
    answer_generator.save("DumpStateRepro", save_field_meta=False)
