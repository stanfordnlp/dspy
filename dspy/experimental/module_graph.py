import dspy

try:
    import graphviz  # type: ignore

    graphviz_available = True
except ImportError:
    graphviz_available = False


class ModuleGraph:
    def __init__(self, module_name, module):
        if graphviz_available is False:
            raise ImportError(
                """Please install graphviz to use this feature.
                Run 'pip install graphviz'"""
            )

        self.graph = graphviz.Digraph(format="png")
        self.nodes = set()
        self.module_name = module_name
        self.module = module
        self.inspect_settings(dspy.settings)
        self.add_module(self.module_name, self.module)

    def inspect_settings(self, settings):
        """Check for the existence and configuration of LM and RM and add them to the graph."""
        components = {"lm": settings.lm, "rm": settings.rm}
        for component_name, component in components.items():
            if component:
                details = {
                    attr: getattr(component, attr)
                    for attr in dir(component)
                    if not attr.startswith("_") and not callable(getattr(component, attr))
                }
                component_details = f"{component_name.upper()} Details: " + ", ".join(
                    f"{k}: {v}" for k, v in details.items() if k != "history"
                )
                self.graph.node(component_name, label=component_details, shape="box")
                self.nodes.add(component_name)

    def add_module(self, module_name, module):
        """Add a module to the graph"""

        module_type = type(module)

        if "dspy.predict" in str(module_type):
            module_name = self.generate_module_name(module_name, module_type)
            self.process_submodule(module_name, module)
        else:
            self.process_submodules(module_name, module)

    def generate_module_name(self, base_name, module_type):
        """Generate a module name based on the module type"""
        type_map = {
            "Predict": "__Predict",
            "ReAct": "__ReAct",
            "ChainOfThought": "__ChainOfThought",
            "ProgramOfThought": "__ProgramOfThought",
            "MultiChainComparison": "__MultiChainComparison",
            "majority": "__majority",
        }

        for key, suffix in type_map.items():
            if key in str(module_type):
                return base_name + suffix
        return base_name

    def process_submodules(self, module_name, module):
        """Process submodules of a module and add them to the graph"""

        for sub_module_name, sub_module in module.__dict__.items():
            if isinstance(sub_module, dspy.Predict):
                sub_module_name = self.generate_module_name(sub_module_name, type(sub_module))
                self.process_submodule(f"{module_name}__{sub_module_name}", sub_module)
            elif isinstance(sub_module, (dspy.Module, dspy.Retrieve)):
                self.add_module(f"{module_name}__{sub_module_name}", sub_module)
                if isinstance(sub_module, dspy.Retrieve):
                    self.graph.edge("rm", "lm", label="RM used in Module")

    def process_submodule(self, sub_module_name, sub_module):
        """Process a submodule and add it to the graph"""

        for field_type, fields in [
            ("input", sub_module.signature.input_fields),
            ("output", sub_module.signature.output_fields),
        ]:
            for field_name, field in fields.items():
                node_id = f"{sub_module_name}_{field_type}_{field_name}"
                if node_id not in self.nodes:
                    label = f"{field_name}: ({field.json_schema_extra['desc']})"
                    self.graph.node(node_id, label=label, shape="ellipse")
                    self.nodes.add(node_id)
                edge_direction = (node_id, sub_module_name) if field_type == "input" else (sub_module_name, node_id)
                self.graph.edge(*edge_direction)

        # Add node for the submodule itself
        self.graph.node(sub_module_name, label=sub_module_name, shape="box")
        self.nodes.add(sub_module_name)

        # Connect submodule to LM if configured
        if "lm" in self.nodes:
            self.graph.edge("lm", sub_module_name, label="LM used in Module")

    def render_graph(self, filename=None):
        """Render the graph to a file(png)"""
        if filename is None:
            filename = self.module_name
        self.graph.render(filename)


# Example usage of the ModuleGraph class:
# import dspy
# import os
# from dotenv import load_dotenv
# from dspy.experimental import ModuleGraph

# load_dotenv()

# # Configuration of dspy models
# llm = dspy.OpenAI(
#     model='gpt-3.5-turbo',
#     api_key=os.environ['OPENAI_API_KEY'],
#     max_tokens=100
# )

# colbertv2_wiki = dspy.ColBERTv2(url='http://20.102.90.50:2017/wiki17_abstracts')

# dspy.settings.configure(lm=llm, rm=colbertv2_wiki)

# class GenerateAnswer(dspy.Signature):
#   "Answer with long and detailed answers"
#   context = dspy.InputField(desc="may content relevant facts")
#   question = dspy.InputField()
#   answer = dspy.OutputField(desc="often between 10 and 50 words")

# class RAG(dspy.Module):
#   def __init__(self, num_passages=3):
#     super().__init__()
#     self.retrieve = dspy.Retrieve(k=num_passages)
#     self.generate_answer = dspy.ChainOfThought(GenerateAnswer)

#   def forward(self, question):
#     context = self.retrieve(question).passages
#     prediction = self.generate_answer(context=context, question=question)
#     return dspy.Prediction(context=context, answer=prediction.answer)

# rag_system = RAG()
# graph = ModuleGraph("RAG", rag_system)

# graph.render_graph()
