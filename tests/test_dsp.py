import unittest


class DspModuleImportsTest(unittest.TestCase):
    def test_convenience_import_of_dsp_modules(self):
        from dsp import Cohere
        from dsp import ColBERTv2
        from dsp import GPT3
        from dsp import HFModel
        from dsp import HFClientTGI
        from dsp import HFClientVLLM
        from dsp import OllamaLocal
        from dsp import PyseriniRetriever
        from dsp import SentenceTransformersCrossEncoder
        from dsp import SentenceTransformersVectorizer
        from dsp import NaiveGetFieldVectorizer
        from dsp import OpenAIVectorizer


if __name__ == '__main__':
    unittest.main()
