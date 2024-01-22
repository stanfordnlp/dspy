import unittest


class DspModuleImportsTest(unittest.TestCase):
    def test_convenience_import_of_dsp_ollama_client(self):
        from dsp import OllamaLocal


if __name__ == '__main__':
    unittest.main()
