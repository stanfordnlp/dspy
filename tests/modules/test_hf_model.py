# from pytest_mock.plugin import MockerFixture
# from transformers import AutoModelForSeq2SeqLM

# import dspy


# class MockConfig:
#     def __init__(self, architectures: list[str]):
#         self.architectures = architectures


# # def test_load_gated_model(mocker: MockerFixture):
# #     conf = MockConfig(architectures=["ConditionalGeneration"])
# #     mocker.patch("transformers.AutoModelForSeq2SeqLM.from_pretrained")
# #     mocker.patch("transformers.AutoConfig.from_pretrained", return_value=conf)
# #     mocker.patch("transformers.AutoTokenizer.from_pretrained")

# #     some_token = "asdfasdfasdf"
# #     model = "google/gemma-7b"
# #     _ = dspy.HFModel(model, token=some_token)
# #     AutoModelForSeq2SeqLM.from_pretrained.assert_called_with(model, device_map="auto", token=some_token)


# # def test_load_ungated_model(mocker: MockerFixture):
# #     conf = MockConfig(architectures=["ConditionalGeneration"])
# #     # Mock the environment to ensure no default token is used
# #     mocker.patch.dict('os.environ', {}, clear=True)  # Clear environment variables
# #     mocker.patch("transformers.AutoModelForSeq2SeqLM.from_pretrained")
# #     mocker.patch("transformers.AutoConfig.from_pretrained", return_value=conf)
# #     mocker.patch("transformers.AutoTokenizer.from_pretrained")
# #     _ = dspy.HFModel("openai-community/gpt2")
# #     # no token used in automodel
# #     AutoModelForSeq2SeqLM.from_pretrained.assert_called_with("openai-community/gpt2", device_map="auto", token=None)
