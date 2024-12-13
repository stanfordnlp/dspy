# from dsp.modules.sentence_vectorizer import FastEmbedVectorizer
# import pytest

# from dspy.primitives.example import Example

# # Skip the test if the 'fastembed' package is not installed
# pytest.importorskip("fastembed", reason="'fastembed' is not installed. Use `pip install fastembed` to install it.")


# @pytest.mark.parametrize(
#     "n_dims,model_name", [(384, "BAAI/bge-small-en-v1.5"), (512, "jinaai/jina-embeddings-v2-small-en")]
# )
# def test_fastembed_with_examples(n_dims, model_name):
#     vectorizer = FastEmbedVectorizer(model_name)

#     examples = [
#         Example(query="What's the price today?", response="The price is $10.00").with_inputs("query", "response"),
#         Example(query="What's the weather today?", response="The weather is sunny").with_inputs("query", "response"),
#         Example(query="Who was leading the team?", response="It was Jim. Rather enthusiastic guy.").with_inputs(
#             "query", "response"
#         ),
#     ]

#     embeddings = vectorizer(examples)

#     assert embeddings.shape == (len(examples), n_dims)


# @pytest.mark.parametrize(
#     "n_dims,model_name", [(384, "BAAI/bge-small-en-v1.5"), (512, "jinaai/jina-embeddings-v2-small-en")]
# )
# def test_fastembed_with_strings(n_dims, model_name):
#     vectorizer = FastEmbedVectorizer(model_name)

#     inputs = [
#         "Jonathan Kent is a fictional character appearing in American comic books published by DC Comics.",
#         "Clark Kent is a fictional character appearing in American comic books published by DC Comics.",
#         "Martha Kent is a fictional character appearing in American comic books published by DC Comics.",
#     ]

#     embeddings = vectorizer(inputs)

#     assert embeddings.shape == (len(inputs), n_dims)
