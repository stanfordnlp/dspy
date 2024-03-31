import dspy

def test_multiple_clients():
    local_api_base = "http://localhost/v1/"
    client_api_key = "key2"
    local_api_key  = "key1"

    openai_client = dspy.OpenAI(model="gpt-3.5-turbo",api_key=client_api_key)
    openai_local  = dspy.OpenAI(model="mistral", api_base=local_api_base, api_key=local_api_key)

    assert openai_client.provider == "openai"
    assert openai_local.provider == "openai"

    assert openai_client.client.api_key == client_api_key
    assert openai_local.client.api_key == local_api_key

    assert str(openai_local.client.base_url) == local_api_base
    assert openai_client.client.base_url != local_api_base