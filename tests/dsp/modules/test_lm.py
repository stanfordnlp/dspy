from dsp.modules.lm import LM


class MockLM(LM):
    def basic_request(self, prompt, **kwargs):
        res = {
            'choices': [{
                "index": 0,
                "message": {
                    "role": "assistant",
                    "content": "   ok",
                },
                "finish_reason": "stop"
            }]
        }
        record = {
            'prompt': prompt,
            'choices': [c['message']['content'] for c in res['choices']],
            'kwargs': kwargs
        }
        self.push_record(**record)
        return res

    def __call__(self, prompt, only_completed=True, return_sorted=False, **kwargs):
        return


def test_history_lm():
    llm = MockLM(model='mock')
    for i in range(3):
        llm.basic_request(f'test {i}', myarg=i)

    hist = llm.format_history(n=3)
    assert 'test 0' in hist
    assert 'test 1' in hist
    assert 'test 2' in hist
