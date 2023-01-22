import openai
import openai.error
import backoff
import functools

from dsp.modules.cache_utils import CacheMemory, NotebookCacheMemory, cache_turn_on


def backoff_hdlr(details):
    # Handler from https://pypi.org/project/backoff/
    print("Backing off {wait:0.1f} seconds after {tries} tries "
          "calling function {target} with args {args} and kwargs "
          "{kwargs}".format(**details))


class GPT3:
    def __init__(self, model='text-davinci-002', api_key=None):
        if api_key:
            openai.api_key = api_key

        self.kwargs = {'model': model, 'temperature': 0.0, 'max_tokens': 150, 'top_p': 1,
                       'frequency_penalty': 0, 'presence_penalty': 0, 'n': 1}

        self.history = []

    def basic_request(self, prompt, **kwargs):
        raw_kwargs = kwargs
        kwargs = {**self.kwargs, 'prompt': prompt, **kwargs}
        response = cached_gpt3_request(**kwargs)

        history = {'prompt': prompt, 'response': response,
                   'kwargs': kwargs, 'raw_kwargs': raw_kwargs}
        self.history.append(history)

        return response

    @backoff.on_exception(backoff.expo,
                          (openai.error.RateLimitError,
                           openai.error.ServiceUnavailableError),
                          max_time=1000,
                          on_backoff=backoff_hdlr)
    def request(self, prompt, **kwargs):
        return self.basic_request(prompt, **kwargs)

    def print_green(self, text, end='\n'):
        print("\x1b[32m" + str(text) + "\x1b[0m", end=end)
    
    def print_red(self, text, end='\n'):
        print("\x1b[31m" + str(text) + "\x1b[0m", end=end)

    def inspect_history(self, n=1):
        last_prompt = None
        printed = []

        for x in reversed(self.history[-100:]):
            prompt = x['prompt']

            if prompt != last_prompt:
                printed.append((prompt, x['response']['choices']))
            
            last_prompt = prompt

            if len(printed) >= n:
                break
                
        for prompt, choices in reversed(printed):
            print('\n\n\n')
            print(prompt, end='')
            self.print_green(choices[0]['text'], end='')

            if len(choices) > 1:
                self.print_red(f" \t (and {len(choices)-1} other completions)", end='')
            print('\n\n\n')


    def __call__(self, prompt, only_completed=True, return_sorted=False, **kwargs):
        assert only_completed, "for now"
        assert return_sorted is False, "for now"

        if kwargs.get('n', 1) > 1:
            kwargs = {**kwargs, 'logprobs': 5}

        response = self.request(prompt, **kwargs)
        choices = response['choices']

        completed_choices = [c for c in choices if c['finish_reason'] != 'length']

        if only_completed and len(completed_choices):
            choices = completed_choices

        completions = [c['text'] for c in choices]

        if return_sorted and kwargs.get('n', 1) > 1:
            scored_completions = []

            for c in choices:
                tokens, logprobs = c['logprobs']['tokens'], c['logprobs']['token_logprobs']

                if '<|endoftext|>' in tokens:
                    index = tokens.index('<|endoftext|>') + 1
                    tokens, logprobs = tokens[:index], logprobs[:index]

                avglog = sum(logprobs) / len(logprobs)
                scored_completions.append((avglog, c['text']))

            scored_completions = sorted(scored_completions, reverse=True)
            completions = [c for _, c in scored_completions]

        return completions


@CacheMemory.cache
def cached_gpt3_request_v2(**kwargs):
    return openai.Completion.create(**kwargs)


@functools.lru_cache(maxsize=None if cache_turn_on else 0)
@NotebookCacheMemory.cache
def cached_gpt3_request_v2_wrapped(**kwargs):
    return cached_gpt3_request_v2(**kwargs)


cached_gpt3_request = cached_gpt3_request_v2_wrapped
