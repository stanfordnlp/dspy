import math
import cohere


class Cohere:
    def __init__(self, model='command-xlarge-nightly', api_key=None, stop_sequences=[]):
        '''
        Parameters
        ----------
        model : str
            Which pre-trained model from Cohere to use?
            Choices are ['medium-20221108', 'xlarge-20221108', 'command-medium-nightly', 'command-xlarge-nightly']
        api_key : str
            The API key for Cohere.
            It can be obtained from https://dashboard.cohere.ai/register.
        stop_sequences : list of str
            Additional stop tokens to end generation.
        '''

        self.co = cohere.Client(api_key)

        self.kwargs = {'model': model, 'temperature': 0.0, 'max_tokens': 150, 'p': 1,
                       'frequency_penalty': 0, 'presence_penalty': 0, 'num_generations': 1,
                       'return_likelihoods': 'GENERATION'}
        self.stop_sequences = stop_sequences
        self.max_num_generations = 5

        self.history = []

    def request(self, prompt, **kwargs):
        raw_kwargs = kwargs
        kwargs = {**self.kwargs, 'stop_sequences': self.stop_sequences, 'prompt': prompt, **kwargs}

        response = self.co.generate(**kwargs)

        history = {'prompt': prompt, 'response': response,
                   'kwargs': kwargs, 'raw_kwargs': raw_kwargs}
        self.history.append(history)

        return response

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
                printed.append((prompt, x['response'].generations))
            
            last_prompt = prompt

            if len(printed) >= n:
                break
                
        for prompt, choices in reversed(printed):
            print('\n\n\n')
            print(prompt, end='')
            self.print_green(choices[0].text, end='')

            if len(choices) > 1:
                self.print_red(f" \t (and {len(choices)-1} other completions)", end='')
            print('\n\n\n')

    def __call__(self, prompt, only_completed=True, return_sorted=False, **kwargs):
        assert only_completed, "for now"
        assert return_sorted is False, "for now"

        # Cohere uses 'num_generations' whereas dsp.generate() uses 'n'
        n = kwargs.pop('n', 1)

        # Cohere can generate upto self.max_num_generations completions at a time
        choices = []
        num_iters = math.ceil(n / self.max_num_generations)
        remainder = n % self.max_num_generations
        for i in range(num_iters):
            if i == (num_iters - 1):
                kwargs['num_generations'] = remainder if remainder != 0 else self.max_num_generations
            else:
                kwargs['num_generations'] = self.max_num_generations
            response = self.request(prompt, **kwargs)
            choices.extend(response.generations)
        completions = [c.text for c in choices]

        if return_sorted and kwargs.get('num_generations', 1) > 1:
            scored_completions = []

            for c in choices:
                scored_completions.append((c.likelihood, c.text))

            scored_completions = sorted(scored_completions, reverse=True)
            completions = [c for _, c in scored_completions]

        return completions
