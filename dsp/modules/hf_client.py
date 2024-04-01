import os
import random
import re
import shutil
import subprocess

# from dsp.modules.adapter import TurboAdapter, DavinciAdapter, LlamaAdapter
import backoff
import requests

from dsp.modules.cache_utils import CacheMemory, NotebookCacheMemory
from dsp.modules.hf import HFModel, openai_to_hf

ERRORS = (Exception)

def backoff_hdlr(details):
    """Handler from https://pypi.org/project/backoff/"""
    print(
        "Backing off {wait:0.1f} seconds after {tries} tries "
        "calling function {target} with kwargs "
        "{kwargs}".format(**details),
    )

class HFClientTGI(HFModel):
    def __init__(self, model, port, url="http://future-hgx-1", http_request_kwargs=None, **kwargs):
        super().__init__(model=model, is_client=True)

        self.url = url
        self.ports = port if isinstance(port, list) else [port]
        self.http_request_kwargs = http_request_kwargs or {}

        self.headers = {"Content-Type": "application/json"}

        self.kwargs = {
            "temperature": 0.01,
            "max_tokens": 75,
            "top_p": 0.97,
            "n": 1,
            "stop": ["\n", "\n\n"],
            **kwargs,
        }

        # print(self.kwargs)

    def _generate(self, prompt, **kwargs):
        kwargs = {**self.kwargs, **kwargs}

        payload = {
        "inputs": prompt,
        "parameters": {
            "do_sample": kwargs["n"] > 1,
            "best_of": kwargs["n"],
            "details": kwargs["n"] > 1,
            # "max_new_tokens": kwargs.get('max_tokens', kwargs.get('max_new_tokens', 75)),
            # "stop": ["\n", "\n\n"],
            **kwargs,
            },
        }

        payload["parameters"] = openai_to_hf(**payload["parameters"])

        payload["parameters"]["temperature"] = max(
            0.1, payload["parameters"]["temperature"],
        )

        # print(payload['parameters'])

        # response = requests.post(self.url + "/generate", json=payload, headers=self.headers)

        response = send_hftgi_request_v01_wrapped(
            f"{self.url}:{random.Random().choice(self.ports)}" + "/generate",
            url=self.url,
            ports=tuple(self.ports),
            json=payload,
            headers=self.headers,
            **self.http_request_kwargs,
        )

        try:
            json_response = response.json()
            # completions = json_response["generated_text"]

            completions = [json_response["generated_text"]]

            if (
                "details" in json_response
                and "best_of_sequences" in json_response["details"]
            ):
                completions += [
                    x["generated_text"]
                    for x in json_response["details"]["best_of_sequences"]
                ]

            response = {"prompt": prompt, "choices": [{"text": c} for c in completions]}
            return response
        except Exception:
            print("Failed to parse JSON response:", response.text)
            raise Exception("Received invalid JSON response from server")


@CacheMemory.cache(ignore=['arg'])
def send_hftgi_request_v01(arg, url, ports, **kwargs):
    return requests.post(arg, **kwargs)

# @functools.lru_cache(maxsize=None if cache_turn_on else 0)
@NotebookCacheMemory.cache(ignore=['arg'])
def send_hftgi_request_v01_wrapped(arg, url, ports, **kwargs):
    return send_hftgi_request_v01(arg, url, ports, **kwargs)


@CacheMemory.cache
def send_hftgi_request_v00(arg, **kwargs):
    return requests.post(arg, **kwargs)


class HFClientVLLM(HFModel):
    def __init__(self, model, port, url="http://localhost", **kwargs):
        super().__init__(model=model, is_client=True)

        if isinstance(url, list):
            self.urls = url
        
        elif isinstance(url, str):
            self.urls = [f'{url}:{port}']
        
        else:
            raise ValueError(f"The url provided to `HFClientVLLM` is neither a string nor a list of strings. It is of type {type(url)}.")
        
        self.headers = {"Content-Type": "application/json"}
        self.kwargs = kwargs


    def _generate(self, prompt, **kwargs):
        kwargs = {**self.kwargs, **kwargs}

        payload = {
            "model": self.model,
            "prompt": prompt,
            **kwargs,
        }

        
        # Round robin the urls.
        url = self.urls.pop(0)
        self.urls.append(url)

        response = send_hfvllm_request_v00(
            f"{url}/v1/completions",
            json=payload,
            headers=self.headers,
        )

        try:
            json_response = response.json()
            completions = json_response["choices"]
            response = {
                "prompt": prompt,
                "choices": [{"text": c["text"]} for c in completions],
            }
            return response

        except Exception:
            print("Failed to parse JSON response:", response.text)
            raise Exception("Received invalid JSON response from server")


@CacheMemory.cache
def send_hfvllm_request_v00(arg, **kwargs):
    return requests.post(arg, **kwargs)


class HFServerTGI:
    def __init__(self, user_dir):
        self.model_weights_dir = os.path.abspath(os.path.join(os.getcwd(), "text-generation-inference", user_dir))
        if not os.path.exists(self.model_weights_dir):
            os.makedirs(self.model_weights_dir)

    def close_server(self, port):
        process = subprocess.Popen(['docker', 'ps'], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        stdout, _ = process.communicate()
        print(stdout)
        if stdout:
            container_ids = stdout.decode().strip().split('\n')
            container_ids = container_ids[1:]
            for container_id in container_ids:
                match = re.search(r'^([a-zA-Z0-9]+)', container_id)
                if match:
                    container_id = match.group(1)
                    port_mapping = subprocess.check_output(['docker', 'port', container_id]).decode().strip()
                    if f'0.0.0.0:{port}' in port_mapping:
                        subprocess.run(['docker', 'stop', container_id], check=False)

    def run_server(self, port, model_name=None, model_path=None, env_variable=None, gpus="all", num_shard=1, max_input_length=4000, max_total_tokens=4096, max_best_of=100):        
        self.close_server(port)
        if model_path:
            model_file_name = os.path.basename(model_path)
            link_path = os.path.join(self.model_weights_dir, model_file_name)
            shutil.copytree(model_path, link_path)
            model_name = os.path.sep + os.path.basename(self.model_weights_dir) + os.path.sep + os.path.basename(model_path)
        docker_command = f'docker run --gpus {gpus} --shm-size 1g -p {port}:80 -v {self.model_weights_dir}:{os.path.sep + os.path.basename(self.model_weights_dir)} -e {env_variable} ghcr.io/huggingface/text-generation-inference:1.1.0 --model-id {model_name} --num-shard {num_shard} --max-input-length {max_input_length} --max-total-tokens {max_total_tokens} --max-best-of {max_best_of}'
        print(f"Connect Command: {docker_command}")
        docker_process = subprocess.Popen(docker_command, shell=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
        connected = False
        output = []
        while True:
            line = docker_process.stdout.readline()
            if not line:
                break
            output.append(line.strip())
            if 'Connected' in line:
                connected = True
                break
        if not connected:
            print("Could not connect to server. Error log:")
            for line in output:
                print(line)
            docker_process.terminate()
        docker_process.wait()

class Together(HFModel):
    def __init__(self, model, **kwargs):
        super().__init__(model=model, is_client=True)
        self.session = requests.Session()
        self.api_base = os.getenv("TOGETHER_API_BASE")
        self.token = os.getenv("TOGETHER_API_KEY")
        self.model = model

        self.use_inst_template = False
        if any(keyword in self.model.lower() for keyword in ["inst", "instruct"]):
            self.use_inst_template = True

        stop_default = "\n\n---"

        self.kwargs = {
            "temperature": 0.0,
            "max_tokens": 512,
            "top_p": 1,
            "top_k": 20,
            "repetition_penalty": 1,
            "n": 1,
            "stop": stop_default if "stop" not in kwargs else kwargs["stop"],
            **kwargs,
        }

    @backoff.on_exception(
        backoff.expo,
        ERRORS,
        max_time=1000,
        on_backoff=backoff_hdlr,
    )
    def _generate(self, prompt, use_chat_api=False, **kwargs):
        url = f"{self.api_base}"

        kwargs = {**self.kwargs, **kwargs}

        stop = kwargs.get("stop")
        temperature = kwargs.get("temperature")
        max_tokens = kwargs.get("max_tokens", 150)
        top_p = kwargs.get("top_p", 0.7)
        top_k = kwargs.get("top_k", 50)
        repetition_penalty = kwargs.get("repetition_penalty", 1)
        prompt = f"[INST]{prompt}[/INST]" if self.use_inst_template else prompt

        if use_chat_api:
            url = f"{self.api_base}/chat/completions"
            messages = [
                {"role": "system", "content": "You are a helpful assistant. You must continue the user text directly without *any* additional interjections."},
                {"role": "user", "content": prompt},
            ]
            body = {
                "model": self.model,
                "messages": messages,
                "temperature": temperature,
                "max_tokens": max_tokens,
                "top_p": top_p,
                "top_k": top_k,
                "repetition_penalty": repetition_penalty,
                "stop": stop,
            }
        else:
            body = {
                "model": self.model,
                "prompt": prompt,
                "temperature": temperature,
                "max_tokens": max_tokens,
                "top_p": top_p,
                "top_k": top_k,
                "repetition_penalty": repetition_penalty,
                "stop": stop,
            }

        headers = {"Authorization": f"Bearer {self.token}"}

        try:
            with self.session.post(url, headers=headers, json=body) as resp:
                resp_json = resp.json()
                if use_chat_api:
                    completions = [resp_json['output'].get('choices', [])[0].get('message', {}).get('content', "")]
                else:
                    completions = [resp_json['output'].get('choices', [])[0].get('text', "")]
                response = {"prompt": prompt, "choices": [{"text": c} for c in completions]}
                return response
        except Exception as e:
            if resp_json:
                print(f"resp_json:{resp_json}")
            print(f"Failed to parse JSON response: {e}")
            raise Exception("Received invalid JSON response from server")


class Anyscale(HFModel):
    def __init__(self, model, **kwargs):
        super().__init__(model=model, is_client=True)
        self.session = requests.Session()
        self.api_base = os.getenv("ANYSCALE_API_BASE")
        self.token = os.getenv("ANYSCALE_API_KEY")
        self.model = model
        self.kwargs = {
            "temperature": 0.0,
            "n": 1,
            **kwargs,
        }

    def _generate(self, prompt, use_chat_api=False, **kwargs):
        url = f"{self.api_base}/completions"
        
        kwargs = {**self.kwargs, **kwargs}

        temperature = kwargs.get("temperature")
        max_tokens = kwargs.get("max_tokens", 150) 

        if use_chat_api:
            url = f"{self.api_base}/chat/completions"
            messages = [
                {"role": "system", "content": "You are a helpful assistant. You must continue the user text directly without *any* additional interjections."},
                {"role": "user", "content": prompt},
            ]
            body = {
                "model": self.model,
                "messages": messages,
                "temperature": temperature,
                "max_tokens": max_tokens,
            }
        else:
            body = {
                "model": self.model,
                "prompt": f"[INST]{prompt}[/INST]",
                "temperature": temperature,
                "max_tokens": max_tokens,
            }

        headers = {"Authorization": f"Bearer {self.token}"}

        try:
            completions = []
            for i in range(kwargs.get('n', 1)):
                with self.session.post(url, headers=headers, json=body) as resp:
                    resp_json = resp.json()
                    if use_chat_api:
                        completions.extend([resp_json.get('choices', [])[0].get('message', {}).get('content', "")])
                    else:
                        completions.extend([resp_json.get('choices', [])[0].get('text', "")])
            response = {"prompt": prompt, "choices": [{"text": c} for c in completions]}
            return response
        except Exception as e:
            print(f"Failed to parse JSON response: {e}")
            raise Exception("Received invalid JSON response from server")


class ChatModuleClient(HFModel):
    def __init__(self, model, model_path):
        super().__init__(model=model, is_client=True)

        from mlc_chat import ChatConfig, ChatModule

        self.cm = ChatModule(
            model=model, lib_path=model_path, chat_config=ChatConfig(conv_template="LM"),
        )

    def _generate(self, prompt, **kwargs):
        output = self.cm.generate(
            prompt=prompt,
        )
        try:
            completions = [{"text": output}]
            response = {"prompt": prompt, "choices": completions}
            return response
        except Exception:
            print("Failed to parse output:", response.text)
            raise Exception("Received invalid output")


class HFClientSGLang(HFModel):
    def __init__(self, model, port, url="http://localhost", **kwargs):
        super().__init__(model=model, is_client=True)
        self.url = f"{url}:{port}"
        self.headers = {"Content-Type": "application/json"}

        self.kwargs = {
            "temperature": 0.01,
            "max_tokens": 75,
            "top_p": 0.97,
            "n": 1,
            "stop": ["\n", "\n\n"],
            **kwargs,
        }

    def _generate(self, prompt, **kwargs):
        kwargs = {**self.kwargs, **kwargs}

        payload = {
            "model": kwargs.get("model", "default"),
            "prompt": prompt,
            **kwargs,
        }

        response = send_hfsglang_request_v00(
            f"{self.url}/v1/completions",
            json=payload,
            headers=self.headers,
        )

        try:
            json_response = response.json()
            completions = json_response["choices"]
            response = {
                "prompt": prompt,
                "choices": [{"text": c["text"]} for c in completions],
            }
            return response

        except Exception:
            print("Failed to parse JSON response:", response.text)
            raise Exception("Received invalid JSON response from server")


@CacheMemory.cache
def send_hfsglang_request_v00(arg, **kwargs):
    return requests.post(arg, **kwargs)
