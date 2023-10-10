import random
import requests
from dsp.modules.hf import HFModel, openai_to_hf
from dsp.modules.cache_utils import CacheMemory, NotebookCacheMemory, cache_turn_on

# from dsp.modules.adapter import TurboAdapter, DavinciAdapter, LlamaAdapter


class HFClientTGI(HFModel):
    def __init__(self, model, port, url="http://future-hgx-1", **kwargs):
        super().__init__(model=model, is_client=True)

        self.url = url
        self.ports = port if isinstance(port, list) else [port]

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
            }
        }

        payload["parameters"] = openai_to_hf(**payload["parameters"])

        payload["parameters"]["temperature"] = max(
            0.1, payload["parameters"]["temperature"]
        )

        # print(payload['parameters'])

        # response = requests.post(self.url + "/generate", json=payload, headers=self.headers)

        response = send_hftgi_request_v01(f"{self.url}:{random.Random().choice(self.ports)}" + "/generate", url=self.url, ports=tuple(self.ports), json=payload, headers=self.headers)

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
        except Exception as e:
            print("Failed to parse JSON response:", response.text)
            raise Exception("Received invalid JSON response from server")


@CacheMemory.cache(ignore=['arg'])
def send_hftgi_request_v01(arg, url, ports, **kwargs):
    return requests.post(arg, **kwargs)

@CacheMemory.cache
def send_hftgi_request_v00(arg, **kwargs):
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
                        subprocess.run(['docker', 'stop', container_id])

    def run_server(self, port, model_name=None, model_path=None, env_variable=None, gpus="all", num_shard=1, max_input_length=4000, max_total_tokens=4096, max_best_of=100):        
        self.close_server(port)
        if model_path:
            model_file_name = os.path.basename(model_path)
            link_path = os.path.join(self.model_weights_dir, model_file_name)
            shutil.copytree(model_path, link_path)
            model_name = os.path.sep + os.path.basename(self.model_weights_dir) + os.path.sep + os.path.basename(model_path)
        docker_command = f'docker run --gpus {gpus} --shm-size 1g -p {port}:80 -v {self.model_weights_dir}:{os.path.sep + os.path.basename(self.model_weights_dir)} -e {env_variable} ghcr.io/huggingface/text-generation-inference:0.9.3 --model-id {model_name} --num-shard {num_shard} --max-input-length {max_input_length} --max-total-tokens {max_total_tokens} --max-best-of {max_best_of}'
        print(f"Connect Command: {docker_command}")
        docker_process = subprocess.Popen(docker_command, shell=True)
        time.sleep(10)
        docker_process.terminate()
        docker_process.wait()

class ChatModuleClient(HFModel):
    def __init__(self, model, model_path):
        super().__init__(model=model, is_client=True)

        from mlc_chat import ChatModule
        from mlc_chat import ChatConfig

        self.cm = ChatModule(
            model=model, lib_path=model_path, chat_config=ChatConfig(conv_template="LM")
        )

    def _generate(self, prompt, **kwargs):
        output = self.cm.generate(
            prompt=prompt,
        )
        try:
            completions = [{"text": output}]
            response = {"prompt": prompt, "choices": completions}
            return response
        except Exception as e:
            print("Failed to parse output:", response.text)
            raise Exception("Received invalid output")
