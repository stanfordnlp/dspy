import subprocess
import time
import regex as re
import os
import shutil

class HFServerTGI:
    def __init__(self, user_dir):
        self.model_weights_dir = os.path.abspath(os.path.join(os.getcwd(), "text-generation-inference", user_dir))
        if not os.path.exists(self.model_weights_dir):
            os.makedirs(self.model_weights_dir)

    def close_server(self, port):
        try:
            process = subprocess.Popen(['docker', 'ps'], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            stdout, stderr = process.communicate()
            if process.returncode != 0:
                print(f"Error executing docker ps: {stderr.decode()}")
                return
            print(stdout)
            if stdout:
                container_ids = stdout.decode().strip().split('\n')
                container_ids = container_ids[1:]
                for container_id in container_ids:
                    match = re.search(r'^([a-zA-Z0-9]+)', container_id)
                    if match:
                        container_id = match.group(1)
                        try:
                            port_mapping = subprocess.check_output(['docker', 'port', container_id]).decode().strip()
                        except subprocess.CalledProcessError as e:
                            print(f"Error fetching port mapping for container {container_id}: {e.output}")
                            continue
                        if f'0.0.0.0:{port}' in port_mapping:
                            try:
                                subprocess.run(['docker', 'stop', container_id], check=True)
                            except subprocess.CalledProcessError as e:
                                print(f"Error stopping container {container_id}: {e.output}")
        except Exception as e:
            print(f"An unexpected error occurred: {e}")

    def run_server(self, port, model_name=None, model_path=None, env_variable=None, gpus="all", num_shard=1, max_input_length=4000, max_total_tokens=4096, max_best_of=100):        
        self.close_server(port)
        if model_path:
            model_file_name = os.path.basename(model_path)
            link_path = os.path.join(self.model_weights_dir, model_file_name)
            shutil.copytree(model_path, link_path)
            model_name = os.path.sep + os.path.basename(self.model_weights_dir) + os.path.sep + os.path.basename(model_path)
        docker_command = f'docker run --gpus {gpus} --shm-size 1g -p {port}:80 -v {self.model_weights_dir}:{os.path.sep + os.path.basename(self.model_weights_dir)} -e {env_variable} ghcr.io/huggingface/text-generation-inference:0.9.3 --model-id {model_name} --num-shard {num_shard} --max-input-length {max_input_length} --max-total-tokens {max_total_tokens} --max-best-of {max_best_of}'
        print(f"Connect Command: {docker_command}")
        docker_process = subprocess.Popen(docker_command, shell=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, universal_newlines=True)

        while True:
            output_line = docker_process.stdout.readline()
            if "Connected" in output_line:
                break
            if docker_process.poll() is not None and not output_line:
                print("Docker process ended without connecting.")
                break


if __name__ == "__main__":
    server = HFServerTGI(user_dir="model_weights")
    
    server.run_server(
        model_name="meta-llama/Llama-2-7b-hf",
        port=9000,
        gpus='"device=2"',
        env_variable="HUGGING_FACE_HUB_TOKEN=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",
        num_shard="1",
        max_input_length="4000",
        max_total_tokens="4096",
        max_best_of="100",
    )
    server.close_server(9000)

    





