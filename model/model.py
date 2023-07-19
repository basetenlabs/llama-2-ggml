from typing import Any
from ctransformers import AutoModelForCausalLM
import os


class Model:
    def __init__(self, **kwargs) -> None:
        self._llm = None
        self._use_gpu = True

        if self._use_gpu:
            os.system("pip uninstall ctransformers --yes")
            os.system("CT_CUBLAS=1 pip install ctransformers --no-binary ctransformers")

    def load(self):
        # TODO(varun): update hf_cache to handle filenames
        if self._use_gpu:
            self._llm = AutoModelForCausalLM.from_pretrained("TheBloke/Llama-2-13B-Chat-GGML", model_file='llama-2-13b-chat.ggmlv3.q4_0.bin', model_type="llama", gpu_layers=100)
        else:
            self._llm = AutoModelForCausalLM.from_pretrained("TheBloke/Llama-2-7B-Chat-GGML", model_file='llama-2-7b-chat.ggmlv3.q4_0.bin', model_type="llama")
        
    def predict(self, model_input: Any) -> Any:
        prompt = model_input.get("prompt")
        template = f"System: You are a helpful, respectful and honest assistant. Always answer as helpfully as possible, while being safe.\nUser: {prompt}\nAssistant: "

        response = {}
        response["response"] = self._llm(template, stop=["\nUser:"])
        return response
