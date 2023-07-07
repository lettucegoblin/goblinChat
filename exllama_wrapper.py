import sys, os, glob, gc
import torch
sys.path.append('exllama')
from exllama.model import ExLlama, ExLlamaCache, ExLlamaConfig
from exllama.tokenizer import ExLlamaTokenizer
from exllama.generator import ExLlamaGenerator

# Directory containing model, tokenizer, generator

model_directory =  'D:\\text-generation-webui\\models\\Wizard-Vicuna-7B-Uncensored-GPTQ'

# Locate files we need within that directory

tokenizer_path = os.path.join(model_directory, "tokenizer.model")
model_config_path = os.path.join(model_directory, "config.json")
st_pattern = os.path.join(model_directory, "*.safetensors")
model_path = glob.glob(st_pattern)[0]


class Exllama_Generator:
    def __init__(self) -> None: 
        self.config = ExLlamaConfig(model_config_path)               # create config from config.json
        self.config.model_path = model_path                          # supply path to model weights file
        pass
    def load_model(self):
        self.model = ExLlama(self.config)                                 # create ExLlama instance and load the weights
        self.tokenizer = ExLlamaTokenizer(tokenizer_path)            # create tokenizer from tokenizer model file

        self.cache = ExLlamaCache(self.model)                             # create cache for inference
        self.generator = ExLlamaGenerator(self.model, self.tokenizer, self.cache)   # create generator

        # Configure generator

        self.generator.disallow_tokens([self.tokenizer.eos_token_id])

        self.generator.settings.token_repetition_penalty_max = 1.2
        self.generator.settings.temperature = 0.95
        self.generator.settings.top_p = 0.65
        self.generator.settings.top_k = 100
        self.generator.settings.typical = 0.5
    def unload_model(self):
        self.model.unload()
        self.tokenizer = self.model = self.cache = self.generator = None
        gc.collect()
        torch.cuda.empty_cache()
        #del self.model
        #del self.generator
        #del self.cache
    def generate(self, prompt): 
        output = self.generator.generate_simple(prompt, max_new_tokens = 200)
        return output[len(prompt):]