from transformers import AutoTokenizer, TextGenerationPipeline, StoppingCriteriaList, StoppingCriteria
from auto_gptq import AutoGPTQForCausalLM, BaseQuantizeConfig
import gc, torch, sys, os, glob
from tinydb import TinyDB, Query
from datetime import datetime
sys.path.append('exllama')
from exllama.model import ExLlama, ExLlamaCache, ExLlamaConfig
from exllama.tokenizer import ExLlamaTokenizer
from exllama.generator import ExLlamaGenerator

# Locate files we need within that directory



#describe a cute goblin character who is curious about the world and likes to explore
character_context = '''The following is a conversation between User and Goblin. Goblin is curious about the world and loves explaining.\n'''
class Ai_Wrapper:
    def __init__(self, model_name_or_path, model_basename, tokenizerName):
        self.model_name_or_path = model_name_or_path
        self.model_basename = model_basename
        self.memory = TinyDB('memory.json')
        self.tokenizerName = tokenizerName  
              
        pass
    def load_exllama(self):
        tokenizer_path = os.path.join(self.model_name_or_path, "tokenizer.model")
        model_config_path = os.path.join(self.model_name_or_path, "config.json")
        st_pattern = os.path.join(self.model_name_or_path, "*.safetensors")
        model_path = glob.glob(st_pattern)[0]
        self.tokenizer = ExLlamaTokenizer(tokenizer_path)            # create tokenizer from tokenizer model file
        self.config = ExLlamaConfig(model_config_path)               # create config from config.json
        self.config.model_path = model_path                          # supply path to model weights file    
        self.model = ExLlama(self.config)                            # create ExLlama instance and load the weights
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
    def load_autogptq(self):
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name_or_path, use_fast=True)
        self.model = AutoGPTQForCausalLM.from_quantized(self.model_name_or_path,
            model_basename=self.model_basename,
            device="cuda:0", 
            quantize_config=None,
            use_safetensors=True, 
            use_triton=False)
    def load(self, tokenizerName = None):
        if tokenizerName is not None:
            self.tokenizerName = tokenizerName
        if tokenizerName == "exllama":
            self.load_exllama()
        elif tokenizerName == "autogptq":
            self.load_autogptq()
    def generate(self, input_text, chat_id):
        if self.tokenizer is None:
            raise ValueError("Tokenizer not found")
        
        
        prompt_template=f'''User: {input_text}
Goblin:'''
        context = self.create_context(prompt_template, chat_id)
        if self.tokenizerName == "exllama":
            output_str = self.generate_exllama(input_text)
        elif self.tokenizerName == "autogptq":
            output_str = self.generate_autogptq(input_text)
        else:
            raise ValueError("Tokenizer not found")
        
        # remove <s> and </s> from beginning and end if they exist
        if output_str.startswith("<s>"):
            output_str = output_str[3:]
        if output_str.endswith("</s>"):
            output_str = output_str[:-4]

        # count number of "User:" in context
        num_user = context.count("User:")
        num_user_output = output_str.count("User:")
        if num_user_output > num_user:
            # remove everything after last "User:"
            for i in range(num_user_output - num_user):
                output_str = output_str[:output_str.rfind("User:")]
        
        #output_str = output_str[4:-5]
        # get string after prompt template in output_str including prompt template
        toMemory = output_str[output_str.find(prompt_template):]
        # if toMemory has "User:" in it, remove everything after "User:" including "User:"
        
        
        self.add_to_memory(toMemory, chat_id)
        
        response = output_str[output_str.find(prompt_template)+len(prompt_template):]
        # trim spaces around response
        response = response.strip()
        print("output: " + output_str + "\n\n" + "toMemory: " + toMemory + "\n" + "response: " + response + "\n\n\n")
        return response
    def generate_exllama(self, context):
        output = self.generator.generate_simple(context, max_new_tokens = 200)
        return output[len(context):]
    def generate_autogptq(self, context):
        if self.tokenizer is None:
            raise ValueError("Tokenizer not found")
        if self.model is None:
            raise ValueError("Model not found")
        
        input_ids = self.tokenizer(context, return_tensors='pt').input_ids.cuda()
        
        output = self.model.generate(
            max_new_tokens=256,
            inputs=input_ids, 
            temperature=0.7)
        return self.tokenizer.decode(output[0])
    def add_to_memory(self, output_str, chat_id):
        # time stamp, chat id, and output string
        self.memory.insert({'chat_id': chat_id, 'output': output_str, 'timestamp': datetime.now().strftime("%d/%m/%Y %H:%M:%S")})
        #self.memory += "\n" + output_str
        # if memory is greater than 2048 tokens, remove from the beginning until it is less than 2048 tokens and also a new line character
        #while len(self.memory) > 2048 and self.memory.find("\n") != -1:
        #    self.memory = self.memory[self.memory.find("\n")+1:]
    def create_context(self, prompt_template, chat_id):
        tokens = 2048 - len(prompt_template) - len(character_context)
        context = character_context + self.get_memory_trimmed(tokens, chat_id) + "\n" + prompt_template
        return context
    def get_memory_trimmed(self, max_tokens, chat_id):
        # get all memory for this chat id 
        User = Query()
        allChats = self.memory.search(User.chat_id == chat_id)
        # loop through allChats and add to memory trimmed up to max_tokens
        memory_trimmed = ""
        allChats.reverse()
        for chat in allChats:
            if len(chat['output'] + "\n" + memory_trimmed) > max_tokens:
                break
            memory_trimmed = chat['output'] + "\n" + memory_trimmed
        # if memory is greater than max_tokens, remove from the beginning until it is less than max_tokens and also a new line character
        #memory_trimmed = self.memory
        #while len(memory_trimmed) > max_tokens and memory_trimmed.find("\n") != -1:
        #    memory_trimmed = memory_trimmed[memory_trimmed.find("\n")+1:]
        return memory_trimmed   
    def unload(self):
        if self.tokenizerName == "exllama":
            self.model.unload()
        self.tokenizer = self.model = self.cache = self.generator = None
        gc.collect()
        torch.cuda.empty_cache()
