from transformers import AutoTokenizer, TextGenerationPipeline, StoppingCriteriaList, StoppingCriteria
from auto_gptq import AutoGPTQForCausalLM, BaseQuantizeConfig
import gc, torch, sys, os, glob
from tinydb import TinyDB, Query
from datetime import datetime
from pynvml import *
nvmlInit()
#sys.path.append('exllama')
#from exllama.model import ExLlama, ExLlamaCache, ExLlamaConfig
#from exllama.tokenizer import ExLlamaTokenizer
#from exllama.generator import ExLlamaGenerator
from exllama_beamchat import ExLlamaChatbot

# Locate files we need within that directory

botName = "Goblin"

class Ai_Wrapper:
    def __init__(self, model_name_or_path, model_basename, tokenizerName):
        self.model_name_or_path = model_name_or_path
        self.model_basename = model_basename
        self.memory = TinyDB('memory.json')
        self.tokenizerName = tokenizerName  
        self.idle_since_timestamp = 0 # timestamp of when the bot last messaged any user
        self.modelIsLoaded = False
        self.character_context = '''The following is a conversation between User and Goblin. Goblin is curious about the world and loves explaining.\n'''
        
        # setting time constants
        self.unloadAfterSeconds = 120 # 2 minutes
        self.minTimeBetweenMessages = 5 * 60 # 5 minutes
        self.maxTimeBetweenMessages = 10 * 60 #10 * 60 * 60 # 10 hours
        self.earliestHour = 10 # 10 am
        self.latestHour = 20 # 8 pm
        self.timeToWaitAfterImpromptuMessage = 26 * 60 * 60 # 26 hour
        # setting time variables
        self.nextMessageTime = 0 # 0 means no message is scheduled
        pass
    def load_exllama(self, past, _character_context = None):
        if _character_context is not None:
            self.character_context = _character_context
            
        print("Loading exllama", past)
        self.exllama_chatbot = ExLlamaChatbot(self.model_name_or_path, self.model_basename, "User", "Goblin", self.character_context + past)
        self.exllama_chatbot.load()
        self.tokenizer = self.exllama_chatbot.tokenizer
    def set_character_context(self, _character_context):
        self.character_context = _character_context
    def load_autogptq(self):
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name_or_path, use_fast=True)
        self.model = AutoGPTQForCausalLM.from_quantized(self.model_name_or_path,
            model_basename=self.model_basename,
            device="cuda:0", 
            quantize_config=None,
            use_safetensors=True, 
            use_triton=False)
    def output_vram_usage(self):
        # Get memory information in bytes
        t = torch.cuda.get_device_properties(0).total_memory
        r = torch.cuda.memory_reserved(0)
        a = torch.cuda.memory_allocated(0)
        f = r-a  # free inside reserved

        # Convert bytes to GB
        total_gb = t / (1024 ** 3)
        reserved_gb = r / (1024 ** 3)
        allocated_gb = a / (1024 ** 3)
        free_gb = f / (1024 ** 3)

        print(f'Total memory    : {total_gb:.2f} GB')
        print(f'Reserved memory : {reserved_gb:.2f} GB')
        print(f'Allocated memory: {allocated_gb:.2f} GB')
        print(f'Free memory     : {free_gb:.2f} GB')
        h = nvmlDeviceGetHandleByIndex(0)
        info = nvmlDeviceGetMemoryInfo(h)

        # Convert bytes to GB
        total_gb = info.total / (1024 ** 3)
        free_gb = info.free / (1024 ** 3)
        used_gb = info.used / (1024 ** 3)

        print(f'total    : {total_gb:.2f} GB')
        print(f'free     : {free_gb:.2f} GB')
        print(f'used     : {used_gb:.2f} GB')
    def enough_vram(self):
        h = nvmlDeviceGetHandleByIndex(0)
        info = nvmlDeviceGetMemoryInfo(h)
        r = torch.cuda.memory_reserved(0)
        free_gb = info.free / (1024 ** 3)
        total_gb = info.total / (1024 ** 3)
        free_total = info.free + r
        free_total_gb = free_total / (1024 ** 3)
        print("free_total_gb: " + str(free_total_gb))
        return (free_total_gb > 5.5)
    def load(self, tokenizerName = None, chat_id = None):
        if tokenizerName is not None:
            self.tokenizerName = tokenizerName
        if tokenizerName == self.tokenizerName and self.modelIsLoaded:
            return
        if tokenizerName == "exllama":
            self.load_exllama(self.get_memory(chat_id))
        elif tokenizerName == "autogptq":
            self.load_autogptq()
        self.modelIsLoaded = True
    def generate(self, input_text, chat_id):
        if self.tokenizer is None:
            raise ValueError("Tokenizer not found")
        now = int(datetime.now().timestamp())
        self.idle_since_timestamp = now
        prompt_template=f'''User: {input_text}
Goblin:'''
        context = self.create_context(prompt_template, chat_id)
        if self.tokenizerName == "exllama":
            output_str = self.exllama_chatbot.generate_exllama(input_text)
            #remove variable botName from beginning of output_str
            output_str = output_str[len(botName) + 2:] # +2 for the colon and space
            output_str = output_str.strip()
            #toMemory = prompt_template + " " + output_str
            #self.add_to_memory(toMemory, chat_id)
            self.add_to_memory(f"User: {input_text}", chat_id, now, False) 
            self.add_to_memory(f"Goblin: {output_str}", chat_id, now + 1)

            return output_str
            # --- Early Exit ---
        
        if self.tokenizerName == "autogptq":
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
        
        
        self.add_to_memory(toMemory, chat_id, now)
        
        response = output_str[output_str.find(prompt_template)+len(prompt_template):]
        # trim spaces around response
        response = response.strip()
        print("output: " + output_str + "\n\n" + "toMemory: " + toMemory + "\n" + "response: " + response + "\n\n\n")
        return response
    def generate_simple_exllama(self, context):
        return self.exllama_chatbot.generate_simple_exllama(context)
    def generate_exllama(self, context):
        return self.exllama_chatbot.generate_exllama(context)
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
    def add_to_memory(self, output_str, chat_id, timestamp, bot_generated = True, impromptu_message = False):
        # time stamp, chat id, and output string
        self.memory.insert({'chat_id': chat_id, 'output': output_str, 'timestamp': timestamp, 'bot_generated': bot_generated, 'impromptu_message': impromptu_message})
        #self.memory += "\n" + output_str
        # if memory is greater than 2048 tokens, remove from the beginning until it is less than 2048 tokens and also a new line character
        #while len(self.memory) > 2048 and self.memory.find("\n") != -1:
        #    self.memory = self.memory[self.memory.find("\n")+1:]
    def create_context(self, prompt_template, chat_id):
        tokens = 2048 - len(prompt_template) - len(self.character_context)
        context = self.character_context + self.get_memory_trimmed(tokens, chat_id) + "\n" + prompt_template
        return context
    def delete_memory(self, chat_id):
        User = Query()
        self.memory.remove(User.chat_id == chat_id)
    def get_memory(self, chat_id):
        # get all memory for this chat id 
        User = Query()
        allChats = self.memory.search(User.chat_id == chat_id)
        # loop through allChats and add to memory
        memory = ""
        #allChats.reverse()
        for chat in allChats:
            memory += chat['output'] + "\n"
        return memory
    def get_latest_memory(self, chat_id):
        User = Query()
        allChats = self.memory.search(User.chat_id == chat_id)
        if len(allChats) == 0:
            return None
        return allChats[-1]

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
        if self.modelIsLoaded == False:
            return
        if self.tokenizerName == "exllama":
            #self.model.unload()
            self.exllama_chatbot.unload()
        self.tokenizer = self.model = self.cache = self.generator = None
        gc.collect()
        torch.cuda.empty_cache()
        self.modelIsLoaded = False
