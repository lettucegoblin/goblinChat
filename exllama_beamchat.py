import sys, os, glob, gc
#
sys.path.append('exllama')
from exllama.model import ExLlama, ExLlamaCache, ExLlamaConfig
from exllama.tokenizer import ExLlamaTokenizer
from exllama.generator import ExLlamaGenerator
import torch
# Simple interactive chatbot script

torch.set_grad_enabled(False)
torch.cuda._lazy_init()
min_response_tokens = 4
max_response_tokens = 256
extra_prune = 256

class ExLlamaChatbot:
    #init
    def __init__(self, model_name_or_path, model_basename, username, botname):
        self.model_name_or_path = model_name_or_path
        self.model_basename = model_basename
        self.username = username
        self.bot_name = botname
        self.break_on_newline = False
        self.past = ""
    def unload(self):
        
        self.model.unload()
        self.tokenizer = self.model = self.cache = self.generator = None
        gc.collect()
        torch.cuda.empty_cache()
    def load(self):
        tokenizer_path = os.path.join(self.model_name_or_path, "tokenizer.model")
        model_config_path = os.path.join(self.model_name_or_path, "config.json")
        st_pattern = os.path.join(self.model_name_or_path, "*.safetensors")
        model_path = glob.glob(st_pattern)[0]
        self.tokenizer = ExLlamaTokenizer(tokenizer_path)            # create tokenizer from tokenizer model file
        self.config = ExLlamaConfig(model_config_path)               # create config from config.json
        self.config.model_path = model_path                          # supply path to model weights file    
        self.model = ExLlama(self.config)                            # create ExLlama instance and load the weights

        self.cache = ExLlamaCache(self.model)                             # create cache for inference
        self.generator = ExLlamaGenerator(self.model, self.tokenizer, self.cache)   # create generator

        # Configure generator

        self.generator.disallow_tokens([self.tokenizer.eos_token_id])

        self.generator.settings.token_repetition_penalty_max = 1.2
        self.generator.settings.temperature = 0.95
        self.generator.settings.top_p = 0.65
        self.generator.settings.top_k = 100
        self.generator.settings.typical = 0.5

    def generate_simple_exllama(self, context):
        output = self.generator.generate_simple(context, max_new_tokens = 200)
        return output[len(context):]
    def generate_exllama(self, context):
        ids = self.tokenizer.encode(self.past)
        self.generator.gen_begin(ids)
        res_line = self.bot_name + ":"
        res_tokens = self.tokenizer.encode(res_line)
        num_res_tokens = res_tokens.shape[-1]  # Decode from here

        # Read and format input

        in_line = context
        in_line = self.username + ": " + in_line.strip() + "\n"

        next_userprompt = self.username + ": "

        # No need for this, really, unless we were logging the chat. The actual history we work on is kept in the
        # tokenized sequence in the generator and the state in the cache.

        self.past += in_line

        # SentencePiece doesn't tokenize spaces separately so we can't know from individual tokens if they start a new word
        # or not. Instead, repeatedly decode the generated response as it's being built, starting from the last newline,
        # and print out the differences between consecutive decodings to stream out the response.

        in_tokens = self.tokenizer.encode(in_line)
        in_tokens = torch.cat((in_tokens, res_tokens), dim = 1)

        # If we're approaching the context limit, prune some whole lines from the start of the context. Also prune a
        # little extra so we don't end up rebuilding the cache on every line when up against the limit.

        expect_tokens = in_tokens.shape[-1] + max_response_tokens
        max_tokens = self.config.max_seq_len - expect_tokens
        if self.generator.gen_num_tokens() >= max_tokens:
            self.generator.gen_prune_to(self.config.max_seq_len - expect_tokens - extra_prune, self.tokenizer.newline_token_id)

        # Feed in the user input and "{bot_name}:", tokenized

        self.generator.gen_feed_tokens(in_tokens)

        # Generate with streaming

        print(res_line, end = "")
        sys.stdout.flush()

        self.generator.begin_beam_search()

        for i in range(max_response_tokens):

            # Disallowing the end condition tokens seems like a clean way to force longer replies.

            if i < min_response_tokens:
                self.generator.disallow_tokens([self.tokenizer.newline_token_id, self.tokenizer.eos_token_id])
            else:
                self.generator.disallow_tokens(None)

            # Get a token

            gen_token = self.generator.beam_search()

            # If token is EOS, replace it with newline before continuing

            if gen_token.item() == self.tokenizer.eos_token_id:
                self.generator.replace_last_token(self.tokenizer.newline_token_id)

            # Decode the current line and print any characters added

            num_res_tokens += 1
            text = self.tokenizer.decode(self.generator.sequence_actual[:, -num_res_tokens:][0])
            new_text = text[len(res_line):]

            skip_space = res_line.endswith("\n") and new_text.startswith(" ")  # Bit prettier console output
            res_line += new_text
            if skip_space: new_text = new_text[1:]

            print(new_text, end="")  # (character streaming output is here)
            

            # End conditions

            if self.break_on_newline and gen_token.item() == self.tokenizer.newline_token_id: break
            if gen_token.item() == self.tokenizer.eos_token_id: break

            # Some models will not (or will inconsistently) emit EOS tokens but in a chat sequence will often begin
            # generating for the user instead. Try to catch this and roll back a few tokens to begin the user round.

            if res_line.endswith(f"{self.username}:"):
                plen = self.tokenizer.encode(f"{self.username}:").shape[-1]
                self.generator.gen_rewind(plen)
                next_userprompt = " "
                break

        self.generator.end_beam_search()

        self.past += res_line
        first_round = False
        return res_line