from exllama_beamchat import ExLlamaChatbot

model_directory = 'Wizard-Vicuna-7B-Uncensored-GPTQ'
mode_name = 'Wizard-Vicuna-7B-Uncensored-GPTQ-4bit-128g.no-act-order'

chatbot = ExLlamaChatbot(model_directory, mode_name, "User", "Goblin")
chatbot.load()

while True:
    text = input("User: ")
    output = chatbot.generate_exllama(text)
    #print("Goblin:", output)

chatbot.unload()