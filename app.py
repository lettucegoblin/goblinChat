import logging
import telegram
import time
from threading import Timer
from telegram import Update, Bot
from telegram.ext import filters, MessageHandler, ApplicationBuilder, CommandHandler, ContextTypes, Updater,InlineQueryHandler
from ai_wrapper import Ai_Wrapper
from functools import wraps
import asyncio
from typing import Callable
model_directory =  'Wizard-Vicuna-7B-Uncensored-GPTQ'
mode_name = 'Wizard-Vicuna-7B-Uncensored-GPTQ-4bit-128g.no-act-order'
TOKEN = '6051016268:AAGRaoNyLpveIMors1T9MwXCgahUVFiSVNk'

#autogptq_wrapper = Autogptq_Wrapper(model_directory,mode_name)
#exllama_generator = Exllama_Generator()
AI_Wrapper = Ai_Wrapper(model_directory,mode_name, "exllama")

LIST_OF_ADMINS = [997001530, 5640466421, 190121446] # List of user_id of authorized users

bot = Bot(TOKEN)
logging.basicConfig(
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    level=logging.ERROR
)
def restricted(func):
    @wraps(func)
    def wrapped(update, context, *args, **kwargs):
        user_id = update.effective_user.id
        if user_id not in LIST_OF_ADMINS:
            print("Unauthorized access denied for {}.".format(user_id))
            return
        return func(update, context, *args, **kwargs)
    return wrapped
async def start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await context.bot.send_message(chat_id=update.effective_chat.id, text="I'm a bot, please talk to me!")

async def echo(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await context.bot.send_message(chat_id=update.effective_chat.id, text=update.message.text)
async def caps(update: Update, context: ContextTypes.DEFAULT_TYPE):
    text_caps = ' '.join(context.args).upper()
    await context.bot.send_message(chat_id=update.effective_chat.id, text=text_caps)
async def unload(update: Update, context: ContextTypes.DEFAULT_TYPE):
    AI_Wrapper.unload()
    await context.bot.send_message(chat_id=update.effective_chat.id, text="Unloaded")
async def character_context(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await context.bot.send_chat_action(chat_id=update.effective_chat.id, action='typing')
    AI_Wrapper.delete_memory(update.effective_chat.id)
    character_context = update.message.text
    

    if character_context is None:
        await context.bot.send_message(chat_id=update.effective_chat.id, text="No character context found")
        return
    # remove text "/character_context " from character_context
    character_context = character_context[18:].strip()
    # if character context doesnt have \n at the end, add it
    if character_context[-1] != "\n":
        character_context += "\n"
    AI_Wrapper.set_character_context(character_context)
    AI_Wrapper.unload()
    await context.bot.send_message(chat_id=update.effective_chat.id, text="Character context loaded")
@restricted
async def generate(update: Update, context: ContextTypes.DEFAULT_TYPE):
    #print("Generating")
    
    await context.bot.send_chat_action(chat_id=update.effective_chat.id, action='typing')
    text = update.message.text
    AI_Wrapper.load("exllama", update.effective_chat.id)
    #print("Loaded")
    #AI_Wrapper.load("autogptq")
    output = AI_Wrapper.generate(text, update.effective_chat.id)
    #AI_Wrapper.unload()
    #output = await generate_autogptq(text, update.effective_chat.id)
    #output = await generate_exllama(text)
    await context.bot.send_message(chat_id=update.effective_chat.id, text=output)

async def my_coroutine(chat_id):
    print("Coroutine executed")
    await asyncio.sleep(1)
    await bot.send_message(chat_id=chat_id, text="I'm back online!")

async def run_periodically(chat_id):
    while True:
        await asyncio.create_task(my_coroutine(chat_id))
        await asyncio.sleep(5)



if __name__ == '__main__':
    application = ApplicationBuilder().token(TOKEN).build()
    
    start_handler = CommandHandler('start', start)
    plain_handler = MessageHandler(filters.TEXT & (~filters.COMMAND), generate)
    unload_handler = CommandHandler('unload', unload)
    character_context_handler = CommandHandler('character_context', character_context)
    echo_handler = CommandHandler('echo', echo)

    application.add_handler(start_handler)
    application.add_handler(plain_handler)
    application.add_handler(unload_handler)
    application.add_handler(character_context_handler)
    application.add_handler(echo_handler)
    bot = Bot(TOKEN)
    #asyncio.run(run_periodically(997001530))
    
    
    
    
    
    application.run_polling(poll_interval=5)