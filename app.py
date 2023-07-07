import logging
import telegram
import time
from threading import Timer
from telegram import Update, Bot
from telegram.ext import filters, MessageHandler, ApplicationBuilder, CommandHandler, ContextTypes, Updater,InlineQueryHandler
from ai_wrapper import Ai_Wrapper
from functools import wraps
model_directory =  'Wizard-Vicuna-7B-Uncensored-GPTQ'
mode_name = 'Wizard-Vicuna-7B-Uncensored-GPTQ-4bit-128g.no-act-order'
TOKEN = '6051016268:AAGRaoNyLpveIMors1T9MwXCgahUVFiSVNk'

#autogptq_wrapper = Autogptq_Wrapper(model_directory,mode_name)
#exllama_generator = Exllama_Generator()
AI_Wrapper = Ai_Wrapper(model_directory,mode_name, "autogptq")

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

@restricted
async def generate(update: Update, context: ContextTypes.DEFAULT_TYPE):
    
    await context.bot.send_chat_action(chat_id=update.effective_chat.id, action='typing')
    text = update.message.text
    AI_Wrapper.load("exllama")
    #AI_Wrapper.load("autogptq")
    output = AI_Wrapper.generate(text, update.effective_chat.id)
    AI_Wrapper.unload()
    #output = await generate_autogptq(text, update.effective_chat.id)
    #output = await generate_exllama(text)
    await context.bot.send_message(chat_id=update.effective_chat.id, text=output)

if __name__ == '__main__':
    application = ApplicationBuilder().token(TOKEN).build()
    
    start_handler = CommandHandler('start', start)
    plain_handler = MessageHandler(filters.TEXT & (~filters.COMMAND), generate)
    caps_handler = CommandHandler('caps', caps)
    generate_handler = CommandHandler('gen', generate)

    application.add_handler(start_handler)
    application.add_handler(plain_handler)
    application.add_handler(caps_handler)
    application.add_handler(generate_handler)
    #bot = Bot(TOKEN)
    #bot.send_message(chat_id=997001530, text="I'm back online!")
    
    
    application.run_polling(poll_interval=5)