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
from datetime import datetime
import random
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
    print("Generating")
    if not AI_Wrapper.enough_vram():
        AI_Wrapper.output_vram_usage()
        print("Not enough vram")
        await context.bot.send_message(chat_id=update.effective_chat.id, text="Not enough vram")
        return
    
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
async def get_memory(update: Update, context: ContextTypes.DEFAULT_TYPE):
    output = AI_Wrapper.get_latest_memory(update.effective_chat.id)['output']
    await context.bot.send_message(chat_id=update.effective_chat.id, text=output)

async def clear_memory(update: Update, context: ContextTypes.DEFAULT_TYPE):
    AI_Wrapper.delete_memory(update.effective_chat.id)
    AI_Wrapper.unload()
    await context.bot.send_message(chat_id=update.effective_chat.id, text="Memory cleared")

generateLock = asyncio.Lock()


async def my_coroutine(chat_id):
    async with generateLock:
        AI_Wrapper.output_vram_usage()
        if not AI_Wrapper.enough_vram():
            print("Not enough vram")
            return
        now = int(datetime.now().timestamp())
        #if it's been more than 5 minutes since the bot last messaged any user, return early
        timeSinceLastMessage = now - AI_Wrapper.idle_since_timestamp
        
        # unload 
        if timeSinceLastMessage > AI_Wrapper.unloadAfterSeconds:
            if AI_Wrapper.modelIsLoaded:
                print("Unloading")
                AI_Wrapper.unload()
                return

        # if last message was an impromptu message, return early
        lastMessage = AI_Wrapper.get_latest_memory(chat_id)
        if lastMessage is not None:
            timeSinceLastMessageHours = timeSinceLastMessage / 3600
            if now - lastMessage['timestamp'] > AI_Wrapper.timeToWaitAfterImpromptuMessage:
                print(f"Last message was impromptu, but it's been more than {AI_Wrapper.timeToWaitAfterImpromptuMessage / 3600} hours since then.")
                
            elif lastMessage['impromptu_message']:
                timeUntilMessage = (AI_Wrapper.timeToWaitAfterImpromptuMessage - (now - lastMessage['timestamp']))
                hoursUntilNextMessage = timeUntilMessage // 3600
                minutesUntilNextMessage = (timeUntilMessage % 3600) // 60
                secondsUntilNextMessage = timeUntilMessage % 60
                print(f"Last message was impromptu time until message: {hoursUntilNextMessage}h {minutesUntilNextMessage}m {secondsUntilNextMessage}s")
                return

        
        if not (AI_Wrapper.earliestHour <= datetime.now().hour < AI_Wrapper.latestHour):
            print("The current time is outside 10 AM and 8 PM.")
            return

        if AI_Wrapper.minTimeBetweenMessages > timeSinceLastMessage:
            print('Not idle enough, Time until message:', AI_Wrapper.minTimeBetweenMessages - timeSinceLastMessage)
            return
        if(AI_Wrapper.nextMessageTime == 0):
            # choose a random amount of time between AI_Wrapper.minTimeBetweenMessages and AI_Wrapper.maxTimeBetweenMessages
            timeUntilMessage = random.randint(AI_Wrapper.minTimeBetweenMessages, AI_Wrapper.maxTimeBetweenMessages)
            AI_Wrapper.nextMessageTime = now + timeUntilMessage
            print("Next message scheduled for", datetime.fromtimestamp(AI_Wrapper.nextMessageTime))
            return
        if now < AI_Wrapper.nextMessageTime:
            # format time left until message
            timeLeft = AI_Wrapper.nextMessageTime - now
            hoursLeft = timeLeft // 3600
            minutesLeft = (timeLeft % 3600) // 60
            secondsLeft = timeLeft % 60
            print("Not time to message yet, Time until message:", hoursLeft, "hours", minutesLeft, "minutes", secondsLeft, "seconds")
            return
        else:
            AI_Wrapper.nextMessageTime = 0
        
        if AI_Wrapper.idle_since_timestamp == 0:
            if lastMessage is not None:
                idle_since = lastMessage['timestamp']
                timeSinceLastMessage = now - idle_since
            else:
                print("No history of messages. Won't send impromptu message.")
                return
        # convert time_since_last_message to days, hours, minutes, seconds
        
        daysSinceLastMessage = timeSinceLastMessage // 86400 # 86400 seconds in a day
        hoursSinceLastMessage = (timeSinceLastMessage % 86400) // 3600 # 3600 seconds in an hour
        minutesSinceLastMessage = (timeSinceLastMessage % 3600) // 60 # 60 seconds in a minute
        secondsSinceLastMessage = timeSinceLastMessage % 60 # 60 seconds in a minute
        user_statement = ""
        if daysSinceLastMessage > 0:
            #a few days is 2-3 days
            if daysSinceLastMessage == 1:
                user_statement = "User hasn't said anything in a day."
            elif 2 <= daysSinceLastMessage <= 3:
                user_statement = "User hasn't said anything in a few days."
            else:
                user_statement = f"User hasn't said anything in {daysSinceLastMessage} days."
        elif hoursSinceLastMessage > 0:
            if hoursSinceLastMessage == 1:
                user_statement = "User hasn't said anything in an hour."
            else:
                user_statement = f"User hasn't said anything in {hoursSinceLastMessage} hours."
        elif minutesSinceLastMessage > 0:
            if minutesSinceLastMessage == 1:
                user_statement = "User hasn't said anything in a minute."
            else:
                user_statement = f"User hasn't said anything in {minutesSinceLastMessage} minutes."
        context = f"You are in a chat with User. {user_statement} You are Goblin. You are curious about the world and love explaining. Maybe write to user to see how they are doing, tell them about your day, or reference any of the topics you have discussed before.\n Goblin:"
        print(context)
        
        AI_Wrapper.idle_since_timestamp = now
        AI_Wrapper.load("exllama", chat_id)
        output = AI_Wrapper.generate_exllama(context)
        print(output)
        output = output[len('Goblin') + 2:] # +2 for the colon and space
        output = output.strip()
        AI_Wrapper.add_to_memory(f"Goblin: {output}", chat_id, now, True, True) 
        await bot.send_message(chat_id=chat_id, text=output)


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
    get_memory_handler = CommandHandler('get_memory', get_memory)
    clear_memory_handler = CommandHandler('clear_memory', clear_memory)

    application.add_handler(clear_memory_handler)
    application.add_handler(start_handler)
    application.add_handler(plain_handler)
    application.add_handler(unload_handler)
    application.add_handler(character_context_handler)
    application.add_handler(echo_handler)
    application.add_handler(get_memory_handler)
    bot = Bot(TOKEN)

    loop = asyncio.new_event_loop()  # Create a new event loop
    asyncio.set_event_loop(loop)  # Set the new event loop as the current event loop
    #loop.create_task(run_periodically(997001530))
    #loop.run_forever()
    asyncio.run_coroutine_threadsafe(run_periodically(997001530), loop)

    application.run_polling(poll_interval=5)
    #asyncio.run(run_periodically(997001530))