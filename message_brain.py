import time

class Message_brain:
    def __init__(self):
        self.users = {} # dictionary of user_id: user
        self.idle_since_timestamp = 0 # timestamp of when the bot last messaged any user
    def add_message(self, user_id, message):
        if user_id not in self.users:
            self.users[user_id] = User(user_id)
        self.idle_since_timestamp = time.time()
        self.users[user_id].add_message(message, self.idle_since_timestamp)
        

class User:
    def __init__(self, user_id):
        self.user_id = user_id
        self.last_messages = []
        self.threshold = 1.0
        self.idle_since_timestamp = 0
    def add_message(self, message, time):
        self.last_messages.append(Message(message, time))
    def set_threshold(self, threshold):
        self.threshold = threshold
    def set_idle_since_timestamp(self, timestamp):
        self.idle_since_timestamp = timestamp
    def get_last_messages(self):
        return self.last_messages
    def get_threshold(self):
        return self.threshold
    def get_idle_since_timestamp(self):
        return self.idle_since_timestamp
    
class Message:
    def __init__(self, text, time):
        self.text = text
        self.time = time
    def __str__(self):
        return self.text + " " + str(self.time)