'''
A language model is a model which understands language – more precisely how words occur together in natural language.
A language model is used to predict what word comes next.
The building and training of models is both complex and resource intensive. Luckily, there are pre-trained language models.
A couple of factors to consider when choosing a pre-trained language model:
    What task are you using it for?
    What are the technical requirements to use the model?
baseWindow - the main GUI window that contains everything
chatWindow - displays the conversation between a user and the chatbot
userEntryBox - for the user to type their queries for the Chatbot
sendButton - a button that sends the user query to the Chatbot
send() - collects the user input from the userEntryBox, gets the bot_response
create_and_insert_user_frame() - inserts the user_input into the chatWindow
create_and_insert_bot_frame() - inserts the bot response into the chatWindow
'''


import os
import transformers


def initialize_model():

  model = transformers.pipeline("conversational", model="facebook/blenderbot_small-90M")
  os.environ["TOKENIZERS_PARALLELISM"] = "true"

  return model


def get_bot_response(model, user_input):

  chat = model(transformers.Conversation(user_input))
  bot_response = str(chat)
  bot_response = bot_response[bot_response.find("bot >> ")+6:].strip()

  return bot_response
