from os import getenv
import discord
from discord.ext import commands
from dotenv import load_dotenv
from chain import Second_Bot, First_Bot  # Asumo que esta importación es necesaria para tu funcionalidad específica
import wave



load_dotenv()

DISCORD_TOKEN = getenv("DISCORD_TOKEN")
SERVER_ID = int(getenv("SERVER_ID"))  # Asegúrate de convertir a entero si tu ID del servidor es un número
# SERVER_ID = '1196073902871421114'

key_words = First_Bot()
chain = Second_Bot()
chain.connect_to_cohere()


intents = discord.Intents.default()
intents.message_content = True

bot = commands.Bot(command_prefix="", intents=intents)



@bot.event
async def on_ready():
    print(f"Logged in as {bot.user.name}")

@bot.command()
async def join(ctx):
    # Verificar si el autor del mensaje está en un canal de voz
    if ctx.author.voice:
        channel = ctx.author.voice.channel
        await channel.connect()
    else:
        await ctx.send("¡No estás en un canal de voz!")

@bot.command()
async def leave(ctx):
    # Verificar si el bot está en un canal de voz
    if ctx.voice_client:
        await ctx.voice_client.disconnect()
    else:
        await ctx.send("No estoy conectado a un canal de voz.")

@bot.event
async def on_message(message):
    if message.author == bot.user:
        return
    print(message.content)
    if message.content == "add":
        print("ADDING TO PINECONE")
        training_data= chain.get_text()
        text= [str(chat) for chat in training_data]
        print(text,type(text))
        chain.add_indexes_to_pinecone(text= text)
        print("ADDED TO PINECONE")
    else: 
        get_words = key_words.run_chain(message.content)
        chain.pinecone_from_existing()
        database = chain.docsearch.similarity_search(query= get_words, k= 5)
        database = [answer.page_content for answer in database]
        rerank = chain.cohere_rerank(query= get_words, docs= database)
        # print("RERANK: ", rerank["page_content"], type(rerank))
        beautifier = chain.formulate_answer(rerank)
        memory = chain.memory.load_memory_variables({'input': message.content})["history_messages"]
        response = chain.run_chain(doc_data=beautifier,user_input=message.content, conversation=memory)
        chain.memory.save_context({"input": message.content}, {"output": response})    
        await message.channel.send(response)

bot.run(DISCORD_TOKEN)
# bot.run('MTE5NjA3ODQ4NjczMzk4Nzg0MA.GzTUEn.AQEOq5mn4ybrc6Wg-w_n6jv97DtWCbFltSEPOQ')