# %% [markdown]
# # YOUR FIRST LAB
# ### Please read this section. This is valuable to get you prepared, even if it's a long read -- it's important stuff.
# 
# ## Your first Frontier LLM Project
# 
# Let's build a useful LLM solution - in a matter of minutes.
# 
# By the end of this course, you will have built an autonomous Agentic AI solution with 7 agents that collaborate to solve a business problem. All in good time! We will start with something smaller...
# 
# Our goal is to code a new kind of Web Browser. Give it a URL, and it will respond with a summary. The Reader's Digest of the internet!!
# 
# Before starting, you should have completed the setup for [PC](../SETUP-PC.md) or [Mac](../SETUP-mac.md) and you hopefully launched this jupyter lab from within the project root directory, with your environment activated.
# 
# ## If you're new to Jupyter Lab
# 
# Welcome to the wonderful world of Data Science experimentation! Once you've used Jupyter Lab, you'll wonder how you ever lived without it. Simply click in each "cell" with code in it, such as the cell immediately below this text, and hit Shift+Return to execute that cell. As you wish, you can add a cell with the + button in the toolbar, and print values of variables, or try out variations.  
# 
# I've written a notebook called [Guide to Jupyter](Guide%20to%20Jupyter.ipynb) to help you get more familiar with Jupyter Labs, including adding Markdown comments, using `!` to run shell commands, and `tqdm` to show progress.
# 
# ## If you're new to the Command Line
# 
# Please see these excellent guides: [Command line on PC](https://chatgpt.com/share/67b0acea-ba38-8012-9c34-7a2541052665) and [Command line on Mac](https://chatgpt.com/canvas/shared/67b0b10c93a081918210723867525d2b).  
# 
# ## If you'd prefer to work in IDEs
# 
# If you're more comfortable in IDEs like VSCode or Pycharm, they both work great with these lab notebooks too.  
# If you'd prefer to work in VSCode, [here](https://chatgpt.com/share/676f2e19-c228-8012-9911-6ca42f8ed766) are instructions from an AI friend on how to configure it for the course.
# 
# ## If you'd like to brush up your Python
# 
# I've added a notebook called [Intermediate Python](Intermediate%20Python.ipynb) to get you up to speed. But you should give it a miss if you already have a good idea what this code does:    
# `yield from {book.get("author") for book in books if book.get("author")}`
# 
# ## I am here to help
# 
# If you have any problems at all, please do reach out.  
# I'm available through the platform, or at ed@edwarddonner.com, or at https://www.linkedin.com/in/eddonner/ if you'd like to connect (and I love connecting!)  
# And this is new to me, but I'm also trying out X/Twitter at [@edwarddonner](https://x.com/edwarddonner) - if you're on X, please show me how it's done 😂  
# 
# ## More troubleshooting
# 
# Please see the [troubleshooting](troubleshooting.ipynb) notebook in this folder to diagnose and fix common problems. At the very end of it is a diagnostics script with some useful debug info.
# 
# ## If this is old hat!
# 
# If you're already comfortable with today's material, please hang in there; you can move swiftly through the first few labs - we will get much more in depth as the weeks progress.
# 
# <table style="margin: 0; text-align: left;">
#     <tr>
#         <td style="width: 150px; height: 150px; vertical-align: middle;">
#             <img src="../important.jpg" width="150" height="150" style="display: block;" />
#         </td>
#         <td>
#             <h2 style="color:#900;">Please read - important note</h2>
#             <span style="color:#900;">The way I collaborate with you may be different to other courses you've taken. I prefer not to type code while you watch. Rather, I execute Jupyter Labs, like this, and give you an intuition for what's going on. My suggestion is that you carefully execute this yourself, <b>after</b> watching the lecture. Add print statements to understand what's going on, and then come up with your own variations. If you have a Github account, use this to showcase your variations. Not only is this essential practice, but it demonstrates your skills to others, including perhaps future clients or employers...</span>
#         </td>
#     </tr>
# </table>
# <table style="margin: 0; text-align: left;">
#     <tr>
#         <td style="width: 150px; height: 150px; vertical-align: middle;">
#             <img src="../resources.jpg" width="150" height="150" style="display: block;" />
#         </td>
#         <td>
#             <h2 style="color:#f71;">Treat these labs as a resource</h2>
#             <span style="color:#f71;">I push updates to the code regularly. When people ask questions or have problems, I incorporate it in the code, adding more examples or improved commentary. As a result, you'll notice that the code below isn't identical to the videos. Everything from the videos is here; but in addition, I've added more steps and better explanations, and occasionally added new models like DeepSeek. Consider this like an interactive book that accompanies the lectures.
#             </span>
#         </td>
#     </tr>
# </table>
# <table style="margin: 0; text-align: left;">
#     <tr>
#         <td style="width: 150px; height: 150px; vertical-align: middle;">
#             <img src="../business.jpg" width="150" height="150" style="display: block;" />
#         </td>
#         <td>
#             <h2 style="color:#181;">Business value of these exercises</h2>
#             <span style="color:#181;">A final thought. While I've designed these notebooks to be educational, I've also tried to make them enjoyable. We'll do fun things like have LLMs tell jokes and argue with each other. But fundamentally, my goal is to teach skills you can apply in business. I'll explain business implications as we go, and it's worth keeping this in mind: as you build experience with models and techniques, think of ways you could put this into action at work today. Please do contact me if you'd like to discuss more or if you have ideas to bounce off me.</span>
#         </td>
#     </tr>
# </table>

# %%
# imports

import os
import requests
from dotenv import load_dotenv
from bs4 import BeautifulSoup
from IPython.display import Markdown, display
from openai import OpenAI

# If you get an error running this cell, then please head over to the troubleshooting notebook!

# %% [markdown]
# # Connecting to OpenAI
# 
# The next cell is where we load in the environment variables in your `.env` file and connect to OpenAI.
# 
# ## Troubleshooting if you have problems:
# 
# Head over to the [troubleshooting](troubleshooting.ipynb) notebook in this folder for step by step code to identify the root cause and fix it!
# 
# If you make a change, try restarting the "Kernel" (the python process sitting behind this notebook) by Kernel menu >> Restart Kernel and Clear Outputs of All Cells. Then try this notebook again, starting at the top.
# 
# Or, contact me! Message me or email ed@edwarddonner.com and we will get this to work.
# 
# Any concerns about API costs? See my notes in the README - costs should be minimal, and you can control it at every point. You can also use Ollama as a free alternative, which we discuss during Day 2.

# %%
# Load environment variables in a file called .env

load_dotenv(override=True)
api_key = os.getenv('OPENAI_API_KEY')

# Check the key

if not api_key:
    print("No API key was found - please head over to the troubleshooting notebook in this folder to identify & fix!")
elif not api_key.startswith("sk-proj-"):
    print("An API key was found, but it doesn't start sk-proj-; please check you're using the right key - see troubleshooting notebook")
elif api_key.strip() != api_key:
    print("An API key was found, but it looks like it might have space or tab characters at the start or end - please remove them - see troubleshooting notebook")
else:
    print("API key found and looks good so far!")


# %%
openai = OpenAI()

# If this doesn't work, try Kernel menu >> Restart Kernel and Clear Outputs Of All Cells, then run the cells from the top of this notebook down.
# If it STILL doesn't work (horrors!) then please see the Troubleshooting notebook in this folder for full instructions

# %% [markdown]
# # Let's make a quick call to a Frontier model to get started, as a preview!

# %%
# To give you a preview -- calling OpenAI with these messages is this easy. Any problems, head over to the Troubleshooting notebook.

message = "Hello, GPT! This is my first ever message to you! Hi!"
response = openai.chat.completions.create(model="gpt-4o-mini", messages=[{"role":"user", "content":message}])
print(response.choices[0].message.content)

# %% [markdown]
# ## OK onwards with our first project

# %%
# A class to represent a Webpage
# If you're not familiar with Classes, check out the "Intermediate Python" notebook

# Some websites need you to use proper headers when fetching them:
headers = {
 "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/117.0.0.0 Safari/537.36"
}

class Website:

    def __init__(self, url):
        """
        Create this Website object from the given url using the BeautifulSoup library
        """
        self.url = url
        response = requests.get(url, headers=headers)
        soup = BeautifulSoup(response.content, 'html.parser')
        self.title = soup.title.string if soup.title else "No title found"
        for irrelevant in soup.body(["script", "style", "img", "input"]):
            irrelevant.decompose()
        self.text = soup.body.get_text(separator="\n", strip=True)

# %%
# Let's try one out. Change the website and add print statements to follow along.

ed = Website("https://edwarddonner.com")
print(ed.title)
print(ed.text)

# %% [markdown]
# ## Types of prompts
# 
# You may know this already - but if not, you will get very familiar with it!
# 
# Models like GPT4o have been trained to receive instructions in a particular way.
# 
# They expect to receive:
# 
# **A system prompt** that tells them what task they are performing and what tone they should use
# 
# **A user prompt** -- the conversation starter that they should reply to

# %%
# Define our system prompt - you can experiment with this later, changing the last sentence to 'Respond in markdown in Spanish."

system_prompt = "You are an assistant that analyzes the contents of a website \
and provides a short summary, ignoring text that might be navigation related. \
Respond in markdown."

# %%
# A function that writes a User Prompt that asks for summaries of websites:

def user_prompt_for(website):
    user_prompt = f"You are looking at a website titled {website.title}"
    user_prompt += "\nThe contents of this website is as follows; \
please provide a short summary of this website in markdown. \
If it includes news or announcements, then summarize these too.\n\n"
    user_prompt += website.text
    return user_prompt

# %%
print(user_prompt_for(ed))

# %% [markdown]
# ## Messages
# 
# The API from OpenAI expects to receive messages in a particular structure.
# Many of the other APIs share this structure:
# 
# ```
# [
#     {"role": "system", "content": "system message goes here"},
#     {"role": "user", "content": "user message goes here"}
# ]
# 
# To give you a preview, the next 2 cells make a rather simple call - we won't stretch the mighty GPT (yet!)

# %%
messages = [
    {"role": "system", "content": "You are a snarky assistant"},
    {"role": "user", "content": "What is 2 + 2?"}
]

# %%
# To give you a preview -- calling OpenAI with system and user messages:

response = openai.chat.completions.create(model="gpt-4o-mini", messages=messages)
print(response.choices[0].message.content)

# %% [markdown]
# ## And now let's build useful messages for GPT-4o-mini, using a function

# %%
# See how this function creates exactly the format above

def messages_for(website):
    return [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt_for(website)}
    ]

# %%
messages_for(ed)

# %%
# Try this out, and then try for a few more websites

messages_for(ed)

# %% [markdown]
# ## Time to bring it together - the API for OpenAI is very simple!

# %%
# And now: call the OpenAI API. You will get very familiar with this!

def summarize(url):
    website = Website(url)
    response = openai.chat.completions.create(
        model = "gpt-4o-mini",
        messages = messages_for(website)
    )
    return response.choices[0].message.content

# %%
summarize("https://www.theregister.com/2025/03/13/microsoft_natural_gas_ai/")

# %%
# A function to display this nicely in the Jupyter output, using markdown

def display_summary(url):
    summary = summarize(url)
    display(Markdown(summary))

# %%
display_summary("https://www.theregister.com/2025/03/13/microsoft_natural_gas_ai/")

# %% [markdown]
# # Let's try more websites
# 
# Note that this will only work on websites that can be scraped using this simplistic approach.
# 
# Websites that are rendered with Javascript, like React apps, won't show up. See the community-contributions folder for a Selenium implementation that gets around this. You'll need to read up on installing Selenium (ask ChatGPT!)
# 
# Also Websites protected with CloudFront (and similar) may give 403 errors - many thanks Andy J for pointing this out.
# 
# But many websites will work just fine!

# %%
display_summary("https://www.theregister.com/2025/02/12/how_to_reduce_ai_hallucinations/")

# %%
display_summary("https://anthropic.com")

# %% [markdown]
# <table style="margin: 0; text-align: left;">
#     <tr>
#         <td style="width: 150px; height: 150px; vertical-align: middle;">
#             <img src="../business.jpg" width="150" height="150" style="display: block;" />
#         </td>
#         <td>
#             <h2 style="color:#181;">Business applications</h2>
#             <span style="color:#181;">In this exercise, you experienced calling the Cloud API of a Frontier Model (a leading model at the frontier of AI) for the first time. We will be using APIs like OpenAI at many stages in the course, in addition to building our own LLMs.
# 
# More specifically, we've applied this to Summarization - a classic Gen AI use case to make a summary. This can be applied to any business vertical - summarizing the news, summarizing financial performance, summarizing a resume in a cover letter - the applications are limitless. Consider how you could apply Summarization in your business, and try prototyping a solution.</span>
#         </td>
#     </tr>
# </table>
# 
# <table style="margin: 0; text-align: left;">
#     <tr>
#         <td style="width: 150px; height: 150px; vertical-align: middle;">
#             <img src="../important.jpg" width="150" height="150" style="display: block;" />
#         </td>
#         <td>
#             <h2 style="color:#900;">Before you continue - now try yourself</h2>
#             <span style="color:#900;">Use the cell below to make your own simple commercial example. Stick with the summarization use case for now. Here's an idea: write something that will take the contents of an email, and will suggest an appropriate short subject line for the email. That's the kind of feature that might be built into a commercial email tool.</span>
#         </td>
#     </tr>
# </table>

# %%
# Step 1: Create your prompts

system_prompt = "you are a cyber security consultant"
user_prompt = """
    explain a windows firewall to a non technical person"""

messages = [
    {"role": "system", "content": system_prompt},
    {"role": "user", "content": user_prompt}
]

# Step 2: Make the messages list

# messages = [] # fill this in


# Step 3: Call OpenAI

response = openai.chat.completions.create(model="gpt-4o-mini", messages=messages)
print(response.choices[0].message.content)

# Step 4: print the result

print(messages)

# %% [markdown]
# ## An extra exercise for those who enjoy web scraping
# 
# You may notice that if you try `display_summary("https://openai.com")` - it doesn't work! That's because OpenAI has a fancy website that uses Javascript. There are many ways around this that some of you might be familiar with. For example, Selenium is a hugely popular framework that runs a browser behind the scenes, renders the page, and allows you to query it. If you have experience with Selenium, Playwright or similar, then feel free to improve the Website class to use them. In the community-contributions folder, you'll find an example Selenium solution from a student (thank you!)

# %% [markdown]
# # Sharing your code
# 
# I'd love it if you share your code afterwards so I can share it with others! You'll notice that some students have already made changes (including a Selenium implementation) which you will find in the community-contributions folder. If you'd like add your changes to that folder, submit a Pull Request with your new versions in that folder and I'll merge your changes.
# 
# If you're not an expert with git (and I am not!) then GPT has given some nice instructions on how to submit a Pull Request. It's a bit of an involved process, but once you've done it once it's pretty clear. As a pro-tip: it's best if you clear the outputs of your Jupyter notebooks (Edit >> Clean outputs of all cells, and then Save) for clean notebooks.
# 
# Here are good instructions courtesy of an AI friend:  
# https://chatgpt.com/share/677a9cb5-c64c-8012-99e0-e06e88afd293

# %%



