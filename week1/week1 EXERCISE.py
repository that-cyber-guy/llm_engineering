#!/usr/bin/env python
# coding: utf-8

# Import necessary libraries
import os
import pandas as pd
from dotenv import load_dotenv
import openai
from IPython.display import Markdown, display, update_display

# Load environment variables
load_dotenv(override=True)
api_key = os.getenv('OPENAI_API_KEY')

if api_key:
    print("API key is set.")
else:
    print("API key is missing or invalid. Please check your environment variables.")

model = 'gpt-4o-mini'
openai.api_key = api_key

# CSV file path
file_path = 'sitemap.csv'  # Replace with your file path

def read_urls_from_csv(file_path, urls_column='URL'):
    """
    Reads URLs from a CSV file and returns them as a list.
    """
    try:
        # Load the CSV file
        df = pd.read_csv(file_path)
        # Check if the specified column exists
        if urls_column in df.columns:
            # Extract URLs
            return df[urls_column].tolist()
        else:
            print(f"Column '{urls_column}' does not exist in the CSV file.")
            return []
    except Exception as e:
        print(f"An error occurred while reading the file '{file_path}' or accessing the column '{urls_column}': {e}")
        return []

# Prompts
system_prompt = """You are a web application developer and security consultant. 
Your role is to analyze URLs and determine which URLs might be similar in function. Respond in markdown. Group similar URLs into groups."""

def get_url_user_prompt(urls_list):
    user_prompt = "Here is the list of URLs to analyze:\n" + "\n".join(urls_list)
    truncated_urls = []
    total_length = 0
    for url in urls_list:
        if total_length + len(url) + 1 > 5000:  # +1 accounts for the newline character
            break
        truncated_urls.append(url)
        total_length += len(url) + 1
    return "Here is the list of URLs to analyze:\n" + "\n".join(truncated_urls)

def analyse_urls(urls_list):
    """
    Analyzes a list of URLs by sending them to the OpenAI API for grouping and analysis.

    Parameters:
    urls_list (list): A list of URLs to be analyzed.

    Returns:
    None: The function displays the analysis result in Markdown format.
    """
    response = openai.ChatCompletion.create(
        model=model,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": get_url_user_prompt(urls_list)}
        ]
    )
    display(Markdown(response.choices[0].message.content))

def stream_url(urls_list):
    stream = openai.ChatCompletion.create(
        model=MODEL,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": get_url_user_prompt(urls_list)}
        ],
        stream=True
    )
    
    response = ""
    display_handle = display(Markdown(""), display_id=True)
    for chunk in stream:
        response += chunk.choices[0].delta.get('content', '')
        response = response.replace("```", "").replace("markdown", "")
        update_display(Markdown(response), display_id=display_handle.display_id)

urls_list = read_urls_from_csv(file_path, 'URL')
analyse_urls(urls_list)
stream_url(urls_list)  # Uncomment to use streaming