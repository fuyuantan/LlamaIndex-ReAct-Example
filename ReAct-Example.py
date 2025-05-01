import os
import re
from dotenv import load_dotenv

# --- LlamaIndex Core Imports ---
from llama_index.core.tools import FunctionTool
from llama_index.core.agent import ReActAgent

# --- LlamaIndex LLM Import for Gemini ---
from llama_index.llms.gemini import Gemini # Correct import path
from llama_index.llms.google_genai import GoogleGenAI

# --- Google Generative AI (for potential direct use or configuration) ---
# import google.generativeai as genai

# --- Load Environment Variables (for API Key) ---
load_dotenv()
google_api_key = os.getenv("GOOGLE_API_KEY")

if not google_api_key:
    raise ValueError("GOOGLE_API_KEY not found in environment variables. Please set it in your .env file.")

# # --- Configure Google Generative AI (Optional but good practice) ---
# genai.configure(api_key=google_api_key)

# --- 1. Define Tools ---
# Simple tools for demonstration purposes

def multiply(a: float, b: float) -> float:
    """
    Multiplies two numbers, a and b. Use this for multiplication tasks.
    Args:
        a (float): The first number.
        b (float): The second number.
    """
    print(f"--- Calling Multiply Tool with: a={a}, b={b} ---")
    return a * b

def add(a: float, b: float) -> float:
    """
    Adds two numbers, a and b. Use this for addition tasks.
     Args:
        a (float): The first number.
        b (float): The second number.
    """
    print(f"--- Calling Add Tool with: a={a}, b={b} ---")
    return a + b

def search_wikipedia(query: str) -> str:
    """
    Looks up a query on a FAKE Wikipedia. Use this to find information about people, places, or concepts.
    Args:
        query (str): The search term to look up.
    """
    print(f"--- Calling Wikipedia Tool with query: {query} ---")
    # --- FAKE WIKIPEDIA ---
    # In a real scenario, you'd use the wikipedia library or API
    query = query.lower()
    if "alan turing" in query:
        return "Alan Turing was a British mathematician, computer scientist, logician, cryptanalyst, philosopher, and theoretical biologist. He was highly influential in the development of theoretical computer science."
    elif "llama" in query:
         return "A llama is a domesticated South American camelid, widely used as a meat and pack animal by Andean cultures since the Pre-Columbian era."
    elif "react agent" in query:
        return "A ReAct Agent combines Reasoning and Acting within large language models. It generates verbal reasoning traces and actions pertaining to a task, allowing for dynamic reasoning, tool use, and information gathering."
    else:
        return f"Couldn't find information about '{query}' on Fake Wikipedia."

# --- Convert Python functions to LlamaIndex Tools ---
# The docstrings are crucial as they tell the LLM what the tool does!
multiply_tool = FunctionTool.from_defaults(fn=multiply, name="multiply_numbers")
add_tool = FunctionTool.from_defaults(fn=add, name="add_numbers")
wikipedia_tool = FunctionTool.from_defaults(fn=search_wikipedia, name="wikipedia_search")

# List of tools the agent can use
tools = [multiply_tool, add_tool, wikipedia_tool]

# --- 2. Configure the LLM (Gemini) ---
# Ensure you are using the correct model name that supports function calling if needed,
# or a strong instruction-following model for ReAct. "models/gemini-pro" is a good default.
llm = GoogleGenAI(model="gemini-2.0-flash-lite", api_key=google_api_key)

# --- 3. Create the ReAct Agent ---
agent = ReActAgent.from_tools(
    tools=tools,
    llm=llm,
    verbose=True # Set to True to see the Thought/Action/Observation steps
)

# --- 4. Run the Agent ---
print("--- Starting Agent ---")
question = "Who was Alan Turing and what is 5 added to 12.5?"
# question = "What is a Llama and what is 3 multiplied by 7?" # Try another question

response = agent.chat(question)

print("\n--- Final Answer ---")
print(response)

print("\n--- Agent Finished ---")


# print system prompt
try:
    # 1. Access the Agent Worker (internal component managing execution)
    agent_worker = agent.agent_worker
    # print(f"Agent Worker type: {type(agent_worker)}") # For debugging

    # 2. Access the Chat Formatter used by the worker (often specific to ReAct)
    #    The attribute name might be internal (`_react_chat_formatter`) or public.
    #    We'll try common names. Let's assume it's `_react_chat_formatter`.
    if hasattr(agent_worker, '_react_chat_formatter'):
        formatter = agent_worker._react_chat_formatter
        # print(f"Formatter type: {type(formatter)}") # For debugging

        # 3. The formatter usually holds the system prompt template/header
        #    This might be a PromptTemplate object or a simple string.
        if hasattr(formatter, 'system_header'):
            system_prompt = formatter.system_header
            print("\n--- System Prompt ---")
            print(system_prompt)
            print("---------------------\n")

            # You might also want to see the tool description part generated,
            # as it's dynamically added *into* the system prompt structure.
            if hasattr(formatter, '_get_tool_description'):
                 tool_desc = formatter._get_tool_description(tools)
                 print("\n--- Generated Tool Description (Part of Prompt) ---")
                 print(tool_desc)
                 print("----------------------------------------------------\n")
            else:
                 print("(Could not find method to get formatted tool description)")

        else:
            print("Could not find 'system_header' attribute on the formatter.")
            print("Agent's internal structure might have changed.")
            print("Try inspecting formatter attributes:", dir(formatter))

    elif hasattr(agent_worker, 'get_system_prompt'): # Check for a direct method
         # Less common, but possible
         system_prompt = agent_worker.get_system_prompt(tools=tools) # Might need tools
         print("\n--- System Prompt (via get_system_prompt) ---")
         print(system_prompt)
         print("---------------------------------------------\n")

    else:
        print("Could not find '_react_chat_formatter' or 'get_system_prompt' on the agent worker.")
        print("Agent's internal structure might differ based on LlamaIndex version.")
        print("Try inspecting agent_worker attributes:", dir(agent_worker))

except AttributeError as e:
    print(f"\nError accessing internal agent attributes: {e}")
    print("This often happens if LlamaIndex internal structure has changed between versions.")
    print("Try inspecting the agent object directly:")
    # print(dir(agent))
    # if hasattr(agent, 'agent_worker'):
    #    print(dir(agent.agent_worker))
except Exception as e:
    print(f"\nAn unexpected error occurred: {e}")