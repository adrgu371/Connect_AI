import aiohttp
import asyncio
import warnings
from bs4 import BeautifulSoup
from langchain_ollama import OllamaLLM
from langchain.agents import Tool, AgentType, initialize_agent
import itertools
import ssl
from cachetools import TTLCache
import os
from black import format_str, FileMode
import json
from datetime import datetime
import fitz  # PyMuPDF for reading PDFs

# Suppress specific deprecation warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)

# API Keys
BRAVE_API_KEY = ""

# Cache for internet data (1 hour TTL, 100 entries)
cache = TTLCache(maxsize=100, ttl=3600)

# File to store conversation history
CONVERSATION_FILE = "conversation_history.json"

# Load conversation history from JSON file
def load_conversation():
    if os.path.exists(CONVERSATION_FILE):
        with open(CONVERSATION_FILE, "r") as file:
            return json.load(file)
    return []

# Save conversation history to JSON file
def save_conversation(conversation):
    with open(CONVERSATION_FILE, "w") as file:
        json.dump(conversation, file, indent=4)

# Initialize conversation history
conversation_history = load_conversation()

# 1. Define Tools (async with caching and SSL handling)
async def search_brave_async(query):
    if query in cache:
        return cache[query]
    async with aiohttp.ClientSession() as session:
        try:
            url = "https://api.search.brave.com/res/v1/web/search"
            headers = {"X-Subscription-Token": BRAVE_API_KEY}
            params = {"q": query, "count": 5}
            # Explicitly handle SSL and verify certificate
            ssl_context = ssl.create_default_context(cafile="/etc/ssl/cert.pem" if os.path.exists("/etc/ssl/cert.pem") else None)
            async with session.get(url, headers=headers, params=params, ssl=ssl_context) as response:
                data = await response.json()
                if data.get("web", {}).get("results"):
                    results = data["web"]["results"]
                    cache[query] = results
                    return results
                cache[query] = "No useful information found."
                return "No useful information found."
        except aiohttp.ClientSSLError as e:
            cache[query] = f"SSL Error with Brave Search API: {str(e)}. Check system date/time, SSL configuration, or contact Brave support."
            return f"SSL Error with Brave Search API: Check system date/time, SSL configuration, or contact Brave support."
        except aiohttp.ClientError as e:
            cache[query] = f"Error with Brave Search API: {str(e)}"
            return f"Error with Brave Search API: {str(e)}"

async def fetch_webpage_content_async(url):
    if url in cache:
        return cache[url]
    async with aiohttp.ClientSession() as session:
        try:
            headers = {"User-Agent": "Mozilla/5.0"}
            # Use a custom SSL context to handle certificate issues
            ssl_context = ssl.create_default_context()
            ssl_context.check_hostname = False
            ssl_context.verify_mode = ssl.CERT_NONE
            async with session.get(url, headers=headers, ssl=ssl_context) as response:
                text = await response.text()
                soup = BeautifulSoup(text, "html.parser")
                paragraphs = soup.find_all("p")
                content = " ".join(p.get_text(strip=True) for p in paragraphs)
                result = content[:1000] if content else "No readable content found on the webpage."
                cache[url] = result
                return result
        except aiohttp.ClientSSLError as e:
            cache[url] = f"SSL Error: {str(e)}. Unable to verify the website's certificate."
            return f"SSL Error: Unable to verify the website's certificate."
        except aiohttp.ClientError as e:
            cache[url] = f"Failed to fetch webpage: {str(e)}"
            return f"Failed to fetch webpage: {str(e)}"
        except Exception as e:
            cache[url] = f"Error processing webpage: {str(e)}"
            return f"Error processing webpage: {str(e)}"

# Function to read PDF files
def read_pdf(file_path):
    try:
        # Open the PDF file
        doc = fitz.open(file_path)
        text = ""
        # Extract text from each page
        for page in doc:
            text += page.get_text()
        return text[:5000]  # Limit to 5000 characters for memory efficiency
    except Exception as e:
        return f"Error reading PDF file: {str(e)}"

tools = [
    Tool(
        name="brave_search",
        func=lambda query: asyncio.run(search_brave_async(query)),
        description="Search using Brave Search API for real-time web results."
    ),
    Tool(
        name="fetch_webpage_content",
        func=lambda url: asyncio.run(fetch_webpage_content_async(url)),
        description="Fetch and extract text content from a specific webpage URL."
    ),
    Tool(
        name="read_pdf",
        func=read_pdf,
        description="Read and extract text content from a PDF file."
    ),
]

# 2. Define Structured Response Function
def structured_response(llm, query, internet_data=None, conversation_history=None):
    if conversation_history:
        history_context = "\n".join([f"User: {entry['user']}\nAI: {entry['ai']}" for entry in conversation_history])
        prompt = f"Conversation History:\n{history_context}\n\nContext: {internet_data}\nQuestion: {query}\nAnswer:"
    elif internet_data:
        prompt = f"Context: {internet_data}\nQuestion: {query}\nAnswer:"
    else:
        prompt = f"Question: {query}\nAnswer:"
    response = llm.invoke(prompt)
    return response.replace(prompt, "").strip()

# 3. Decision Function: When to Search
def should_search_internet(query):
    # Keywords indicating time sensitivity or real-time data
    time_sensitive_words = ["today", "now", "current", "latest", "recent", "weather", "news"]
    # Specific entities or post-2023 topics (assuming Qwen2.5:7B cutoff ~2023)
    specific_or_new = any(char.isupper() for char in query) or "202" in query  # e.g., names or years
    
    return any(word in query.lower() for word in time_sensitive_words) or specific_or_new

# 4. Initialize Ollama LLM (quantized for Apple M3)
llm = OllamaLLM(
    model="llama3.2:1B",  # Use quantized version for better performance on M3
    temperature=0.7,
    top_p=0.9,
    max_tokens=500  # Increased for more detailed responses
)

# 5. Async loading animation
async def animate_loading():
    for c in itertools.cycle(['|', '/', '-', '\\']):
        print(f'\rLoading {c}', end='')
        await asyncio.sleep(0.1)
    print('\r', end='')

# 6. Enhanced Invocation Function
async def main_async(query):
    global conversation_history
    
    # Correct typo if present
    if "Bo布tanu" in query or "Boțanu" in query:
        query = query.replace("Bo布tanu", "Bobouțanu").replace("Boțanu", "Bobouțanu")
    
    # Decide whether to search
    loading_task = asyncio.create_task(animate_loading())
    try:
        if should_search_internet(query) or "2024" in query or "2025" in query:
            search_results = await search_brave_async(query)
            if isinstance(search_results, list) and len(search_results) > 0:
                # Try each result until we successfully fetch content
                for result in search_results:
                    url = result.get("url")
                    if url:
                        webpage_content = await fetch_webpage_content_async(url)
                        if "SSL Error" not in webpage_content and "Failed to fetch" not in webpage_content:
                            # Generate detailed commentary on the content
                            commentary_prompt = f"Read the following content and provide a detailed summary or commentary:\n{webpage_content}\nCommentary:"
                            commentary = llm.invoke(commentary_prompt)
                            response = f"**Source:** {url}\n**Content Summary/Commentary:** {commentary}"
                            break
                else:
                    response = "Unable to fetch content from any of the search results due to SSL or connection issues."
            else:
                response = "No useful information found."
        else:
            # Local response if no search needed
            response = structured_response(llm, query, conversation_history=conversation_history)
    finally:
        loading_task.cancel()
    
    # Add the current interaction to the conversation history
    conversation_history.append({
        "user": query,
        "ai": response,
        "timestamp": datetime.now().isoformat()
    })
    save_conversation(conversation_history)
    
    return response

def agent_invoke_async(query):
    return asyncio.run(main_async(query))

# 7. Initialize the LangChain Agent
agent = initialize_agent(
    tools=tools,
    llm=llm,
    agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
    verbose=True,
    handle_parsing_errors=True,
)

# 8. Test the System
print("Ask a question (type 'exit' to quit):")
while True:
    query = input("Your question: ")
    if query.lower() == "exit":
        break
    try:
        # Check if the query is a PDF file path
        if query.endswith(".pdf"):
            pdf_content = read_pdf(query)
            if "Error reading PDF file" not in pdf_content:
                response = structured_response(llm, "Summarize the following PDF content:", internet_data=pdf_content)
            else:
                response = pdf_content
        else:
            response = agent_invoke_async(query)
        # Format the code if it's a code response
        if "```python" in response:
            code = response.split("```python")[1].split("```")[0]
            try:
                formatted_code = format_str(code, mode=FileMode())
                response = response.replace(code, formatted_code)
            except Exception as e:
                print(f"Error formatting code: {str(e)}")
        print(f"Custom Structured Answer: {response}\n")
    except Exception as e:
        print(f"An error occurred: {str(e)}\n")
