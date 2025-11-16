🧠 AI Voice RAG Agent

This project implements a complete Retrieval-Augmented Generation (RAG) agent that operates through a voice interface. It combines Speech-to-Text (STT), a local Large Language Model (LLM), web search for up-to-date information, a knowledge graph for enhanced context, and Text-to-Speech (TTS) for an audible response.

The agent is designed to run locally, leveraging hardware acceleration (MPS/CUDA) where available.

✨ Features

•
Voice Interface: Uses sounddevice and Whisper (tiny.en model) for real-time voice recording and transcription.

•
Local LLM: Powered by TinyLlama-1.1B-Chat-v1.0 for fast, local text generation.

•
Retrieval-Augmented Generation (RAG): Integrates web search capabilities using the Serper API to fetch up-to-date information.

•
Vector Store: Utilizes FAISS and HuggingFace Embeddings (all-MiniLM-L6-v2) to create a vector database from retrieved web content.

•
Contextual Memory: Maintains conversation history using langchain's ConversationBufferMemory.

•
Knowledge Graph: Builds a simple, session-based knowledge graph using spaCy and NetworkX to extract and relate entities from the retrieved context, further enriching the LLM's prompt.

•
Text-to-Speech (TTS): Uses Edge TTS for high-quality, natural-sounding voice output.

•
Hardware Acceleration: Automatically detects and utilizes MPS (Apple Silicon) or falls back to CPU.

⚙️ Prerequisites

Before running the agent, ensure you have the following:

1.
Python 3.x

2.
A working microphone

3.
Serper API Key: A key from Serper is required for the web search functionality.

🚀 Installation

1.
Clone the repository:

2.
Install dependencies: This project relies on several complex libraries, including PyTorch, sounddevice, and whisper. It is highly recommended to use a virtual environment.

🛠️ Configuration

1.
Serper API Key: Open agent_core.py and replace the placeholder API key on line 105 with your actual Serper API key:

▶️ Usage

Run the main script from your terminal:

Bash


python agent_core.py


The agent will first load the models (Whisper, TinyLlama) and then enter the main loop.

1.
The console will prompt you to "Press ENTER and start speaking:"

2.
Press ENTER.

3.
Speak your question into the microphone for the specified duration (default is 7 seconds).

4.
The agent will:

•
Transcribe your speech.

•
Search the web and create a vector store.

•
Generate an answer using the LLM, augmented by the RAG context and Knowledge Graph.

•
Save the spoken response to static/response.mp3.

•
Print the final text answer.



Note on Voice Recording: The voice recording and playback functionality is designed for a local environment with microphone access. If you are running this in a constrained environment (like a remote server or sandbox), you may need to modify the async_main function to use a text-based input loop instead of the record_voice and speech_to_text functions.

📂 Project Structure

Plain Text


.
├── agent_core.py       # The main agent logic
├── requirements.txt    # List of Python dependencies
└── static/             # Directory for generated files
    └── response.mp3    # Output file for the TTS response


📝 Code Overview

The core logic is divided into modular sections within agent_core.py:

Section
Functionality
Key Libraries
Device Setup
Initializes PyTorch for MPS/CPU.
torch
STT Model
Loads the Whisper model for transcription.
whisper
LLM Model
Loads the TinyLlama model and sets up the text generation pipeline.
transformers
Text-to-Speech
async def speak(text): Generates and saves the audio response.
edge_tts, asyncio
Memory/KG
Initializes conversation memory and the Knowledge Graph structure.
langchain, spacy, networkx
Google Search
def google_search_and_embed(query): Fetches web results, scrapes content, chunks it, and creates a FAISS vector store.
requests, BeautifulSoup, langchain-text-splitters, FAISS, HuggingFaceEmbeddings
Core AI Logic
async def run_agent(query): Orchestrates RAG, KG building, LLM prompting, and memory updates.
All core components
STT Functions
def record_voice() and def speech_to_text(): Handles microphone input and transcription.
sounddevice, scipy, whisper
Main Loop
async def async_main(): The continuous loop that drives the agent's operation.
asyncio


🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request or open an Issue.


