# Nindra
a Godot game engine Ai assistance
Nindra – AI Assistant for Godot

Nindra is an editor plugin for Godot 4 that adds a chat dock with a local LLM assistant. It leverages Ollama to run models on your own machine, providing:

  💬 Chat with streaming responses

  🧠 RAG (Retrieval-Augmented Generation) using your project’s code

  🔍 Automatic indexing of scripts and scenes

  ⚡ Code task helpers: “Suggest next” and “Improve” (uses the currently open script)

  🖼️ Avatar pop‑up (click the avatar to enlarge)


✨ Features

  + Local & private – no API keys, no cloud.

  + Context‑aware – chunks your whole project and retrieves relevant snippets on the fly.

  + Streaming – responses appear word by word.

  + Conversation memory – keeps the last few exchanges.

  + Markdown‑to‑BBCode – nicely formatted code blocks, bold, italic.

  + Editor integration – buttons for “Suggest next” and “Improve” act on the current script.


🧩 Requirements

  + Godot 4.2+

  + Ollama installed and running (ollama serve)

  + Two models pulled: 

        phi4-mini:3.8b-q8_0 (chat)

        mxbai-embed-large:335m-v1-fp16 (embeddings)
    
  + (You can choose bigger models if you have the hardware resources)

You can pull them with:
bash

    ollama pull phi4-mini:3.8b-q8_0  
    ollama pull mxbai-embed-large:335m-v1-fp16

📦 **Installation**

  1. Copy the addons/nindra/ folder into your project’s addons/ directory.

  2. Enable the plugin in Project → Project Settings → Plugins.

  3. A new dock “Nindra” will appear on the right side.

  4. The plugin assumes Ollama is running on http://127.0.0.1:11434. If you changed the port, edit OLLAMA_CHAT_URL and EMBED_URL in chat_dock.gd.

🗂️ **Preparing your project context**

Nindra needs a single text file that contains your whole project’s code and scene structure.
By default it expects res://context_export.txt.

You can generate this file with any script that walks your project and outputs:

```
--- AUTOLOADS / GLOBALS ---
... (gdscript + scene trees) ...

--- SCRIPTS ---
--- SCRIPT: path/to/script.gd ---
```gdscript
code here

--- SCENES ---
scene.tscn:
node structure ...
```

A simple exporter is **not** included – you can write a small tool or use the [Github pages](https://github.com/1Stalk/godot-context-exporter addon).  
The built‑in chunker understands the format described above.

If the file is missing, Nindra will warn you but still work without RAG.

---

🚀 **Usage**

1. **Index your project** – click *Index Project*. This reads `context_export.txt`, splits it into chunks, and computes embeddings.  
   Wait for the “Indexing complete!” message.
2. **Ask a question** – type in the chat and press Send. The assistant will retrieve relevant code snippets and answer with context.
3. **Code tasks** – with a script open in the editor, click:
   - *Suggest Next* – continues the script from the cursor position (inference only)
   - *Improve* – reviews the script and suggests improvements
4. **Clear conversation** – resets the chat history.
5. **Enlarge avatar** – click the character portrait.

---

⚙️ **Configuration (optional)**

All settings are at the top of `chat_dock.gd`:

```gdscript
const OLLAMA_CHAT_URL = "http://127.0.0.1:11434/api/chat"
const EMBED_URL = "http://127.0.0.1:11434/api/embed"
const MODEL = "granite-4.0-1b-bf16:latest"
const EMBED_MODEL = "mxbai-embed-large:335m-v1-fp16"
const MAX_HISTORY_PAIRS = 5
const RETRIEVAL_TOP_K = 5
const CHUNK_MAX_CHARS = 1024
const MMR_LAMBDA = 0.7
```

Change these to match your preferred models or tweak the RAG behaviour.
🧠 How it works (briefly)

  + Chunking – the context file is split into logical pieces (scripts, scenes, autoloads). GDScript is split by top‑level declarations; scenes are kept as whole files or fallback‑split.

  + Embedding – every chunk is sent to Ollama’s /api/embed endpoint. The resulting vectors are stored.

  + Retrieval – the user’s question is embedded and compared to all chunk vectors using cosine similarity. The top chunks are then re‑ranked with MMR (maximal marginal relevance) to balance relevance and diversity.

  + Generation – a system prompt containing the retrieved snippets + conversation history is sent to Ollama’s /api/chat endpoint. Streaming responses are parsed and displayed in real time.

  + Code tasks – the current script is injected into a user message; the assistant responds without streaming and without polluting the chat history.

> [!WARNING]
🐛 **Known limitations**

  1. The chunker uses regex – it may fail on very complex or malformed code.

  2. Only context_export.txt is supported; dynamic live indexing is not implemented.

  3. Embedding requests are blocking for the UI while indexing (a send_button.disabled is set, but the editor may feel sluggish for large projects).

  4. Streaming uses HTTPClient.poll() in _process – be careful not to cause performance issues with many open docks.

  5. The plugin assumes Ollama is always reachable; missing models or connection errors will show red messages.
