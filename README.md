# Agentic Base React

A full-stack chatbot application with a React frontend and FastAPI backend, supporting multiple LLM providers (Groq, OpenAI, Gemini, and Ollama).

## Features

- 🤖 Multiple LLM provider support (Groq, OpenAI, Gemini, Ollama)
- 💬 Interactive chat interface with markdown rendering
- 🎯 Multiple use cases (Basic Chatbot, Weather Chatbot, etc.)
- 🔄 Session-based conversation history management
- 📱 Responsive UI with collapsible sidebar
- 🎨 Modern, clean design

## Prerequisites

- **Python 3.13+** (for backend)
- **Node.js 18+** and **npm** (for frontend)
- **API Keys** for at least one LLM provider:
  - Groq API key
  - OpenAI API key
  - Google Gemini API key
  - (Optional) Ollama running locally

## Installation

### 1. Clone the repository

```bash
git clone <repository-url>
cd Agentic-Base-React
```

### 2. Backend Setup

Navigate to the backend directory:

```bash
cd backend
```

Install dependencies using `uv` (recommended) or `pip`:

**Using uv (recommended):**

```bash
uv sync
```

**Using pip:**

```bash
pip install -e .
```

Set up environment variables:

```bash
# Copy the example.env file
cp example.env .env

# Edit .env and add your API keys
# OPENAI_API_KEY=sk-your-key-here
# GROQ_API_KEY=gsk_your-key-here
# GEMINI_API_KEY=your-key-here
# OLLAMA_BASE_URL=http://localhost:11434 (optional)
```

### 3. Frontend Setup

Navigate to the frontend directory:

```bash
cd ../react_frontend
```

Install dependencies:

```bash
npm install
```

## Running the Application

### Start the Backend

From the `backend` directory:

```bash
# Using uv
uv run uvicorn main:app --reload --port 8000

# Or using python directly
python -m uvicorn main:app --reload --port 8000
```

The backend will be available at `http://localhost:8000`

### Start the Frontend

From the `react_frontend` directory:

```bash
npm run dev
```

The frontend will be available at `http://localhost:5173`

### Filesystem MCP Server

This project uses a filesystem MCP server (provided by `@modelcontextprotocol/server-filesystem`) to expose a local directory to the MCP client.

- Prerequisites: Node.js 18+ and `npm`/`npx` available.
- Configure the directory to expose in the backend `.env` file (example in `backend/example.env`):

```
MCP_FILESYSTEM_DIR=./workspace
```

- Run on-demand with `npx` (no install required):

```bash
# from the project root or `backend/`
npx -y @modelcontextprotocol/server-filesystem ./backend/workspace
```

- Install globally (optional) if you prefer the binary to be available system-wide:

```bash
npm install --location=global @modelcontextprotocol/server-filesystem
# or
npm install -g @modelcontextprotocol/server-filesystem
```

- The backend's MCP client will also spawn this command automatically using the configuration in `backend/langgraph_agent/mcps/mcp_config.json`. If the filesystem tool fails to load, check:
  - that `MCP_FILESYSTEM_DIR` points to an existing, writable directory (create `backend/workspace` if needed),
  - `node -v` and `npx -v` output, and
  - backend logs when starting the server (`python -m uvicorn main:app --reload` from `backend/`).

Troubleshooting tips:
- If you see `ENOENT` related to the workspace path, create the folder: `mkdir backend\workspace`.
- To run the filesystem server persistently, run the global binary (if installed) or use a process manager (for example, `pm2` or a background PowerShell job).


## Project Structure

```
Agentic-Base-React/
├── backend/
│   ├── main.py                 # FastAPI application
│   ├── configurations.py       # Configuration settings
│   ├── example.env             # Environment variables template
│   ├── pyproject.toml          # Python dependencies
│   └── langgraph_agent/
│       ├── graphs/             # LangGraph graph definitions
│       ├── llms/               # LLM provider implementations
│       ├── nodes/              # Graph nodes
│       ├── states/             # State definitions
│       └── tools/              # Available tools
├── react_frontend/
│   ├── src/
│   │   ├── App.jsx             # Main React component
│   │   ├── App.css             # Styles
│   │   ├── components/         # React components
│   │   ├── constants.js        # Constants and configurations
│   │   └── utils/              # Utility functions
│   ├── package.json            # Node.js dependencies
│   └── vite.config.js          # Vite configuration
└── README.md                   # This file
```

## Usage

1. **Select a Use Case**: Choose from the dropdown (e.g., Basic Chatbot, Weather Chatbot)
2. **Choose a Provider**: Select your preferred LLM provider (Groq, OpenAI, Gemini, or Ollama)
3. **Select a Model**: Pick a specific model from the selected provider
4. **Start Chatting**: Type your message and press Enter or click Send
5. **Clear Conversation**: Use the red "Clear" button to reset the conversation history

## API Endpoints

- `GET /health` - Health check endpoint
- `POST /chat` - Send a chat message
- `POST /chat/reset` - Reset conversation history

## Development

### Backend Development

The backend uses FastAPI with LangGraph for building stateful, multi-actor applications with LLMs.

### Frontend Development

The frontend is built with React and Vite for fast development and hot module replacement.

## License

[Add your license here]

## Contributing

[Add contribution guidelines here]
