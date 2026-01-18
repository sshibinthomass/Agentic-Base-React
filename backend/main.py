import os
import sys
from pathlib import Path
from contextlib import asynccontextmanager
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Dict, List, Optional
from dotenv import load_dotenv

# Add project root to path
current_file = Path(__file__).resolve()
project_root = current_file.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from langgraph_agent.graphs.graph_builder import GraphBuilder
from langgraph_agent.llms.groq_llm import GroqLLM
from langgraph_agent.llms.openai_llm import OpenAiLLM
from langgraph_agent.llms.gemini_llm import GeminiLLM
from langgraph_agent.llms.ollama_llm import OllamaLLM
from langgraph_agent.llms.anthropic_llm import AnthropicLLM
from langchain_core.messages import HumanMessage, SystemMessage, AIMessage
from langgraph.store.memory import InMemoryStore
from langgraph_agent.stores.persistent_store import create_persistent_store

# Load environment variables
load_dotenv()

# Global chatbot graph instance
chatbot_graph = None
# Global MCP tools (loaded once at startup)
mcp_tools = None
# In-memory session store: (session_id, use_case) -> list of LangChain messages
session_store: Dict[str, List] = {}
# Global store for long memory (can be per-session or shared)
# Using a shared persistent store with namespacing per session
long_memory_store = None


async def load_mcp_tools():
    """
    Load MCP tools once at startup. This function caches the tools
    so they're only loaded once and reused for all requests.
    Returns the list of tools that can be reused.
    """
    global mcp_tools
    if mcp_tools is not None:
        return mcp_tools  # Return cached tools if already loaded

    try:
        from langgraph_agent.nodes.mcp_chatbot_node import load_mcp_tools as load_tools

        # Load tools using the function from mcp_chatbot_node
        tools = await load_tools()

        mcp_tools = tools
        print(f"MCP tools loaded: {len(mcp_tools)} tools")
        return mcp_tools
    except Exception as e:
        print(f"Error loading MCP tools: {e}")
        return []


async def initialize_chatbot():
    """Initialize the chatbot graph with Groq LLM"""
    global chatbot_graph
    try:
        user_controls_input = {
            "GROQ_API_KEY": os.getenv("GROQ_API_KEY"),
            "selected_llm": "openai/gpt-oss-20b",
        }
        llm = GroqLLM(user_controls_input)
        llm = llm.get_base_llm()
        graph_builder = GraphBuilder(llm, user_controls_input)
        chatbot_graph = await graph_builder.setup_graph("basic_chatbot")
        return True
    except Exception as e:
        print(f"Error initializing chatbot: {e}")
        return False


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Lifespan context manager for startup and shutdown events"""
    # Startup
    # Load MCP tools once at startup
    await load_mcp_tools()

    if not await initialize_chatbot():
        print(
            "Warning: Failed to initialize chatbot. API will still work but chatbot endpoints may fail."
        )
    yield
    # Shutdown (if needed, add cleanup code here)
    # For example: cleanup resources, close connections, etc.


# Initialize FastAPI app
app = FastAPI(
    title="Agentic Base React Backend",
    description="FastAPI backend for the Agentic Base React application",
    version="0.1.0",
    lifespan=lifespan,
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:3000",
        "http://127.0.0.1:3000",
        "http://localhost:5173",
        "http://127.0.0.1:5173",
        "http://localhost:5174",
        "http://127.0.0.1:5174",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class ChatResponse(BaseModel):
    response: str
    status: str = "success"


class SimpleChatRequest(BaseModel):
    message: str
    session_id: Optional[str] = "default"
    provider: Optional[str] = "groq"  # groq | openai | gemini | ollama
    selected_llm: Optional[str] = None
    use_case: Optional[str] = "basic_chatbot"


class ResetChatRequest(BaseModel):
    session_id: Optional[str] = "default"
    use_case: Optional[str] = "basic_chatbot"


@app.get("/")
async def root():
    """Root endpoint"""
    return {"message": "Agentic Base React Backend API", "status": "running"}


@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {"status": "healthy", "chatbot_initialized": chatbot_graph is not None}


@app.post("/chat", response_model=ChatResponse)
async def chat_simple(request: SimpleChatRequest):
    """
    Simple chat endpoint that takes a message.
    Conversation history is maintained on the backend per session_id.
    """
    if chatbot_graph is None:
        # Even if global init failed, we can still serve requests if provider creds are valid
        # so don't hard error here.
        pass

    try:
        # Choose LLM based on provider/model from request
        print("request-----", request)
        provider = (request.provider or "groq").lower()
        selected_llm = request.selected_llm
        if provider == "groq":
            user_controls_input = {
                "GROQ_API_KEY": os.getenv("GROQ_API_KEY", ""),
                "selected_llm": selected_llm or "openai/gpt-oss-20b",
            }
            llm = GroqLLM(user_controls_input).get_base_llm()
        elif provider == "openai":
            user_controls_input = {
                "OPENAI_API_KEY": os.getenv("OPENAI_API_KEY", ""),
                "selected_llm": selected_llm or "gpt-4o-mini",
            }
            llm = OpenAiLLM(user_controls_input).get_base_llm()
        elif provider == "gemini":
            user_controls_input = {
                "GEMINI_API_KEY": os.getenv("GEMINI_API_KEY", ""),
                "selected_llm": selected_llm or "gemini-2.5-flash",
            }
            llm = GeminiLLM(user_controls_input).get_base_llm()
        elif provider == "ollama":
            user_controls_input = {
                "selected_llm": selected_llm or "gemma3:1b",
                "OLLAMA_BASE_URL": os.getenv(
                    "OLLAMA_BASE_URL", "http://localhost:11434"
                ),
            }
            llm = OllamaLLM(user_controls_input).get_base_llm()
        elif provider == "anthropic":
            user_controls_input = {
                "ANTHROPIC_API_KEY": os.getenv("ANTHROPIC_API_KEY", ""),
                "selected_llm": selected_llm or "claude-haiku-4-5-20251001",
            }
            llm = AnthropicLLM(user_controls_input).get_base_llm()
        else:
            raise HTTPException(
                status_code=400, detail=f"Unsupported provider: {provider}"
            )

        use_case = request.use_case or "basic_chatbot"

        # Resolve session ID early (needed for long memory)
        session_id = request.session_id or "default"
        
        # Log session information for debugging
        if use_case == "long_memory_chatbot":
            print(f"\n[LONG MEMORY] Session ID: {session_id}")
            print(f"[LONG MEMORY] Namespace: ('memories', '{session_id}')")

        # Build a lightweight graph for this request with the chosen LLM
        try:
            graph_builder = GraphBuilder(llm, {"selected_llm": selected_llm or ""})

            # For MCP chatbot, use pre-loaded tools
            tools = None
            if use_case == "mcp_chatbot":
                # Use globally loaded tools (loaded once at startup)
                tools = mcp_tools if mcp_tools is not None else await load_mcp_tools()

            # For long memory chatbot, create/use persistent store
            store = None
            if use_case == "long_memory_chatbot":
                global long_memory_store
                # Create persistent store if not exists (shared store with session namespacing)
                if long_memory_store is None:
                    embed_model = os.getenv("LANGMEM_EMBED_MODEL", "openai:text-embedding-3-small")
                    db_path = os.getenv("LANGMEM_DB_PATH", None)  # Optional custom DB path
                    long_memory_store = create_persistent_store(
                        db_path=db_path,
                        embed_model=embed_model
                    )
                    # Initialize the store (loads existing data from DB)
                    await long_memory_store.setup()
                    print(f"[LONG MEMORY] Store initialized and loaded from database")
                store = long_memory_store

            graph = await graph_builder.setup_graph(
                use_case, 
                tools=tools, 
                store=store, 
                session_id=session_id
            )
        except ValueError as graph_error:
            raise HTTPException(status_code=400, detail=str(graph_error))

        # Initialize session store if needed (for conversation history)
        session_key = f"{session_id}::{use_case}"
        if session_key not in session_store:
            session_store[session_key] = []

        # Build messages from stored history and current input
        messages = [SystemMessage(content="You are a helpful and efficient assistant.")]
        messages.extend(session_store[session_key])
        user_msg = HumanMessage(content=request.message)
        messages.append(user_msg)

        # Create state with all messages for context
        state = {"messages": messages}
        print("state-----", state)
        # Process with chatbot graph (use ainvoke for async graphs)
        result = await graph.ainvoke(state)
        # Extract response from graph result
        # The graph returns a state dict with messages, get the last message (should be AI response)
        result_messages = result.get("messages", [])
        if result_messages:
            last_message = result_messages[-1]
            # Extract content if it's a message object
            if hasattr(last_message, "content"):
                response_text = last_message.content
            elif isinstance(last_message, dict) and "content" in last_message:
                response_text = last_message["content"]
            else:
                response_text = str(last_message)
        else:
            response_text = "No response generated"

        # Persist history for this session (user + assistant)
        session_store[session_key].append(user_msg)
        session_store[session_key].append(AIMessage(content=response_text))

        return ChatResponse(response=response_text, status="success")

    except Exception as e:
        raise HTTPException(
            status_code=500, detail=f"Error processing chat request: {str(e)}"
        )


@app.post("/chat/reset")
async def reset_chat(request: ResetChatRequest):
    session_id = request.session_id or "default"
    use_case = request.use_case or "basic_chatbot"
    session_key = f"{session_id}::{use_case}"
    session_store.pop(session_key, None)
    return {"status": "success"}


@app.get("/sessions")
async def list_sessions(use_case: Optional[str] = None):
    """
    List all available sessions.
    If use_case is provided, only return sessions for that use case.
    """
    sessions = []
    for session_key in session_store.keys():
        parts = session_key.split("::")
        if len(parts) == 2:
            sess_id, sess_use_case = parts
            if use_case is None or sess_use_case == use_case:
                message_count = len(session_store[session_key])
                sessions.append({
                    "session_id": sess_id,
                    "use_case": sess_use_case,
                    "message_count": message_count,
                    "last_activity": "recent"  # Could be enhanced with timestamps
                })
    return {"sessions": sessions}


@app.get("/sessions/{session_id}")
async def get_session_info(session_id: str, use_case: Optional[str] = None):
    """
    Get information about a specific session.
    """
    sessions = []
    for session_key in session_store.keys():
        parts = session_key.split("::")
        if len(parts) == 2:
            sess_id, sess_use_case = parts
            if sess_id == session_id and (use_case is None or sess_use_case == use_case):
                message_count = len(session_store[session_key])
                sessions.append({
                    "session_id": sess_id,
                    "use_case": sess_use_case,
                    "message_count": message_count,
                    "last_activity": "recent"
                })
    return {"sessions": sessions}


def main():
    """Main function to run the FastAPI server"""
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)


if __name__ == "__main__":
    main()
