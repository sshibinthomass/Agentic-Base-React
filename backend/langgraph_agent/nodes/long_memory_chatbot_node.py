import os
import sys
from pathlib import Path
from dotenv import load_dotenv
from typing import Optional
from langchain_core.messages import SystemMessage, AIMessage, HumanMessage
from langgraph.prebuilt import ToolNode
from langgraph.store.memory import InMemoryStore
from langgraph.store.base import BaseStore
from langgraph_agent.stores.persistent_store import create_persistent_store

# Add project root to path
current_file = Path(__file__).resolve()
project_root = current_file.parent.parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from langgraph_agent.states.chatbotState import ChatbotState
from langmem import create_manage_memory_tool, create_search_memory_tool

load_dotenv()


class LongMemoryChatbotNode:
    """
    Long Memory Chatbot node implementation with LangMem integration.
    Supports persistent memory across conversations using LangMem's memory tools.
    The agent can store and retrieve information from long-term memory.
    """

    def __init__(self, model, store: Optional[BaseStore] = None, session_id: str = "default"):
        """
        Initialize the chatbot node with an LLM and memory store.

        Args:
            model: The language model to use
            store: Optional BaseStore instance for memory persistence. If None, creates InMemoryStore.
            session_id: Session ID for namespacing memories
        """
        self.llm = model
        self.session_id = session_id
        
        # Create persistent store if not provided
        if store is None:
            # Default to OpenAI embeddings for compatibility
            # Can be configured via environment variable later
            embed_model = os.getenv("LANGMEM_EMBED_MODEL", "openai:text-embedding-3-small")
            db_path = os.getenv("LANGMEM_DB_PATH", None)  # Optional custom DB path
            store = create_persistent_store(
                db_path=db_path,
                embed_model=embed_model
            )
            # Initialize the store (loads existing data from DB)
            # Note: This is async, but we're in __init__, so we'll need to handle this
            # For now, we'll initialize it when the store is first used
        
        self.store = store
        self._store_initialized = False
        self.session_id = session_id
        
        # Create memory tools with session-based namespacing
        # Note: Tools will get store from graph context when executed
        # According to LangMem docs, tools automatically access store from graph runtime
        namespace = ("memories", session_id)
        # Try to pass store if the tools support it, otherwise rely on graph context
        try:
            self.manage_memory_tool = create_manage_memory_tool(namespace=namespace, store=store)
            self.search_memory_tool = create_search_memory_tool(namespace=namespace, store=store)
        except TypeError:
            # If tools don't accept store parameter, they'll get it from graph context
            self.manage_memory_tool = create_manage_memory_tool(namespace=namespace)
            self.search_memory_tool = create_search_memory_tool(namespace=namespace)
        
        # Combine memory tools
        self.memory_tools = [self.manage_memory_tool, self.search_memory_tool]
        
        # Create ToolNode for executing memory tools
        self.tool_node = ToolNode(self.memory_tools)
        
        # Bind memory tools to LLM
        self.llm = self.llm.bind_tools(self.memory_tools)

    async def process(self, state: ChatbotState) -> dict:
        """
        Processes the input state and generates a chatbot response with memory support.
        Returns the AI response as an AIMessage object to maintain conversation history.
        Handles memory tool calls by executing them and getting the final response.
        Supports multiple rounds of tool calls if needed.
        """
        # Initialize store if needed (for persistent stores)
        if not self._store_initialized and hasattr(self.store, 'setup'):
            await self.store.setup()
            self._store_initialized = True
            print(f"[LONG MEMORY] Store initialized for session: {self.session_id}")
        
        # Create a copy of messages to avoid modifying the input state
        messages = list(state["messages"])

        # PROACTIVE MEMORY SEARCH: Search memory before generating response
        # Extract the latest user message for memory search
        user_query = None
        for msg in reversed(messages):
            if isinstance(msg, HumanMessage):
                user_query = msg.content
                break
        
        # Proactively search memory if we have a user query
        retrieved_memories_text = None
        if user_query and self.store and self._store_initialized:
            try:
                namespace = ("memories", self.session_id)
                print(f"[LONG MEMORY] Proactively searching memory for query: '{user_query[:50]}...'")
                print(f"[LONG MEMORY] Using namespace: {namespace}")
                
                # Use the store directly to search for memories
                # This is more reliable than invoking the tool directly
                try:
                    # First, try to use the store's search method if available (for semantic search)
                    # Note: BaseStore.asearch() signature is asearch(query, namespace) not asearch(namespace, query)
                    if hasattr(self.store, 'asearch'):
                        try:
                            # Try correct signature: asearch(query, namespace=namespace)
                            results = await self.store.asearch(user_query, namespace=namespace)
                            if results:
                                memory_contents = []
                                for result in results:
                                    if isinstance(result, dict):
                                        content = result.get('content', result.get('memory', str(result)))
                                    else:
                                        content = str(result)
                                    memory_contents.append(content)
                                if memory_contents:
                                    retrieved_memories_text = "\n".join(memory_contents)
                                    print(f"[LONG MEMORY] Semantic search found {len(memory_contents)} relevant memories")
                        except Exception as search_error:
                            print(f"[LONG MEMORY] Semantic search failed: {search_error}, using fallback")
                    
                    # Fallback: list all memories and get their values
                    # This ensures we get memories even if semantic search isn't available
                    if not retrieved_memories_text:
                        all_keys = await self.store.alist(namespace)
                        print(f"[LONG MEMORY] Found {len(all_keys)} total memories in namespace")
                        if all_keys:
                            # Get all memories (or last N if there are many)
                            memory_contents = []
                            keys_to_check = all_keys[-10:] if len(all_keys) > 10 else all_keys
                            for key in keys_to_check:
                                value = await self.store.aget(namespace, key)
                                if value:
                                    if isinstance(value, dict):
                                        content = value.get('content', value.get('memory', value.get('text', str(value))))
                                    else:
                                        content = str(value)
                                    memory_contents.append(content)
                            if memory_contents:
                                retrieved_memories_text = "\n".join(memory_contents)
                                print(f"[LONG MEMORY] Retrieved {len(memory_contents)} memories from store")
                            else:
                                print(f"[LONG MEMORY] No memory content found in stored values")
                        else:
                            print(f"[LONG MEMORY] No memories found in namespace")
                except Exception as store_error:
                    print(f"[LONG MEMORY] Error accessing store: {store_error}")
                    import traceback
                    traceback.print_exc()
            except Exception as e:
                print(f"[LONG MEMORY] Error during proactive search: {e}")
                import traceback
                traceback.print_exc()
        
        # Add system prompt to guide the agent on memory usage
        system_prompt = """You are a helpful assistant with long-term memory capabilities. 
You can remember important information from conversations using your memory tools.
When users share preferences, facts, or information they want you to remember, use the manage_memory tool to store it.
When you need to recall past information, use the search_memory tool to find relevant memories.
Be proactive about storing important information and retrieving it when relevant."""
        
        # Inject retrieved memories into context if found
        if retrieved_memories_text:
            memory_context = f"Relevant information from previous conversations:\n{retrieved_memories_text}"
            system_prompt += f"\n\n{memory_context}"
            print(f"[LONG MEMORY] Injected memories into context")
        
        # Prepend system message if not already present
        if not any(isinstance(msg, SystemMessage) for msg in messages):
            messages = [SystemMessage(content=system_prompt)] + messages
        else:
            # Update existing system message with memory context
            for i, msg in enumerate(messages):
                if isinstance(msg, SystemMessage):
                    messages[i] = SystemMessage(content=system_prompt)
                    break

        # Handle tool calls in a loop (in case multiple rounds are needed)
        max_tool_iterations = 10  # Prevent infinite loops
        iteration = 0

        while iteration < max_tool_iterations:
            iteration += 1

            # Get response from LLM
            response = self.llm.invoke(messages)

            # Ensure response is an AIMessage
            if not isinstance(response, AIMessage):
                if hasattr(response, "content") and hasattr(response, "type"):
                    response = AIMessage(content=response.content)
                elif isinstance(response, dict) and "content" in response:
                    response = AIMessage(content=response["content"])
                elif isinstance(response, str):
                    response = AIMessage(content=response)
                else:
                    response = AIMessage(content=str(response))

            # Check if response has tool calls
            if hasattr(response, "tool_calls") and response.tool_calls:
                # Display tool call information
                print("\n\n< MEMORY TOOL CALLS DETECTED >\n")

                for tool_call in response.tool_calls:
                    # Handle both dict and object-style tool calls
                    if isinstance(tool_call, dict):
                        tool_name = tool_call.get("name", "unknown")
                        tool_args = tool_call.get("args", {})
                        tool_id = tool_call.get("id", "")
                    else:
                        # Handle ToolCall objects with attributes
                        tool_name = getattr(tool_call, "name", "unknown")
                        tool_id = getattr(tool_call, "id", "")

                        # Get args
                        tool_args = {}
                        if hasattr(tool_call, "args"):
                            args_val = tool_call.args
                            if isinstance(args_val, dict):
                                tool_args = args_val
                            elif isinstance(args_val, str):
                                try:
                                    import json
                                    tool_args = json.loads(args_val)
                                except (json.JSONDecodeError, ValueError):
                                    tool_args = {"raw": args_val}
                            elif args_val is not None:
                                tool_args = args_val

                        # Also try to get from __dict__ if available
                        if not tool_args and hasattr(tool_call, "__dict__"):
                            tool_args = tool_call.__dict__.get("args", {})

                    print(f"< MEMORY TOOL CALL: {tool_name} >")
                    if tool_id:
                        print(f"Tool ID: {tool_id}")
                    if tool_args:
                        print(f"Arguments: {tool_args}")
                    print(f"[LONG MEMORY] Executing tool: {tool_name} with args: {tool_args}")
                    if tool_name == "manage_memory":
                        print(f"[LONG MEMORY] Storing memory for session: {self.session_id}")
                    elif tool_name == "search_memory":
                        print(f"[LONG MEMORY] Searching memory for session: {self.session_id}")
                    print()

                # Add the AI response with tool calls to messages
                messages.append(response)

                # Execute memory tools using ToolNode
                # Note: Memory tools need the store to be available in the graph context
                # The store should be available from the graph's runtime context when compiled with store
                tool_state = {"messages": messages}
                
                print(f"[LONG MEMORY] Executing tools with store available: {self.store is not None}")
                print(f"[LONG MEMORY] Store initialized: {self._store_initialized}")
                
                # Execute tools (memory tools will use the store from graph context)
                # The store is passed to the graph when compiling, so tools should have access
                try:
                    tool_result = await self.tool_node.ainvoke(tool_state)
                    tool_messages = tool_result.get("messages", [])
                    print(f"[LONG MEMORY] Tool execution completed, got {len(tool_messages)} tool messages")
                    for tool_msg in tool_messages:
                        if hasattr(tool_msg, 'content'):
                            print(f"[LONG MEMORY] Tool message content: {str(tool_msg.content)[:100]}...")
                    messages.extend(tool_messages)
                except Exception as tool_exec_error:
                    print(f"[LONG MEMORY] Error executing tools: {tool_exec_error}")
                    import traceback
                    traceback.print_exc()
                    # Continue anyway - don't break the flow

                # Continue loop to get final response after tool execution
                continue

            # No tool calls, return the response
            return {"messages": [response]}

        # If we've exceeded max iterations, return the last response
        return {"messages": [response]}


if __name__ == "__main__":
    import asyncio
    from langgraph_agent.llms.openai_llm import OpenAiLLM
    from langchain_core.messages import HumanMessage, SystemMessage

    async def main():
        # Create LLM instance
        user_controls_input = {
            "OPENAI_API_KEY": os.getenv("OPENAI_API_KEY"),
            "selected_llm": "gpt-4o-mini",
        }
        llm = OpenAiLLM(user_controls_input)
        llm = llm.get_base_llm()

        # Create LongMemoryChatbotNode instance
        node = LongMemoryChatbotNode(llm, session_id="test_session")

        # Test storing a memory
        print("\n ----  TEST: Storing Memory  ---- \n")
        state1 = {
            "messages": [
                SystemMessage(content="You are a helpful assistant."),
                HumanMessage(content="Remember that I prefer dark mode for my applications."),
            ]
        }
        result1 = await node.process(state1)
        print("Response:", result1["messages"][-1].content if result1.get("messages") else "No response")

        # Test retrieving the memory
        print("\n ----  TEST: Retrieving Memory  ---- \n")
        state2 = {
            "messages": [
                SystemMessage(content="You are a helpful assistant."),
                HumanMessage(content="What are my preferences for application themes?"),
            ]
        }
        result2 = await node.process(state2)
        print("Response:", result2["messages"][-1].content if result2.get("messages") else "No response")

    # Run the async main function
    import nest_asyncio
    nest_asyncio.apply()
    asyncio.run(main())
