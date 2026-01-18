from langgraph.graph import START, END
from typing import Optional
from langgraph.store.base import BaseStore

from pathlib import Path
import sys

current_file = Path(__file__).resolve()
project_root = current_file.parent.parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from langgraph_agent.nodes.long_memory_chatbot_node import LongMemoryChatbotNode


def long_memory_chatbot_build_graph(graph_builder, llm, store: Optional[BaseStore] = None, session_id: str = "default"):
    """
    Builds a long memory chatbot graph using LangGraph with LangMem integration.
    This method initializes a chatbot node using the `LongMemoryChatbotNode` class
    with memory tools support and integrates it into the graph. The chatbot node
    is set as both the entry and exit point of the graph.

    Args:
        graph_builder: The StateGraph instance to add nodes to
        llm: The language model to use for the chatbot
        store: Optional BaseStore instance for memory persistence. If None, node will create InMemoryStore.
        session_id: Session ID for namespacing memories
    """
    long_memory_chatbot_node = LongMemoryChatbotNode(llm, store=store, session_id=session_id)

    # LangGraph can handle async nodes, so we can add the async process method directly
    graph_builder.add_node("long_memory_chatbot", long_memory_chatbot_node.process)
    graph_builder.add_edge(START, "long_memory_chatbot")
    graph_builder.add_edge("long_memory_chatbot", END)


if __name__ == "__main__":
    import asyncio
    import os
    from dotenv import load_dotenv
    from langgraph.graph import StateGraph
    from langgraph_agent.llms.openai_llm import OpenAiLLM
    from langchain_core.messages import HumanMessage, SystemMessage
    from langgraph_agent.states.chatbotState import ChatbotState

    load_dotenv()

    async def main():
        # Create LLM instance
        user_controls_input = {
            "OPENAI_API_KEY": os.getenv("OPENAI_API_KEY"),
            "selected_llm": "gpt-4o-mini",
        }
        llm = OpenAiLLM(user_controls_input)
        llm = llm.get_base_llm()

        # Create graph builder
        graph_builder = StateGraph(ChatbotState)

        # Build the graph with long memory support
        long_memory_chatbot_build_graph(graph_builder, llm, session_id="test_session")

        # Compile the graph
        graph = graph_builder.compile()

        # Create input state for the graph
        initial_state = {
            "messages": [
                SystemMessage(content="You are a helpful and efficient assistant."),
                HumanMessage(content="Remember that I love chocolate ice cream."),
            ]
        }

        # Run the graph and print the output
        result = await graph.ainvoke(initial_state)
        print("result: ", result)
        print("\n---- Graph Result ----")
        result_messages = result.get("messages", [])
        if result_messages:
            last_message = result_messages[-1]
            if hasattr(last_message, "content"):
                print(last_message.content)
            else:
                print(result)

    # Run the async main function
    import nest_asyncio

    nest_asyncio.apply()
    asyncio.run(main())
