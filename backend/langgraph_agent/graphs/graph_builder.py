from langgraph.graph import StateGraph

from pathlib import Path
import sys
import dotenv

current_file = Path(__file__).resolve()
project_root = current_file.parent.parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from langgraph_agent.states.chatbotState import (
    ChatbotState,
)  # ignoring the import error
from langgraph_agent.graphs.basic_chatbot_graph import basic_chatbot_build_graph

dotenv.load_dotenv()


class GraphBuilder:
    def __init__(self, model, user_controls_input: dict):
        self.llm = model
        self.user_controls_input = user_controls_input
        self.graph_builder = StateGraph(
            ChatbotState
        )  # StateGraph is a class in LangGraph that is used to build the graph

    def setup_graph(self, usecase: str):
        """
        Sets up the graph for the selected use case.
        """
        if usecase == "basic_chatbot":
            basic_chatbot_build_graph(self.graph_builder, self.llm)
        elif usecase == "weather_chatbot":
            basic_chatbot_build_graph(self.graph_builder, self.llm)
        else:
            raise ValueError(f"Unsupported use case: {usecase}")

        return self.graph_builder.compile()


if __name__ == "__main__":
    from langgraph_agent.llms.groq_llm import GroqLLM
    from langchain_core.messages import HumanMessage, SystemMessage
    import os

    user_controls_input = {
        "GROQ_API_KEY": os.getenv("GROQ_API_KEY"),
        "selected_llm": "openai/gpt-oss-20b",
    }
    llm = GroqLLM(user_controls_input)
    llm = llm.get_base_llm()
    graph_builder = GraphBuilder(llm, user_controls_input)
    graph = graph_builder.setup_graph("basic_chatbot")

    # Create input state for the graph
    initial_state = {
        "messages": [
            SystemMessage(content="You are a helpful assistant."),
            HumanMessage(content="Hello, how are you?"),
        ]
    }

    # Run the graph and print the output
    result = graph.invoke(initial_state)
    print("Graph Output:", result)
