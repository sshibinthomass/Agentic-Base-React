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
from langgraph_agent.graphs.expert_graph import ExpertGraphBuilder

dotenv.load_dotenv()


class GraphBuilder:
    def __init__(self, model, user_controls_input: dict):
        self.llm = model
        self.user_controls_input = user_controls_input
        self.graph_builder = StateGraph(
            ChatbotState
        )  # StateGraph is a class in LangGraph that is used to build the graph

    def setup_graph(self, usecase: str, pdf_path: str = None):
        """
        Sets up the graph for the selected use case.

        Args:
            usecase: The use case identifier
            pdf_path: Optional PDF path for no_expert usecase
        """
        if usecase == "basic_chatbot":
            basic_chatbot_build_graph(self.graph_builder, self.llm)
            return self.graph_builder.compile()
        elif usecase == "no_expert":
            if not pdf_path:
                raise ValueError("PDF path is required for no_expert usecase")
            expert_graph = ExpertGraphBuilder(pdf_path)
            result = expert_graph.run_expert_identification()
            return result
        else:
            raise ValueError(f"Unsupported use case: {usecase}")


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
