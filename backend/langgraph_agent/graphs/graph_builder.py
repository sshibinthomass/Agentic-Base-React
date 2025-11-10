from langgraph.graph import StateGraph
from langgraph.graph import START, END

from pathlib import Path
import sys
import dotenv

current_file = Path(__file__).resolve()
project_root = current_file.parent.parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from langgraph_agent.states.state import State  # ignoring the import error
from langgraph_agent.nodes.basic_chatbot_node import BasicChatbotNode

dotenv.load_dotenv()


class GraphBuilder:
    def __init__(self, model, user_controls_input: dict, message: str):
        self.llm = model
        self.user_controls_input = user_controls_input
        self.message = message
        self.current_llm = user_controls_input["selected_llm"]
        self.graph_builder = StateGraph(
            State
        )  # StateGraph is a class in LangGraph that is used to build the graph

    def basic_chatbot_build_graph(self):
        """
        Builds a basic chatbot graph using LangGraph.
        This method initializes a chatbot node using the `BasicChatbotNode` class
        and integrates it into the graph. The chatbot node is set as both the
        entry and exit point of the graph.
        """
        self.basic_chatbot_node = BasicChatbotNode(self.llm)

        self.graph_builder.add_node("chatbot", self.basic_chatbot_node.process)
        self.graph_builder.add_edge(START, "chatbot")
        self.graph_builder.add_edge("chatbot", END)

    def setup_graph(self, usecase: str):
        """
        Sets up the graph for the selected use case.
        """
        if usecase == "basic_chatbot":
            self.basic_chatbot_build_graph()
        elif usecase == "weather_chatbot":
            self.basic_chatbot_build_graph()
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
    graph_builder = GraphBuilder(llm, user_controls_input, "Hi")
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
