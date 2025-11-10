from langgraph.graph import StateGraph
from langgraph.graph import START, END
from dotenv import load_dotenv
import sys
import time
from openai import OpenAI
from langchain_openai import ChatOpenAI
from pathlib import Path
from typing import Optional, Callable
import json
from datetime import datetime

load_dotenv()


# Setup path for local imports
current_file = Path(__file__).resolve()
project_root = current_file.parent.parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))


from langgraph_agent.states.state import ExpertSearchState
from langgraph_agent.nodes.extract_expertise_areas_node import (
    ExtractExpertiseAreasNode,
)
from langgraph_agent.nodes.expert_search_node import ExpertSearchNode
from langgraph_agent.nodes.get_purpose_description import (
    GetPurposeDescriptionNode,
)
from langgraph_agent.nodes.get_roadmap_node import (
    GetRoadmapNode,
)
from langgraph_agent.nodes.update_talking_points_node import (
    UpdateTalkingPointsNode,
)
from langgraph_agent.nodes.reorder_node import ReorderNode


class ExpertGraphBuilder:
    def __init__(
        self,
        pdf_path: Optional[str] = None,
        progress_callback: Optional[Callable[[str, float, str], None]] = None,
    ):
        self.pdf_path = pdf_path
        self.progress_callback = progress_callback
        self.graph_builder = StateGraph(ExpertSearchState)
        self._initialize_nodes()
        self._captured_output = []

    def _capture_print(self, message: str, step: str = None, progress: float = None):
        """Capture print output and send to progress callback"""
        self._captured_output.append(message)
        print(message)  # Still print to console

        # Send to progress callback if available
        if self.progress_callback and step and progress is not None:
            # Create a detailed message with the captured output
            details = "\n".join(self._captured_output[-10:])  # Last 10 lines
            self.progress_callback(step, progress, details)

    def _update_progress(self, step: str, progress: float, message: str = None):
        """Update progress with clean, meaningful messages"""
        if message:
            self._captured_output.append(message)
            print(message)  # Still print to console

        # Send to progress callback if available
        if self.progress_callback:
            # Create a detailed message with the captured output
            details = "\n".join(self._captured_output[-10:])  # Last 10 lines
            self.progress_callback(step, progress, details)

    def _initialize_nodes(self):
        """Initialize all the expert identification nodes"""
        # Initialize models
        openai_client = OpenAI()
        fast_model = ChatOpenAI(model="gpt-3.5-turbo")  # Faster model for scoring

        # Initialize nodes
        self.extract_node = ExtractExpertiseAreasNode(model=openai_client)
        self.search_node = ExpertSearchNode(
            model=fast_model
        )  # Use faster model for scoring

        # Initialize sequential processing nodes
        self.purpose_node = GetPurposeDescriptionNode(model=openai_client)
        self.roadmap_node = GetRoadmapNode(model=openai_client)
        self.talking_points_node = UpdateTalkingPointsNode(model=openai_client)
        self.reorder_node = ReorderNode(model=openai_client)

    def build_expert_identification_graph(self):
        """
        Builds the expert identification pipeline graph.
        Flow: Extract Expertise Areas → Search for Experts → Purpose Descriptions → Roadmap → Talking Points → Reorder
        """
        # Add nodes to the graph
        self.graph_builder.add_node(
            "extract_expertise", self._extract_expertise_wrapper
        )
        self.graph_builder.add_node("search_experts", self._search_experts_wrapper)

        # Add sequential processing nodes
        self.graph_builder.add_node(
            "purpose_descriptions", self._purpose_descriptions_wrapper
        )
        self.graph_builder.add_node("roadmap", self._roadmap_wrapper)
        self.graph_builder.add_node("talking_points", self._talking_points_wrapper)
        self.graph_builder.add_node("reorder", self._reorder_wrapper)

        # Define the sequential flow
        self.graph_builder.add_edge(START, "extract_expertise")
        self.graph_builder.add_edge("extract_expertise", "search_experts")
        self.graph_builder.add_edge("search_experts", "purpose_descriptions")
        self.graph_builder.add_edge("purpose_descriptions", "roadmap")
        self.graph_builder.add_edge("roadmap", "talking_points")
        self.graph_builder.add_edge("talking_points", "reorder")
        self.graph_builder.add_edge("reorder", END)

        return self.graph_builder.compile()

    def _extract_expertise_wrapper(self, state: ExpertSearchState) -> ExpertSearchState:
        """
        Wrapper function for extract expertise areas node.
        Extracts expertise areas from the PDF document.
        """
        if not self.pdf_path:
            raise ValueError("PDF path is required for extracting expertise areas")

        self._update_progress(
            "extract_expertise",
            0.15,
            "Analyzing document to identify required expertise areas...",
        )
        start_time = time.time()

        result = self.extract_node.extract_expertise_areas_agent(self.pdf_path)

        self._update_progress(
            "extract_expertise", 0.30, "Processing expertise requirements..."
        )

        end_time = time.time()
        duration = end_time - start_time
        roles_count = len(result.get("roles_needed", []))

        self._update_progress(
            "extract_expertise",
            0.45,
            f"Found {roles_count} expertise areas in {duration:.1f}s",
        )

        # Only show the first 3 roles to keep it clean
        for i, role in enumerate(result.get("roles_needed", [])[:3], 1):
            self._update_progress(
                "extract_expertise", 0.45, f"  • {role.get('broader_area', 'Unknown')}"
            )

        return result

    def _search_experts_wrapper(self, state: ExpertSearchState) -> ExpertSearchState:
        """
        Wrapper function for expert search node.
        Searches for experts and scores them based on the extracted expertise areas.
        """
        roles_count = len(state.get("roles_needed", []))
        self._update_progress(
            "search_experts",
            0.45,
            f"Searching for experts across {roles_count} roles...",
        )
        start_time = time.time()

        result = self.search_node.expert_search_agent(state)

        self._update_progress("search_experts", 0.70, "Scoring and ranking experts...")

        end_time = time.time()
        duration = end_time - start_time
        total_experts = 0

        # Count total experts
        for role in result.get("roles_needed", []):
            experts = role.get("experts_list", {}).get("enriched_scored_experts", [])
            total_experts += len(experts)

        self._update_progress(
            "search_experts",
            0.85,
            f"Found {total_experts} qualified experts in {duration:.1f}s",
        )

        # Show top expert for each role (limit to first 3)
        for i, role in enumerate(result.get("roles_needed", [])[:3], 1):
            experts = role.get("experts_list", {}).get("enriched_scored_experts", [])
            if experts:
                top_expert = max(experts, key=lambda x: x.get("total_score", 0))
                score = int(top_expert.get("total_score", 0))
                self._update_progress(
                    "search_experts",
                    0.85,
                    f"  • {role.get('broader_area', 'Unknown')}: {top_expert.get('first_name', '')} {top_expert.get('last_name', '')} ({score}%)",
                )

        return result

    def _purpose_descriptions_wrapper(
        self, state: ExpertSearchState
    ) -> ExpertSearchState:
        """
        Wrapper function for purpose descriptions node.
        Generates purpose descriptions for all experts in the state.
        """
        if not self.pdf_path:
            raise ValueError("PDF path is required for generating purpose descriptions")

        self._update_progress(
            "purpose_descriptions", 0.75, "Generating detailed expert profiles..."
        )
        start_time = time.time()

        result = self.purpose_node.update_purpose_descriptions_agent(
            state, self.pdf_path
        )

        self._update_progress(
            "purpose_descriptions", 0.80, "Enriching expert information..."
        )

        end_time = time.time()
        duration = end_time - start_time

        self._update_progress(
            "purpose_descriptions",
            0.85,
            f"Expert profiles completed in {duration:.1f}s",
        )
        return result

    def _roadmap_wrapper(self, state: ExpertSearchState) -> ExpertSearchState:
        """
        Wrapper function for roadmap node.
        Generates roadmap for the research result.
        """
        if not self.pdf_path:
            raise ValueError("PDF path is required for generating roadmap")

        print("📄 [ROADMAP] Starting roadmap generation...")
        start_time = time.time()

        if self.progress_callback:
            self.progress_callback("roadmap", 0.80)

        result = self.roadmap_node.generate_roadmap_agent(state, self.pdf_path)

        if self.progress_callback:
            self.progress_callback("roadmap", 0.85)

        end_time = time.time()
        duration = end_time - start_time

        print(f"✅ [ROADMAP] Completed in {duration:.2f}s - Roadmap generated")
        return result

    def _talking_points_wrapper(self, state: ExpertSearchState) -> ExpertSearchState:
        """
        Wrapper function for talking points node.
        Generates talking points for all roles in the state.
        """
        if not self.pdf_path:
            raise ValueError("PDF path is required for generating talking points")

        print("🗺️  [TALKING POINTS] Starting talking points generation...")
        start_time = time.time()

        if self.progress_callback:
            self.progress_callback("talking_points", 0.85)

        result = self.talking_points_node.update_talking_points_agent(
            state, self.pdf_path
        )

        if self.progress_callback:
            self.progress_callback("talking_points", 0.95)

        end_time = time.time()
        duration = end_time - start_time

        print(
            f"✅ [TALKING POINTS] Completed in {duration:.2f}s - Talking points generated for all roles"
        )
        return result

    def _reorder_wrapper(self, state: ExpertSearchState) -> ExpertSearchState:
        """
        Wrapper function for reorder node.
        Reorders roles based on their order field to ensure proper step-by-step sequence.
        """
        print("🔄 [REORDER] Starting role reordering...")
        start_time = time.time()

        if self.progress_callback:
            self.progress_callback("reorder", 0.95)

        result = self.reorder_node.reorder_roles_agent(state)

        if self.progress_callback:
            self.progress_callback("reorder", 1.0)

        end_time = time.time()
        duration = end_time - start_time

        print(
            f"✅ [REORDER] Completed in {duration:.2f}s - Roles reordered by step sequence"
        )
        return result

    def _export_to_json(self, result: ExpertSearchState) -> str:
        """
        Export the expert identification results to JSON file in json_data/ directory.

        Args:
            result: The ExpertSearchState to export

        Returns:
            str: Path to the exported JSON file
        """
        # Create json_data directory if it doesn't exist
        json_data_dir = Path("json_data")
        json_data_dir.mkdir(exist_ok=True)

        # Generate filename with timestamp
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        json_filename = f"expert_search_results_{timestamp}.json"
        json_path = json_data_dir / json_filename

        # Export the result
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(result, f, indent=2, ensure_ascii=False)

        print(f"📁 Results exported to: {json_path}")
        return str(json_path)

    def _load_from_json(self, json_path: str) -> ExpertSearchState:
        """
        Load expert identification results from a JSON file.

        Args:
            json_path: Path to the JSON file to load

        Returns:
            ExpertSearchState: The loaded expert search state
        """
        json_file = Path(json_path)

        if not json_file.exists():
            raise FileNotFoundError(f"JSON file not found: {json_path}")

        # Load the JSON file
        with open(json_file, "r", encoding="utf-8") as f:
            result = json.load(f)

        print(f"📂 Results loaded from: {json_path}")
        return result

    def run_expert_identification(self):
        """
        Run the complete expert identification pipeline.

        Returns:
            ExpertSearchState with scored and ranked experts
        """
        self._update_progress(
            "research", 0.05, "Initializing expert identification pipeline..."
        )
        pipeline_start_time = time.time()

        # Build and compile the graph
        graph = self.build_expert_identification_graph()

        # Initialize with empty state
        initial_state = ExpertSearchState(roles_needed=[])

        # Execute the graph
        final_result = graph.invoke(initial_state)

        pipeline_end_time = time.time()
        total_duration = pipeline_end_time - pipeline_start_time

        # Final summary
        total_roles = len(final_result.get("roles_needed", []))
        total_experts = sum(
            len(role.get("experts_list", {}).get("enriched_scored_experts", []))
            for role in final_result.get("roles_needed", [])
        )

        print(
            f"📈 Results: {total_roles} roles identified, {total_experts} experts found and scored"
        )
        print(
            "⚡ Sequential processing: search → purpose descriptions → roadmap → talking_points"
        )

        # Export to JSON in json_data/ directory
        export_path = self._export_to_json(final_result)
        self._update_progress("done", 1.0, "Results saved successfully")

        return final_result


if __name__ == "__main__":
    # Example usage
    pdf_path = "/Users/qtf4195/Github_Projects/Agentic-Base-React/backend/data/BMW_CX_Hub_Project_Idea_Proposal1.pdf"

    # Create and run the expert identification pipeline
    expert_graph = ExpertGraphBuilder(pdf_path)
    result = expert_graph.run_expert_identification()

    # The JSON export is now handled automatically in run_expert_identification()
    print("✅ Expert identification completed and exported to json_data/")
