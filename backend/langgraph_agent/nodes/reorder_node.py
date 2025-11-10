import sys
from pathlib import Path
from dotenv import load_dotenv
import json
from openai import OpenAI

load_dotenv()


# Setup path for local imports
current_file = Path(__file__).resolve()
project_root = current_file.parent.parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from langgraph_agent.states.state import ExpertSearchState
from langgraph_agent.tools.return_prompt import return_prompt


class ReorderNode:
    def __init__(self, model):
        self.model = model
        self._load_prompts()

    def _load_prompts(self):
        """Load system prompts for reordering"""
        self.system_prompt_reorder = return_prompt("reorder_tool")

    def reorder_roles_agent(self, state: ExpertSearchState) -> ExpertSearchState:
        """
        Use LLM to intelligently reorder roles based on roadmap analysis.

        Args:
            state: ExpertSearchState with roles and roadmap that need to be reordered

        Returns:
            ExpertSearchState with roles reordered based on roadmap analysis
        """
        try:
            roles_needed = state.get("roles_needed", [])
            roadmap = state.get("roadmap", "")

            if not roles_needed:
                print("⚠️  No roles found to reorder")
                return state

            if not roadmap:
                print("⚠️  No roadmap found, using original order")
                return state

            print("🤖 Using LLM to analyze roadmap and reorder broader_area field...")

            # Prepare the input for the LLM
            roles_json = json.dumps(roles_needed, indent=2)

            response = self.model.chat.completions.create(
                model="gpt-4o",
                messages=[
                    {
                        "role": "system",
                        "content": self.system_prompt_reorder,
                    },
                    {
                        "role": "user",
                        "content": f"""
                        ROADMAP:
                        {roadmap}

                        CURRENT ROLES:
                        {roles_json}

                        Please analyze ONLY the roadmap content and reorder the broader_area field of roles based on the roadmap's phase sequence. Focus exclusively on the roadmap to determine which broader_area values should come first, second, third, etc.
                        """,
                    },
                ],
                temperature=0.3,
            )

            response_text = response.choices[0].message.content.strip()

            try:
                # Remove markdown fences if present
                if "```json" in response_text:
                    response_text = response_text.split("```json")[1].split("```")[0]
                elif "```" in response_text:
                    response_text = response_text.split("```")[1].split("```")[0]

                # Parse the reordered roles
                reordered_roles = json.loads(response_text.strip())

                # Validate that we got the expected number of roles
                if len(reordered_roles) != len(roles_needed):
                    print(
                        "⚠️  LLM returned different number of roles, using original order"
                    )
                    return state

                # Ensure order field is present and sequential
                for index, role in enumerate(reordered_roles, 1):
                    if "order" not in role:
                        role["order"] = index
                    else:
                        role["order"] = index  # Ensure sequential numbering

                # Update the state with reordered roles
                state["roles_needed"] = reordered_roles

                print(
                    f"✅ LLM reordered {len(reordered_roles)} broader_area fields based on roadmap analysis"
                )

                # Show the reordered sequence
                for i, role in enumerate(reordered_roles, 1):
                    broader_area = role.get("broader_area", "Unknown")
                    print(f"  Step {i}: {broader_area}")

                return state

            except json.JSONDecodeError as e:
                print(f"⚠️  Error parsing LLM response: {e}")
                print(f"Raw response: {response_text[:200]}...")
                return state

        except Exception as e:
            print(f"⚠️  Error in LLM reordering: {e}")
            return state


if __name__ == "__main__":
    # Load real data from JSON file
    from langgraph_agent.states.state import RoleNeeded, ExpertsList
    from openai import OpenAI
    import json
    from pathlib import Path

    # Load the real expert search results
    json_file_path = (
        Path(__file__).parent.parent.parent.parent
        / "json_data"
        / "expert_search_results_20251026_015511.json"
    )

    with open(json_file_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    # Extract roles and roadmap from the JSON data
    roles_data = data.get("roles_needed", [])
    roadmap = data.get("roadmap", "")

    # Convert to RoleNeeded objects
    test_roles = []
    for role_data in roles_data:
        # Convert experts_list to proper format
        experts_list = ExpertsList(
            enriched_scored_experts=role_data.get("experts_list", {}).get(
                "enriched_scored_experts", []
            )
        )

        role = RoleNeeded(
            order=role_data.get("order", 0),
            broader_area=role_data.get("broader_area", ""),
            role_description=role_data.get("role_description", ""),
            search_terms=role_data.get("search_terms", []),
            why_needed=role_data.get("why_needed", ""),
            experts_list=experts_list,
            talking_points=role_data.get(
                "talking_points", {"talking_points": [], "blocker_points": []}
            ),
        )
        test_roles.append(role)

    print(f"Loaded {len(test_roles)} roles from JSON file")
    print(f"Roadmap length: {len(roadmap)} characters")

    # Create test state with real data
    test_state = ExpertSearchState(roles_needed=test_roles, roadmap=roadmap)

    # Test LLM-based reordering
    reorder_node = ReorderNode(model=OpenAI())
    result = reorder_node.reorder_roles_agent(test_state)

    print("\n" + "=" * 50)
    print("UPDATED STATE AFTER REORDERING:")
    print("=" * 50)
    print(json.dumps(result, indent=2, default=str))
