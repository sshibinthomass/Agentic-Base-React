import json
import sys
from pathlib import Path
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()

# Setup path for local imports
current_file = Path(__file__).resolve()
project_root = current_file.parent.parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

# Import after path setup
try:
    from langgraph_agent.states.state import (
        ExpertSearchState,
        TalkingPoints,
        RoleNeeded,
        ExpertsList,
    )
except ImportError:
    # Fallback import if path setup fails
    import sys

    sys.path.append(str(Path(__file__).parent.parent.parent.parent))
    from langgraph_agent.states.state import (
        ExpertSearchState,
        TalkingPoints,
        RoleNeeded,
        ExpertsList,
    )


class UpdateTalkingPointsNode:
    def __init__(self, model):
        self.model = model

    def generate_talking_points_for_role(
        self, role: RoleNeeded, pdf_path: Path
    ) -> TalkingPoints:
        """Generate talking points for a specific role based on the PDF.

        Args:
            role: RoleNeeded object containing role information
            pdf_path: Path to the PDF file to analyze

        Returns:
            TalkingPoints object with talking_points and blocker_points
        """
        try:
            client = self.model

            file = client.files.create(file=open(pdf_path, "rb"), purpose="user_data")

            system_prompt = f"""
                CRITICAL: Read the attached PDF project proposal carefully. Generate talking points and blockers that reference the project BUT keep questions conversational and open.
                
                Generate conversation talking points for engaging with a specific expert on the BMW project described in the attached PDF.
                
                EXPERT ROLE NEEDED:
                Domain: {role.get("broader_area", "")}
                Role: {role.get("role_description", "")}
                {f"Why Needed: {role.get('why_needed', '')}" if role.get("why_needed") else ""}
                
                YOUR TASK:
                Generate TWO types of items based on the ACTUAL PROJECT DETAILS in the PDF:
                
                1. TALKING POINTS (5-6 items):
                  ⚠️ BALANCE: Reference the project but keep questions short and conversational
                  - TOO generic: "How do you handle sales operations?" ❌
                  - TOO specific: "How would your dealer operations team handle the SprachAssist multilingual service hub rollout across 200 service centers, especially considering the integration of systems like ISTA/AIR, DMS, and Retail Next?" ❌❌
                  - JUST RIGHT: "How would your team approach rolling out the SprachAssist platform to our service network?" ✓
                  
                  Requirements:
                  - Reference the project name or key feature (e.g., "SprachAssist", "multilingual support")
                  - Keep questions under 20 words when possible
                  - Frame as open conversation starters, not detailed specifications
                  - Show you know the project, but let the expert explain details
                
                2. BLOCKER POINTS (3-4 items):
                  ⚠️ BALANCE: Mention project context but don't list every detail
                  - TOO generic: "Lack of training" ❌
                  - TOO specific: "Integration timeline between SprachAssist AI and existing IST/AIR service systems may conflict with Q2 pilot deadline mentioned on page 3" ❌❌
                  - JUST RIGHT: "Integration timelines between the AI platform and existing service systems" ✓
                  
                  Requirements:
                  - Reference specific project elements (systems, goals, timeline) but be concise
                  - One sentence maximum per blocker
                  - Focus on the challenge, not every detail
                
                GUIDELINES:
                - Conversational tone - like you're preparing for a coffee chat, not a detailed review
                - Reference enough to show you read the proposal, but leave room for discussion
                - Questions should invite explanation, not just yes/no answers
                
                Return ONLY valid JSON in this exact format:
                {{
                  "talking_points": [
                    "Short project-aware question 1",
                    "Short project-aware question 2",
                    "Short project-aware question 3",
                    "Short project-aware question 4",
                    "Short project-aware question 5"
                  ],
                  "blocker_points": [
                    "Concise project-specific blocker 1",
                    "Concise project-specific blocker 2",
                    "Concise project-specific blocker 3"
                  ]
                }}
            """

            response = client.responses.create(
                model="gpt-4o",
                input=[
                    {
                        "role": "system",
                        "content": system_prompt,
                    },
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "input_file",
                                "file_id": file.id,
                            }
                        ],
                    },
                ],
            )

            # Parse the response and convert to TalkingPoints
            response_text = response.output_text

            try:
                # Remove markdown fences if present
                if "```json" in response_text:
                    response_text = response_text.split("```json")[1].split("```")[0]
                elif "```" in response_text:
                    response_text = response_text.split("```")[1].split("```")[0]

                data = json.loads(response_text.strip())

                talking_points = TalkingPoints(
                    talking_points=data.get("talking_points", []),
                    blocker_points=data.get("blocker_points", []),
                )

                return talking_points

            except Exception as e:
                print(f"⚠️  Error parsing talking points response: {e}")
                # Return empty talking points if parsing fails
                return TalkingPoints(talking_points=[], blocker_points=[])

        except Exception as e:
            print(f"⚠️  Error generating talking points for role: {e}")
            return TalkingPoints(talking_points=[], blocker_points=[])

    def update_talking_points_agent(
        self, state: ExpertSearchState, pdf_path: Path
    ) -> ExpertSearchState:
        """Update talking points for all roles in the state.

        Args:
            state: ExpertSearchState with roles_needed populated
            pdf_path: Path to the PDF file to analyze

        Returns:
            Updated ExpertSearchState with talking_points populated for each role
        """
        try:
            roles = state.get("roles_needed", [])

            if not roles:
                print("⚠️  No roles found in state for talking points update")
                return state

            print(f"🗺️  Starting talking points generation for {len(roles)} roles...")

            updated_roles = []
            for i, role in enumerate(roles, 1):
                role_name = role.get("broader_area", f"Role {i}")
                print(f"📋 Generating talking points for: {role_name}")

                # Generate talking points for this role
                talking_points = self.generate_talking_points_for_role(role, pdf_path)

                # Create updated role with talking points
                updated_role = RoleNeeded(
                    broader_area=role.get("broader_area", ""),
                    role_description=role.get("role_description", ""),
                    search_terms=role.get("search_terms", []),
                    why_needed=role.get("why_needed", ""),
                    experts_list=role.get("experts_list", {}),
                    talking_points=talking_points,
                )

                updated_roles.append(updated_role)
                print(f"✅ Completed talking points for: {role_name}")

            print("✅ Talking points generation completed!")

            # Return updated ExpertSearchState
            return ExpertSearchState(roles_needed=updated_roles)

        except Exception as e:
            print(f"⚠️  Error updating talking points: {e}")
            return state


if __name__ == "__main__":
    # Example usage
    talking_points_node = UpdateTalkingPointsNode(model=OpenAI())

    # Example: Update talking points from ExpertSearchState and PDF
    pdf_path = "/Users/qtf4195/Github_Projects/No-Expert-Agent/backend/test/Motorrad_Accessory_Bundles_1-Pager_reference.pdf"

    # Create example ExpertSearchState with some roles
    example_state = ExpertSearchState(
        roles_needed=[
            RoleNeeded(
                broader_area="IT Systems and Integration",
                role_description="IT systems expert for service platform integration",
                search_terms=["IT systems", "integration", "service platform"],
                why_needed="Need expertise in integrating AI platform with existing BMW service systems",
                experts_list=ExpertsList(enriched_scored_experts=[]),
                talking_points={"talking_points": [], "blocker_points": []},
            ),
            RoleNeeded(
                broader_area="Data Privacy and Compliance",
                role_description="Data privacy expert for GDPR compliance",
                search_terms=["data privacy", "GDPR", "compliance"],
                why_needed="Need expertise in data protection for multilingual service platform",
                experts_list=ExpertsList(enriched_scored_experts=[]),
                talking_points={"talking_points": [], "blocker_points": []},
            ),
        ]
    )

    updated_state = talking_points_node.update_talking_points_agent(
        example_state, pdf_path
    )
    print("Updated state with talking points:")
    print(updated_state)
