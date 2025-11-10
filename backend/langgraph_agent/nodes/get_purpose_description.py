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
        RoleNeeded,
        ExpertsList,
        EnrichedScoredExpert,
    )
except ImportError:
    # Fallback import if path setup fails
    import sys

    sys.path.append(str(Path(__file__).parent.parent.parent.parent))
    from langgraph_agent.states.state import (
        ExpertSearchState,
        RoleNeeded,
        ExpertsList,
        EnrichedScoredExpert,
    )


class GetPurposeDescriptionNode:
    def __init__(self, model):
        self.model = model

    def generate_purpose_description_for_expert(
        self, expert: EnrichedScoredExpert, pdf_path: Path
    ) -> str:
        """Generate a purpose description for a specific expert based on the PDF.

        Args:
            expert: EnrichedScoredExpert object
            pdf_path: Path to the PDF file to analyze

        Returns:
            str: Generated purpose description
        """
        try:
            client = self.model

            file = client.files.create(file=open(pdf_path, "rb"), purpose="user_data")

            system_prompt = f"""
                CRITICAL: Read the attached project proposal document THOROUGLY to understand:
                - Project goals and timeline
                - Technical dependencies and constraints
                - Implementation phases
                - Risk areas and blockers
                
                EXPERT INFORMATION:
                Name: {expert.get("first_name", "")} {expert.get("last_name", "")}
                Function: {expert.get("function", "")}
                Key Alignment: {expert.get("key_alignment", "")}
                Reasoning: {expert.get("reasoning", "")}

                YOUR TASK:
                Write a brief purpose description (50-100 words) explaining how this specific expert 
                would contribute to the project described in the PDF. Focus on their specific role 
                and value-add to the project.

                REQUIREMENTS:
                - Reference the expert's specific function and alignment
                - Connect their expertise to the project needs
                - Be concise but informative
                - Write in third person
                - Focus on their unique contribution to the project
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

            # Parse the response and return the purpose description
            response_text = response.output_text

            return response_text

        except Exception as e:
            print(f"⚠️  Error generating purpose description for expert: {e}")
            return ""

    def update_purpose_descriptions_agent(
        self, state: ExpertSearchState, pdf_path: Path
    ) -> ExpertSearchState:
        """Update purpose descriptions for all experts in the state.

        Args:
            state: ExpertSearchState with roles_needed populated
            pdf_path: Path to the PDF file to analyze

        Returns:
            Updated ExpertSearchState with purpose_description populated for each expert
        """
        try:
            roles = state.get("roles_needed", [])

            if not roles:
                print("⚠️  No roles found in state for purpose description update")
                return state

            print("📝 Starting purpose description generation for all experts...")

            updated_roles = []
            for i, role in enumerate(roles, 1):
                role_name = role.get("broader_area", f"Role {i}")
                print(f"📋 Processing experts for: {role_name}")

                experts_list = role.get("experts_list", {})
                experts = experts_list.get("enriched_scored_experts", [])

                if not experts:
                    updated_roles.append(role)
                    continue

                # Update purpose descriptions for all experts in this role
                updated_experts = []
                for expert in experts:
                    expert_name = f"{expert.get('first_name', '')} {expert.get('last_name', '')}".strip()
                    print(f"   📝 Generating purpose description for: {expert_name}")

                    # Generate purpose description for this expert
                    purpose_description = self.generate_purpose_description_for_expert(
                        expert, pdf_path
                    )

                    # Create updated expert with purpose description
                    updated_expert = EnrichedScoredExpert(
                        first_name=expert.get("first_name", ""),
                        last_name=expert.get("last_name", ""),
                        function=expert.get("function", ""),
                        total_score=expert.get("total_score", 0),
                        reasoning=expert.get("reasoning", ""),
                        key_alignment=expert.get("key_alignment", ""),
                        source=expert.get("source", ""),
                        source_link=expert.get("source_link", ""),
                        linkedin=expert.get("linkedin", ""),
                        linkedin_image=expert.get("linkedin_image", ""),
                        email=expert.get("email", ""),
                        email_confidence_score=expert.get("email_confidence_score", 0),
                        purpose_description=purpose_description,
                    )

                    updated_experts.append(updated_expert)

                # Create updated ExpertsList
                updated_experts_list = ExpertsList(
                    enriched_scored_experts=updated_experts
                )

                # Create updated RoleNeeded
                updated_role = RoleNeeded(
                    broader_area=role.get("broader_area", ""),
                    role_description=role.get("role_description", ""),
                    search_terms=role.get("search_terms", []),
                    why_needed=role.get("why_needed", ""),
                    experts_list=updated_experts_list,
                    talking_points=role.get(
                        "talking_points", {"talking_points": [], "blocker_points": []}
                    ),
                )

                updated_roles.append(updated_role)
                print(f"✅ Completed purpose descriptions for: {role_name}")

            print("✅ Purpose description generation completed!")

            # Return updated ExpertSearchState
            return ExpertSearchState(
                roles_needed=updated_roles,
                roadmap=state.get("roadmap", ""),
            )

        except Exception as e:
            print(f"⚠️  Error updating purpose descriptions: {e}")
            return state


if __name__ == "__main__":
    # Example usage
    purpose_node = GetPurposeDescriptionNode(model=OpenAI())

    # Example: Update purpose descriptions from ExpertSearchState and PDF
    pdf_path = "/Users/qtf4195/Github_Projects/No-Expert-Agent/backend/test/Motorrad_Accessory_Bundles_1-Pager_reference.pdf"

    # Create example ExpertSearchState with some experts
    example_state = ExpertSearchState(
        roles_needed=[
            RoleNeeded(
                broader_area="IT Systems and Integration",
                role_description="IT systems expert for service platform integration",
                search_terms=["IT systems", "integration", "service platform"],
                why_needed="Need expertise in integrating AI platform with existing BMW service systems",
                experts_list=ExpertsList(
                    enriched_scored_experts=[
                        EnrichedScoredExpert(
                            first_name="John",
                            last_name="Smith",
                            function="IT Systems Architect",
                            total_score=85,
                            reasoning="Strong background in system integration",
                            key_alignment="Expertise in AI platform integration",
                            source="LinkedIn",
                            source_link="https://linkedin.com/in/johnsmith",
                            linkedin="https://linkedin.com/in/johnsmith",
                            linkedin_image="https://linkedin.com/in/johnsmith",
                            email="john.smith@bmw.de",
                            email_confidence_score=90,
                            purpose_description="",
                        )
                    ]
                ),
                talking_points={"talking_points": [], "blocker_points": []},
            ),
        ],
        roadmap="",
    )

    updated_state = purpose_node.update_purpose_descriptions_agent(
        example_state, pdf_path
    )
    print("Updated state with purpose descriptions:")
    print(updated_state)
