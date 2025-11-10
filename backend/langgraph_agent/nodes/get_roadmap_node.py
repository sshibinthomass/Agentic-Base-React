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
        RoleNeeded,
        ExpertsList,
    )
except ImportError:
    # Fallback import if path setup fails
    import sys

    sys.path.append(str(Path(__file__).parent.parent.parent.parent))
    from langgraph_agent.states.state import (
        ExpertSearchState,
        RoleNeeded,
        ExpertsList,
    )


class GetRoadmapNode:
    def __init__(self, model):
        self.model = model

    def generate_roadmap_agent(
        self, state: ExpertSearchState, pdf_path: Path
    ) -> ExpertSearchState:
        """Generate a concise roadmap for the research result and update the state.

        Args:
            state: ExpertSearchState with roles_needed populated
            pdf_path: Path to the PDF file to analyze

        Returns:
            Updated ExpertSearchState with roadmap populated
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
                
                Use this understanding to create a LOGICAL engagement sequence.

                IDENTIFIED EXPERTISE AREAS:
                {json.dumps(state, indent=2)}

                YOUR TASK:
                Write a brief roadmap (150-250 words) in FLOWING PROSE—no bullet points, no numbered lists.
                Write it as 2-3 cohesive paragraphs that someone can quickly read to understand their next steps.

                ## STRUCTURE (but write as flowing text):

                **Paragraph 1 - Overview & First Contacts:**
                Start with how many areas were identified. Then explain who to contact first and second, 
                referencing SPECIFIC project details (timelines, systems mentioned, risks flagged) to 
                justify the sequence.

                **Paragraph 2 - Middle & Later Contacts:**
                Continue the sequence, explaining the logical flow of remaining expert engagements based 
                on dependencies and project phases mentioned in the document.

                **Paragraph 3 - Approach (optional, can merge with paragraph 2):**
                Briefly describe how to approach these conversations.

                ## EXAMPLE OUTPUT (FLOWING TEXT FORMAT):

                "We identified six expertise areas for the SprachAssist Service Hub project. Begin with 
                IT Systems and Integration, as the document outlines a Q2 pilot requiring integration 
                with ISTA/AIR, DMS, and Retail Next systems—understanding technical feasibility and 
                timeline constraints upfront is critical before committing resources. Immediately after, 
                engage Data Privacy and Compliance experts, since page 4 explicitly flags workers' 
                council concerns and GDPR requirements that could become legal blockers during development.

                Once technical and legal boundaries are clear, bring in Service Operations specialists 
                to define the hands-free voice requirements and workflow integration needs for service 
                technicians and advisors mentioned in section 2. After operational requirements are 
                scoped, work with UX and Training experts to design the adaptive learning and onboarding 
                approach referenced on page 3. Business Development should validate the €120-180k budget 
                estimate and ROI assumptions once the operational scope is defined, followed finally by 
                Marketing and Communications to plan launch activities after the pilot evaluation phase.

                Treat these early conversations as risk mitigation rather than solution pitching—your 
                goal is to validate the Q2 timeline and uncover any legal or technical constraints before 
                designing solutions."

                REQUIREMENTS:
                - Write in flowing paragraphs, NOT lists or bullet points
                - Reference SPECIFIC document details (timelines, page numbers, systems, budget figures)
                - Show clear cause-and-effect logic in the sequence
                - Use natural transitions between experts ("Once...", "After...", "Following this...")
                - Make it feel project-driven, not generic
                - Keep it concise and scannable despite being prose
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

            # Parse the response and return the roadmap
            response_text = response.output_text

            # Update the state with the roadmap
            updated_state = ExpertSearchState(
                roles_needed=state.get("roles_needed", []),
                roadmap=response_text,
            )

            return updated_state

        except Exception as e:
            print(f"⚠️  Error generating roadmap: {e}")
            # Return original state if error occurs
            return state


if __name__ == "__main__":
    # Example usage
    roadmap_node = GetRoadmapNode(model=OpenAI())

    # Example: Generate roadmap from ExpertSearchState and PDF
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
        ],
        roadmap="",
    )

    updated_state = roadmap_node.generate_roadmap_agent(
        example_state, pdf_path
    )
    print("Updated state with roadmap:")
    print(updated_state)

