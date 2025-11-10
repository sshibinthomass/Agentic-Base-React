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

from langgraph_agent.tools.return_prompt import return_prompt
from langgraph_agent.states.state import (
    ExpertSearchState,
    RoleNeeded,
    ExpertsList,
)


class ExtractExpertiseAreasNode:
    def __init__(self, model):
        self.model = model
        self._load_prompts()

    def _load_prompts(self):
        """Load system prompts for different tools"""
        self.system_prompt_expertise_extractor = return_prompt(
            "expertise_extractor_tool"
        )

    def extract_expertise_areas_agent(self, pdf_path) -> ExpertSearchState:
        """
        Extract expertise areas from a PDF document and return as ExpertSearchState.

        Args:
            pdf_path: Path to the PDF file to analyze

        Returns:
            ExpertSearchState with roles_needed populated from the PDF analysis
        """
        try:
            client = self.model

            file = client.files.create(file=open(pdf_path, "rb"), purpose="user_data")

            response = client.responses.create(
                model="gpt-4.1-2025-04-14",
                input=[
                    {
                        "role": "system",
                        "content": self.system_prompt_expertise_extractor,
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

            # Parse the response and convert to ExpertSearchState
            response_text = response.output_text

            try:
                # Remove markdown fences if present
                if "```json" in response_text:
                    response_text = response_text.split("```json")[1].split("```")[0]
                elif "```" in response_text:
                    response_text = response_text.split("```")[1].split("```")[0]

                data = json.loads(response_text.strip())
                roles_data = data.get("roles_needed", [])

                # Convert to ExpertSearchState format with proper ordering
                roles_needed = []

                # Sort roles by order if provided, otherwise maintain original sequence
                if all("order" in role for role in roles_data):
                    roles_data = sorted(roles_data, key=lambda x: x.get("order", 999))

                for index, role_data in enumerate(roles_data, 1):
                    # Use provided order or fallback to sequential numbering
                    order_value = role_data.get("order", index)

                    role = RoleNeeded(
                        order=order_value,
                        broader_area=role_data.get("broader_area", ""),
                        role_description=role_data.get("role_description", ""),
                        search_terms=role_data.get("search_terms", []),
                        why_needed=role_data.get("why_needed", ""),
                        experts_list=ExpertsList(enriched_scored_experts=[]),
                    )
                    roles_needed.append(role)

                return ExpertSearchState(roles_needed=roles_needed)

            except Exception as e:
                print(f"⚠️  Error parsing expertise extraction response: {e}")
                # Return empty state if parsing fails
                return ExpertSearchState(roles_needed=[])

        except Exception as e:
            print(f"⚠️  Error processing PDF: {e}")
            return ExpertSearchState(roles_needed=[])


if __name__ == "__main__":
    # Example usage
    extract_node = ExtractExpertiseAreasNode(model=OpenAI())

    # Example 2: Extract from PDF (if PDF file exists)
    pdf_path = "/Users/qtf4195/Github_Projects/No-Expert-Agent/backend/test/Motorrad_Accessory_Bundles_1-Pager_reference.pdf"
    result = extract_node.extract_expertise_areas_agent(pdf_path)
    print("Extracted expertise areas from PDF:")
    print(result)
