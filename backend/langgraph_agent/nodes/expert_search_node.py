import os
from openai import OpenAI
import json
import concurrent.futures
from typing import List
import sys
from pathlib import Path
from dotenv import load_dotenv

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
    EnrichedScoredExpert,
    ExpertsList,
)

os.environ["EXA_API_KEY"] = os.getenv("EXA_API_KEY")
os.environ["PERPLEXITY_API_KEY"] = os.getenv("PERPLEXITY_API_KEY")


class ExpertSearchNode:
    def __init__(self, model=None):
        self.exa_client = OpenAI(
            base_url="https://api.exa.ai",
            api_key=os.getenv("EXA_API_KEY"),
        )
        self.perplexity_client = OpenAI(
            api_key=os.getenv("PERPLEXITY_API_KEY"),
            base_url="https://api.perplexity.ai",
        )
        self.model = model  # For scoring experts
        self._load_prompts()

    def _load_prompts(self):
        """Load system prompts for different tools"""
        self.system_prompt_exa = return_prompt("exa_tool")
        self.system_prompt_perplexity = return_prompt("perplexity_tool")

    def _is_individual_person(self, name: str) -> bool:
        """
        Check if the name represents an individual person rather than a team/department

        Args:
            name: The name to check

        Returns:
            True if it appears to be an individual person, False otherwise
        """
        if not name or not name.strip():
            return False

        name_lower = name.lower().strip()

        # Exclude team/department names
        exclude_patterns = [
            "bmw team",
            "bmw group",
            "bmw operations",
            "bmw department",
            "bmw motorrad team",
            "bmw motorrad group",
            "bmw motorrad operations",
            "team",
            "group",
            "department",
            "operations",
            "specialists",
            "managers",
            "coordinators",
            "careers team",
            "recruitment team",
            "logistics team",
            "legal team",
            "compliance team",
            "marketing team",
            "aftersales team",
            "sales team",
            "hr team",
            "it team",
        ]

        for pattern in exclude_patterns:
            if pattern in name_lower:
                return False

        # Must have at least one space (first and last name)
        if " " not in name.strip():
            return False

        # Must not start with "BMW" unless it's clearly a person's name
        if name_lower.startswith("bmw ") and not any(char.isdigit() for char in name):
            return False

        return True

    def search_exa_tool(self, query) -> List[EnrichedScoredExpert]:
        """
        Search for experts using Exa API and return as EnrichedScoredExpert objects

        Args:
            query: Search query describing the expertise needed

        Returns:
            List of EnrichedScoredExpert objects
        """
        completion = self.exa_client.chat.completions.create(
            model="exa",
            messages=[
                {"role": "system", "content": self.system_prompt_exa},
                {"role": "user", "content": query + " person bmw"},
            ],
            stream=False,
        )

        response_text = completion.choices[0].message.content

        # Parse the response and convert to EnrichedScoredExpert objects
        try:
            # Remove markdown fences if present
            if "```json" in response_text:
                response_text = response_text.split("```json")[1].split("```")[0]
            elif "```" in response_text:
                response_text = response_text.split("```")[1].split("```")[0]

            data = json.loads(response_text.strip())
            experts = data.get("experts", [])

            enriched_experts = []
            for expert in experts:
                # Handle both old format (name field) and new format (first_name/last_name fields)
                if "first_name" in expert and "last_name" in expert:
                    # New format with separate first_name and last_name
                    first_name = expert.get("first_name", "").strip()
                    last_name = expert.get("last_name", "").strip()
                    full_name = f"{first_name} {last_name}".strip()
                else:
                    # Old format with combined name field
                    name = expert.get("name", "").strip()
                    first_name = name.split()[0] if name.split() else name
                    last_name = (
                        " ".join(name.split()[1:]) if len(name.split()) > 1 else ""
                    )
                    full_name = name

                if full_name and self._is_individual_person(full_name):
                    enriched_expert = EnrichedScoredExpert(
                        first_name=first_name,
                        last_name=last_name,
                        function=expert.get("function", ""),
                        total_score=0,
                        reasoning="Found via Exa search",
                        key_alignment=expert.get("function", ""),
                        source="Exa",
                        source_link=expert.get("source_link", ""),
                        linkedin="",
                        linkedin_image="",
                        email="",
                        email_confidence_score=0,
                    )
                    enriched_experts.append(enriched_expert)
                elif full_name:
                    print(
                        f"🔍 Filtered out non-individual result from Exa: '{full_name}'"
                    )

            return enriched_experts

        except Exception as e:
            print(f"⚠️  Error parsing Exa response: {e}")
            return []

    def search_perplexity_tool(self, query) -> List[EnrichedScoredExpert]:
        """
        Search for experts using Perplexity API and return as EnrichedScoredExpert objects

        Args:
            query: Search query describing the expertise needed

        Returns:
            List of EnrichedScoredExpert objects
        """
        response = self.perplexity_client.chat.completions.create(
            model="sonar-pro",
            messages=[
                {"role": "system", "content": self.system_prompt_perplexity},
                {"role": "user", "content": query + "person bmw"},
            ],
        )

        response_text = response.choices[0].message.content

        # Parse the response and convert to EnrichedScoredExpert objects
        try:
            # Remove markdown fences if present
            if "```json" in response_text:
                response_text = response_text.split("```json")[1].split("```")[0]
            elif "```" in response_text:
                response_text = response_text.split("```")[1].split("```")[0]

            data = json.loads(response_text.strip())
            experts = data.get("experts", [])

            enriched_experts = []
            for expert in experts:
                # Handle both old format (name field) and new format (first_name/last_name fields)
                if "first_name" in expert and "last_name" in expert:
                    # New format with separate first_name and last_name
                    first_name = expert.get("first_name", "").strip()
                    last_name = expert.get("last_name", "").strip()
                    full_name = f"{first_name} {last_name}".strip()
                else:
                    # Old format with combined name field
                    name = expert.get("name", "").strip()
                    first_name = name.split()[0] if name.split() else name
                    last_name = (
                        " ".join(name.split()[1:]) if len(name.split()) > 1 else ""
                    )
                    full_name = name

                if full_name and self._is_individual_person(full_name):
                    enriched_expert = EnrichedScoredExpert(
                        first_name=first_name,
                        last_name=last_name,
                        function=expert.get("function", ""),
                        total_score=0,
                        reasoning="Found via Perplexity search",
                        key_alignment=expert.get("function", ""),
                        source="Perplexity",
                        source_link=expert.get("source_link", ""),
                        linkedin="",
                        linkedin_image="",
                        email="",
                        email_confidence_score=0,
                    )
                    enriched_experts.append(enriched_expert)
                elif full_name:
                    print(
                        f"🔍 Filtered out non-individual result from Perplexity: '{full_name}'"
                    )

            return enriched_experts

        except Exception as e:
            print(f"⚠️  Error parsing Perplexity response: {e}")
            return []

    def _combine_and_deduplicate_experts(
        self,
        exa_experts: List[EnrichedScoredExpert],
        perplexity_experts: List[EnrichedScoredExpert],
    ) -> List[EnrichedScoredExpert]:
        """
        Combine experts from both sources and remove duplicates

        Args:
            exa_experts: List of EnrichedScoredExpert objects from Exa
            perplexity_experts: List of EnrichedScoredExpert objects from Perplexity

        Returns:
            Combined list of unique experts as EnrichedScoredExpert objects
        """
        all_experts = []
        seen_names = set()

        # Add Exa experts
        for expert in exa_experts:
            full_name = f"{expert['first_name']} {expert['last_name']}".strip()
            if full_name and full_name not in seen_names:
                all_experts.append(expert)
                seen_names.add(full_name)

        # Add Perplexity experts (skip duplicates)
        for expert in perplexity_experts:
            full_name = f"{expert['first_name']} {expert['last_name']}".strip()
            if full_name and full_name not in seen_names:
                all_experts.append(expert)
                seen_names.add(full_name)

        return all_experts

    def _score_experts_for_role(
        self, role: RoleNeeded, experts: List[EnrichedScoredExpert]
    ) -> List[EnrichedScoredExpert]:
        """
        Score experts for a specific role using AI evaluation.

        Args:
            role: RoleNeeded containing the context for scoring
            experts: List of EnrichedScoredExpert objects to score

        Returns:
            List of EnrichedScoredExpert objects with updated scores
        """
        if not self.model:
            print("⚠️  No model provided for scoring, returning unscored experts")
            return experts

        # Convert experts to dict format for the prompt
        experts_dict = []
        for expert in experts:
            experts_dict.append(
                {
                    "name": f"{expert['first_name']} {expert['last_name']}".strip(),
                    "function": expert["function"],
                }
            )

        scoring_prompt = f"""
        Score experts for this role: {role.get("broader_area", "")} - {role.get("role_description", "")}

        Candidates:
        {json.dumps(experts_dict, indent=1)}

        Return JSON with scores (0-100):
        {{
          "scored_experts": [
            {{
              "name": "expert name",
              "total_score": 85,
              "reasoning": "Brief explanation",
              "key_alignment": "Key fit factor"
            }}
          ]
        }}
        Sort by score descending.
        """

        response = self.model.invoke(
            [
                {
                    "role": "system",
                    "content": "Score candidates quickly and accurately.",
                },
                {"role": "user", "content": scoring_prompt},
            ],
            config={
                "max_tokens": 1000,
                "temperature": 0.1,
            },  # Faster, more deterministic
        )

        # Parse the response content
        response_content = (
            response.content if hasattr(response, "content") else str(response)
        )

        # Parse JSON response and convert to EnrichedScoredExpert objects
        try:
            # Remove markdown fences if present
            if "```json" in response_content:
                response_content = response_content.split("```json")[1].split("```")[0]
            elif "```" in response_content:
                response_content = response_content.split("```")[1].split("```")[0]

            result = json.loads(response_content.strip())
            scored_experts_data = result.get("scored_experts", [])

            # Convert scored experts back to EnrichedScoredExpert format
            scored_experts = []
            for scored_expert in scored_experts_data:
                # Find the original expert to preserve other fields
                original_expert = None
                for expert in experts:
                    full_name = f"{expert['first_name']} {expert['last_name']}".strip()
                    if full_name == scored_expert.get("name", ""):
                        original_expert = expert
                        break

                if original_expert:
                    # Create updated EnrichedScoredExpert with scores
                    updated_expert = EnrichedScoredExpert(
                        first_name=original_expert["first_name"],
                        last_name=original_expert["last_name"],
                        function=original_expert["function"],
                        total_score=scored_expert.get("total_score", 0),
                        reasoning=scored_expert.get("reasoning", ""),
                        key_alignment=scored_expert.get("key_alignment", ""),
                        source=original_expert.get("source", ""),
                        source_link=original_expert.get("source_link", ""),
                        linkedin=original_expert.get("linkedin", ""),
                        linkedin_image=original_expert.get("linkedin_image", ""),
                        email=original_expert.get("email", ""),
                        email_confidence_score=original_expert.get(
                            "email_confidence_score", 0
                        ),
                    )
                    scored_experts.append(updated_expert)

            return scored_experts

        except Exception as e:
            print(f"⚠️  Error parsing scoring response: {e}")
            # Return original experts with default scores if parsing fails
            return experts

    def _search_experts_for_role(self, role: RoleNeeded) -> RoleNeeded:
        """
        Search for experts for a single role using both Exa and Perplexity in parallel.
        After search completion, score the experts for this role.

        Args:
            role: RoleNeeded object containing search terms and role details

        Returns:
            Updated RoleNeeded with expert results populated and scored
        """
        role_name = role.get("broader_area", "Unknown")
        print(f"   🔍 Processing {role_name}...")

        # Extract search terms from the role
        search_terms = role.get("search_terms", [])

        # Combine all search terms into a single query
        combined_query = " ".join(search_terms) if search_terms else "expert"

        # Execute both searches in parallel for this role
        with concurrent.futures.ThreadPoolExecutor(max_workers=2) as executor:
            # Submit both tasks
            exa_future = executor.submit(self.search_exa_tool, combined_query)
            perplexity_future = executor.submit(
                self.search_perplexity_tool, combined_query
            )

            # Wait for both to complete
            exa_experts = exa_future.result()
            perplexity_experts = perplexity_future.result()

        # Combine and deduplicate experts
        all_experts = self._combine_and_deduplicate_experts(
            exa_experts, perplexity_experts
        )

        # Limit experts to top 5 per role for faster scoring
        limited_experts = all_experts[:5] if len(all_experts) > 5 else all_experts

        # Score the experts for this role if model is available
        if self.model and limited_experts:
            scored_experts = self._score_experts_for_role(role, limited_experts)
        else:
            scored_experts = limited_experts

        # Create ExpertsList with found and scored experts
        experts_list = ExpertsList(enriched_scored_experts=scored_experts)

        # Update the role with the found and scored experts
        updated_role = RoleNeeded(
            broader_area=role.get("broader_area", ""),
            role_description=role.get("role_description", ""),
            search_terms=role.get("search_terms", []),
            why_needed=role.get("why_needed", ""),
            experts_list=experts_list,
        )

        return updated_role

    def expert_search_agent(self, state: ExpertSearchState) -> ExpertSearchState:
        """
        Agent that searches for experts using both Exa and Perplexity in parallel.
        After search completion, scores all experts for each role.
        Now processes all roles in parallel for maximum efficiency.

        Args:
            state: ExpertSearchState containing roles_needed with search terms

        Returns:
            Updated ExpertSearchState with expert results populated and scored
        """
        roles = state.get("roles_needed", [])

        if not roles:
            return ExpertSearchState(roles_needed=[])

        # Process all roles in parallel using ThreadPoolExecutor with limited workers
        max_workers = min(
            len(roles), 4
        )  # Limit concurrent workers to avoid API rate limits
        with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
            # Submit all role searches in parallel
            future_to_role = {
                executor.submit(self._search_experts_for_role, role): role
                for role in roles
            }

            # Collect results as they complete
            updated_roles = []
            completed_count = 0
            for future in concurrent.futures.as_completed(future_to_role):
                try:
                    updated_role = future.result()
                    updated_roles.append(updated_role)
                    completed_count += 1
                    print(f"   ✅ Completed {completed_count}/{len(roles)} roles")
                except Exception as e:
                    print(f"⚠️  Error processing role: {e}")
                    # Keep the original role if processing fails
                    original_role = future_to_role[future]
                    updated_roles.append(original_role)
                    completed_count += 1

        # Return updated ExpertSearchState
        return ExpertSearchState(roles_needed=updated_roles)


if __name__ == "__main__":
    # Example usage
    expert_search = ExpertSearchNode()

    # Create example ExpertSearchState
    example_state = ExpertSearchState(
        roles_needed=[
            RoleNeeded(
                broader_area="Data Science",
                role_description="Data scientist with machine learning expertise",
                search_terms=["data scientist", "machine learning", "python"],
                why_needed="Need expertise in data analysis and ML model development",
                experts_list=ExpertsList(enriched_scored_experts=[]),
            ),
            RoleNeeded(
                broader_area="Cloud Computing",
                role_description="Cloud engineer with AWS expertise",
                search_terms=["cloud engineer", "AWS", "infrastructure"],
                why_needed="Need expertise in cloud infrastructure setup and management",
                experts_list=ExpertsList(enriched_scored_experts=[]),
            ),
        ]
    )

    result = expert_search.expert_search_agent(example_state)
    print(result)
