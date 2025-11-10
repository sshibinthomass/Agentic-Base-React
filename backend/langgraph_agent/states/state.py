from typing_extensions import TypedDict, List
from langgraph.graph.message import add_messages
from typing import Annotated


class State(TypedDict):
    """
    Represent the structure of the state used in graph,
    add_messages is a function that adds messages to the state for history of the conversation
    """

    messages: Annotated[List, add_messages]


class EnrichedScoredExpert(TypedDict):
    """
    Individual expert with enriched scoring data
    """

    first_name: str
    last_name: str
    function: str
    total_score: int
    reasoning: str
    key_alignment: str
    source: str
    source_link: str
    linkedin: str
    linkedin_image: str
    email: str
    email_confidence_score: int
    purpose_description: str


class ExpertsList(TypedDict):
    """
    Container for enriched scored experts
    """

    enriched_scored_experts: List[EnrichedScoredExpert]


class TalkingPoints(TypedDict):
    talking_points: List[str]
    blocker_points: List[str]


class RoleNeeded(TypedDict):
    """
    Individual role with its requirements and expert list
    """

    broader_area: str
    role_description: str
    search_terms: List[str]
    why_needed: str
    experts_list: ExpertsList
    talking_points: TalkingPoints
    order: int


class ExpertSearchState(TypedDict):
    """
    Complete state for expert search functionality with all roles and expert data
    """

    roles_needed: List[RoleNeeded]
    roadmap: str
