from pydantic_settings import BaseSettings, SettingsConfigDict


class MCPConfiguration(BaseSettings):
    model_config = SettingsConfigDict(env_nested_delimiter="__")

    mcp_chatbot_node_config: list[str] = [
        "subtract",
        "multiply",
        "search",
        "fetch_content",
        "get_me",
        "search_repositories",
        "brave_news_search",
    ]
    wiki_agent_config: list[str] = [
        "extract_key_facts",
        "get_article",
        "get_coordinates",
        "get_links",
        "get_related_topicsget_sections",
        "get_summary",
        "search_wikipedia",
        "summarize_article_for_query",
        "summarize_article_section",
        "test_wikipedia_connectivity",
    ]


class AppConfiguration(BaseSettings):
    mcps: MCPConfiguration = MCPConfiguration()
