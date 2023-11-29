from langchain.agents import AgentExecutor, Tool, initialize_agent, AgentType
from langchain.agents.agent_toolkits import GmailToolkit
from langchain.tools.gmail.utils import build_resource_service, get_gmail_credentials


def get_gmail_agent(llm, is_verbose=False, max_iterations=3) -> AgentExecutor:
    # Can review scopes here https://developers.google.com/gmail/api/auth/scopes
    # For instance, readonly scope is 'https://www.googleapis.com/auth/gmail.readonly'
    credentials = get_gmail_credentials(
        token_file="token.json",
        scopes=['https://www.googleapis.com/auth/gmail.readonly',
                'https://www.googleapis.com/auth/gmail.compose'],
        client_secrets_file="credentials.json",
    )
    api_resource = build_resource_service(credentials=credentials)
    gmail_toolkit = GmailToolkit(api_resource=api_resource)

    gmail_agent = initialize_agent(
        agent=AgentType.STRUCTURED_CHAT_ZERO_SHOT_REACT_DESCRIPTION,
        tools=gmail_toolkit.get_tools(),
        llm=llm,
        verbose=is_verbose,
        max_iterations=max_iterations
    )

    return gmail_agent


def get_gmail_tool(llm, is_verbose=False, max_iterations=3) -> Tool:
    gmail_agent = get_gmail_agent(
        llm=llm,
        is_verbose=is_verbose,
        max_iterations=max_iterations
    )

    gmail_tool = Tool(
        name='Gmail agent',
        func=gmail_agent.run,
        description='Useful for when you need access to gmail'
    )

    return gmail_tool
