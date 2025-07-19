from langgraph.graph import StateGraph, START, END
from langchain_openai import ChatOpenAI
from langchain.schema.messages import AIMessage, HumanMessage, SystemMessage, AnyMessage
from langchain.prompts import ChatPromptTemplate
from dotenv import load_dotenv
from typing import TypedDict, Literal
from typing_extensions import Dict, List
from datetime import datetime

load_dotenv(override=True)
llm = ChatOpenAI(model_name="gpt-4o-mini")

class State(TypedDict):
    messages: List[AnyMessage]
    research_data:str = ""
    analysis:str = ""
    final_report:str = ""
    current_task:str = ""
    next_agent:str = ""
    complete_task:bool = False,
    next_node: str = ""


def classify_user_intent(state:State)->Dict:
    user_input = state["messages"][-1].content if state["messages"] else ""

    classifier_prompt = ChatPromptTemplate.from_messages([
        SystemMessage(
            content="""
            You are a classifier that determines if a user's request requires a multi-agent research system.

            Return "MULTI_AGENT" if the request involves:
            - Research on a specific topic
            - Analysis of data, trends, or phenomena
            - Creating reports or comprehensive summaries
            - Gathering information about companies, markets, technologies, etc.
            - Comparative studies or investigations
            - Any task requiring structured research → analysis → reporting workflow

            Return "NORMAL_CHAT" if the request is:
            - General questions or conversations
            - Simple explanations or definitions
            - Coding help or technical questions
            - Personal advice or opinions
            - Mathematical calculations
            - Creative writing requests
            - Casual conversation

            Respond with ONLY "MULTI_AGENT" or "NORMAL_CHAT".
            """
        ),
        HumanMessage(content=f"Classify this request: {user_input}")
    ])

    response = llm.invoke(classifier_prompt.format_messages())
    decision = response.content.strip().upper()

    if "MULTI_AGENT" in decision:
        next_node = "supervisor"
        classifier_msg = "🔄 Intent Classifier: Detected research/analysis request. Routing to multi-agent system..."
    else:
        next_node = "normal_chatbot"
        classifier_msg = "💬 Intent Classifier: Routing to normal chat..."

    return {
        "messages": [AIMessage(content=classifier_msg)],
        "next_agent": next_node,
        "current_task": user_input
    }


def handle_normal_chat(state:State)->Dict:
    user_input = state.get("current_task", "") or state["messages"][-1].content if state["messages"] else ""

    normal_chat_prompt = ChatPromptTemplate.from_messages([
        SystemMessage(
            content="""
            You are a helpful AI assistant. Provide clear, accurate, and helpful responses to user queries.
            Be conversational and engaging while maintaining professionalism.
            If you do not know the answer or do not have the real-time information, answer accordingly. 
            """
        ),
        HumanMessage(content=user_input)
    ])

    response = llm.invoke(normal_chat_prompt.format_messages())
    return {
        "messages": [AIMessage(content=response.content)],
        "task_complete": True,
        "next_agent": "end"
    }


def create_supervisor_chain():
    supervisor_chain = ChatPromptTemplate.from_messages([
        SystemMessage(
            content="""
    You are a supervisor managing a team of agents:

    1. Researcher - Gathers information and data
    2. Analyst - Analyzes data and provides insights
    3. Writer - Creates reports and summaries

    Based on the current state and conversation, decide which agent should work next.
    If the task is complete, respond with 'DONE'.
    If the user asks for a task that has no concern with any of the agents in the team, then respond with
    'There is no agent dedicated for this task, do you anything else to work on?'

    Current state:
    - Has research data: {has_research}
    - Has analysis: {has_analysis}
    - Has report: {has_report}

    Respond with ONLY the agent name (researcher/analyst/writer) or 'DONE'.

    """
    ),
    (HumanMessage(content="{task}"))

])
    return supervisor_chain | llm


def supervisor_agent(state:State)->Dict:
    messages = state["messages"][-1].content
    task = messages if messages else "No Task"
    has_research = bool(state.get("research_data", ""))
    has_analysis = bool(state.get("analysis", ""))
    has_report = bool(state.get("final_report", ""))

    chain = create_supervisor_chain()
    decision = chain.invoke({
        "has_research" : has_research,
        "has_analysis" : has_analysis,
        "has_report" : has_report,
        "task" : task
    })
    decision_text = decision.content.strip().lower()
    if "done" in decision_text or has_report:
        supervisor_msg = "✅ Supervisor: All tasks complete! Great work team."
        next_agent = "end"
    elif "researcher" in decision_text or not has_research:
        supervisor_msg = "📋 Supervisor: Let's start with research. Assigning to Researcher..."
        next_agent = "researcher"
    elif "analyst" in decision_text or (has_research and not has_analysis):
        supervisor_msg = "📋 Supervisor: Research done. Time for analysis. Assigning to Analyst..."
        next_agent = "analyst"
    elif "writer" in decision_text or (has_analysis and not has_report):
        supervisor_msg = "📋 Supervisor: Analysis complete. Let's create the report. Assigning to Writer..."
        next_agent = "writer"
    else:
        supervisor_msg = "✅ Supervisor: Task seems complete."
        next_agent = "end"

    return {
        "messages": [AIMessage(content=supervisor_msg)],
        "next_agent": next_agent,
        "current_task": task
    }


def researcher_agent(state:State)->Dict:
    task = state.get("current_task", "research topic")

    research_system_prompt = SystemMessage(content=
    """
    You are a research agent that understands the user's task and provides the research based on the mentioned topic.
    Be concise but thorough.
    """
    )
    research_user_prompt = HumanMessage(content=
    f"""
    As a research specialist, provide comprehensive information about: {task}

    Include:
    1. Key facts and background
    2. Current trends or developments
    3. Important statistics or data points
    4. Notable examples or case studies

    Be concise but thorough."
    """
    )
    research_response = llm.invoke([research_system_prompt, research_user_prompt])
    research_data = research_response.content
    agent_msg = f"🔍 Researcher: I've completed the research on '{task}'.\n\nKey findings:\n{research_data[:500]}..."

    return {
        "messages": [AIMessage(content=agent_msg)],
        "next_agent": "supervisor",
        "research_data": research_data
    }

def analysis_agent(state:State)->Dict:
    task = state.get("current_task", "")
    research_data = state.get("research_data", "")

    analysis_system_prompt = SystemMessage(
        content="""
        You are an analysis agent that understands the research data and provides the relevant analysis based on the
        mentioned topic. Be concise but thorough.
        """
    )
    analysis_human_prompt = HumanMessage(
        content=f"""
        As a data analyst, analyze this research data and provide insights:

        Research Data:
        {research_data}

        Provide:
        1. Key insights and patterns
        2. Strategic implications
        3. Risks and opportunities
        4. Recommendations

        Focus on actionable insights related to: {task}
        """
    )
    analysis_response = llm.invoke([analysis_system_prompt, analysis_human_prompt])
    analysis = analysis_response.content
    agent_msg = f"📊 Analyst: I've completed the analysis.\n\nTop insights:\n{analysis[:400]}..."

    return {
        "messages": [AIMessage(content=agent_msg)],
        "next_agent": "supervisor",
        "analysis": analysis
    }

def writer_agent(state:State)->Dict:
    task = state.get("current_task", "")
    research_data = state.get("research_data", "")
    analysis = state.get("analysis", "")


    writer_system_prompt = SystemMessage(
        content="""
        You are a writer agent that understands the research data, analyzes the insights, and then writes a
        final report based on the mentioned topic. Be concise but thorough.
        """
    )
    writer_human_prompt = HumanMessage(
        content=f"""
        As a professional writer, create an executive report based on:

        Task: {task}

        Research Findings:
        {research_data[:1000]}

        Analysis:
        {analysis[:1000]}

        Create a well-structured report with:
        1. Executive Summary
        2. Key Findings
        3. Analysis & Insights
        4. Recommendations
        5. Conclusion

        Keep it professional and concise
        """
    )

    writer_response = llm.invoke([writer_system_prompt, writer_human_prompt])
    report = writer_response.content

    final_report = f"""
    📄 FINAL REPORT
    {'='*50}
    Generated: {datetime.now().strftime('%Y-%m-%d %H:%M')}
    Topic: {task}
    {'='*50}

    {report}

    {'='*50}
    Report compiled by Multi-Agent AI System powered by ChatGPT"""

    return {
        "messages": [AIMessage(content=f"✍️ Writer: Report complete! See below for the full document.")],
        "next_agent": "supervisor",
        "final_report": final_report,
        "complete_task": True
    }

def agent_router(state:State)-> Literal["supervisor", "researcher", "analyst", "writer", "end"]:
    next_agent = state.get("next_agent", "")

    if "end" in next_agent or state.get("complete_task"):
        return END

    if next_agent in ["researcher", "analyst", "writer"]:
        return next_agent

    return "supervisor"

def intent_router(state:State)-> Literal["supervisor", "normal chatbot", "end"]:
    next_agent = state.get("next_agent", "supervisor")

    if "end" in next_agent or state.get("complete_task", False):
        return END
    elif next_agent == "normal chatbot":
        return "normal chatbot"
    else:
        return "supervisor"

graph = StateGraph(State)

graph.add_node("intent classifier", classify_user_intent)
graph.add_node("normal chatbot", handle_normal_chat)
graph.add_node("supervisor", supervisor_agent)
graph.add_node("researcher", researcher_agent)
graph.add_node("analyst", analysis_agent)
graph.add_node("writer", writer_agent)

graph.add_edge(START, "intent classifier")
graph.add_conditional_edges(
    "intent classifier",
    intent_router,
    {
    "normal chatbot": "normal chatbot",
    "supervisor": "supervisor",
    END:END
    })

for node in ["supervisor", "researcher", "analyst", "writer"]:
    graph.add_conditional_edges(
    node,
    agent_router,
    {
    "supervisor": "supervisor",
    "researcher": "researcher",
    "analyst": "analyst",
    "writer": "writer",
    END: END
    })

builder = graph.compile()

def run_chatbot():
    print("🤖 AI Assistant: Hello! I can help with both general questions and comprehensive research tasks.")
    print("💡 For research/analysis tasks, I'll use my specialized team of agents.")
    print("💬 For general questions, I'll respond directly. Type 'quit' to exit.\n")


    while True:
        user_input = input("User: ").strip()
        if not user_input:
            continue

        if user_input.lower() in ["quit", "exit"]:
            print("Good Bye!")
            break

        # Classify user intent
        classify_intent = {"messages": [HumanMessage(content=user_input)]}
        current_intent = classify_user_intent(classify_intent)
        requires_multi_agent = current_intent.get("next_agent")

        if requires_multi_agent == "supervisor":
            print("🔄 Detected research/analysis/report request. Activating multi-agent system...")

            # Reset multi-agent state for new task
            multi_agent_state = {
                "research_data": "",
                "analysis": "",
                "final_report": "",
                "current_task": "",
                "next_agent": "",
                "complete_task": False,
                "messages": [HumanMessage(content=user_input)]
            }

            # Run multi-agent workflow
            result_state = builder.invoke(multi_agent_state)

            # Display all agent messages
            for message in result_state["messages"]:
                if isinstance(message, AIMessage):
                    print(f"Assistant: {message.content}")

            # Display final report if available
            if result_state.get("final_report"):
                print(f"\n{result_state['final_report']}")

        else:
            # Handle as normal chat
            normal_intent = {"messages": [HumanMessage(content=user_input)]}
            response = handle_normal_chat(normal_intent)
            chatbot_response = response.get("messages")[-1].content
            print(f"Assistant: {chatbot_response}")


def draw_graph():
    with open("supervisor_langgraph_diagram.png", "wb") as f:
        f.write(builder.get_graph().draw_mermaid_png())


if __name__ == "__main__":
    run_chatbot()
    draw_graph()

