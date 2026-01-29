from typing import TypedDict, Optional, Annotated, List
from langchain_core.messages import BaseMessage, AIMessage
from langgraph.graph import StateGraph, END
from langgraph.graph.message import add_messages
from langgraph.errors import NodeInterrupt

class SupportState(TypedDict):
    messages: Annotated[List[BaseMessage], add_messages]
    need_human: bool
    needs_tool: bool
    question_result: Optional[str]

def triage(state: SupportState):
    last = state["messages"][-1].content.lower()
    need_human = any(k in last for k in ["refund", "angry", "complaint", "chargeback"])
    needs_tool = any(k in last for k in ["order", "status", "tracking", "shipment"])
    return {"need_human": need_human, "needs_tool": needs_tool}

def route_after_triage(state: SupportState):
    if state["need_human"]:
        return "handoff"
    if state["needs_tool"]:
        return "question_lookup"
    return "answer"

def question_lookup(state: SupportState):
    return {"question_result": "Homework question is completed! Good job!"}

def answer(state: SupportState):
    if state.get("question_result"):
        text = f"Update: {state['question_result']}"
    else:
        text = "Got it. Homework question is now answered!"
    return {"messages": [AIMessage(content=text)]}

def handoff(state: SupportState):
    raise NodeInterrupt("Escalate to a human agent: refund/complaint requires approval.")

def build_graph():
    b = StateGraph(SupportState)
    b.add_node("triage", triage)
    b.add_node("question_lookup", question_lookup)
    b.add_node("answer", answer)
    b.add_node("handoff", handoff)

    b.set_entry_point("triage")
    b.add_conditional_edges("triage", route_after_triage)
    b.add_edge("question_lookup", "answer")
    b.add_edge("answer", END)

    return b.compile()

# Required for `langgraph dev`
graph = build_graph()