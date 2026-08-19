"""LangGraph multi-step reasoning agent with iterative self-correction."""

import os
from typing import Any, Dict, List
from langgraph.graph import StateGraph, START, END

from app.core.config import settings
from app.core.logging import logger
from app.services.agent.state import AgentState
from app.services.agent import tools


def investigate_node(state: AgentState) -> Dict[str, Any]:
    """Inspect workspace files and search for relevant keywords."""
    workspace = state["workspace_dir"]
    desc = state["task_description"]

    file_list = tools.list_files(workspace)
    files = file_list.get("files", [])

    keywords = [w for w in desc.split() if len(w) > 3]
    matches = []
    for kw in keywords[:3]:
        res = tools.search_code(workspace, kw)
        if res.get("matches"):
            matches.extend(res["matches"])

    findings = f"Found {len(files)} files in workspace. Discovered {len(matches)} initial code matches."
    return {
        "status": "investigating",
        "investigation_findings": findings,
        "messages": state.get("messages", []) + [{"role": "agent", "content": findings}],
    }


def retrieve_node(state: AgentState) -> Dict[str, Any]:
    """Retrieve relevant code context."""
    workspace = state["workspace_dir"]
    file_list = tools.list_files(workspace)
    files = file_list.get("files", [])

    retrieved: List[Dict[str, Any]] = []
    for f in files[:5]:
        content_res = tools.read_file(workspace, f)
        if content_res.get("success"):
            retrieved.append({
                "file_path": f,
                "content": content_res.get("content", ""),
                "total_lines": content_res.get("total_lines", 0),
            })

    return {
        "status": "retrieved",
        "retrieved_context": retrieved,
    }


def plan_node(state: AgentState) -> Dict[str, Any]:
    """Generate repair plan based on task description and context."""
    desc = state["task_description"]
    retrieved = state.get("retrieved_context", [])
    files_str = ", ".join(c["file_path"] for c in retrieved)

    plan = f"Plan to address '{desc}': Inspect and patch targets in [{files_str}]."
    return {
        "status": "planning",
        "repair_plan": plan,
    }


def edit_node(state: AgentState) -> Dict[str, Any]:
    """Apply targeted patches to the repository workspace."""
    attempt = state.get("attempt_count", 0) + 1
    patches = state.get("proposed_patches", [])
    workspace = state["workspace_dir"]

    applied_count = 0
    for p in patches:
        fpath = p.get("file_path")
        code = p.get("code")
        s_line = p.get("start_line")
        e_line = p.get("end_line")
        if fpath and code:
            res = tools.apply_patch(workspace, fpath, code, s_line, e_line)
            if res.get("success"):
                applied_count += 1

    return {
        "status": "edited",
        "attempt_count": attempt,
    }


def test_node(state: AgentState) -> Dict[str, Any]:
    """Run pytest suite in sandbox targeting specific test or whole suite."""
    workspace = state["workspace_dir"]
    test_target = state.get("test_target")
    test_res = tools.run_tests(workspace, test_path=test_target)

    return {
        "status": "tested",
        "test_results": test_res,
    }


def verify_node(state: AgentState) -> Dict[str, Any]:
    """Verify test outputs."""
    test_res = state.get("test_results") or {}
    success = test_res.get("success", False) and test_res.get("failed", 0) == 0

    return {
        "is_verified": success,
        "status": "verified" if success else "verification_failed",
    }


def analyze_failure_node(state: AgentState) -> Dict[str, Any]:
    """Analyze test failure output to refine the next edit attempt."""
    test_res = state.get("test_results") or {}
    output = test_res.get("output", "No test output")

    analysis = f"Failure in attempt {state.get('attempt_count')}: {output[:300]}"
    return {
        "status": "analyzing_failure",
        "error_analysis": analysis,
        "messages": state.get("messages", []) + [{"role": "agent", "content": analysis}],
    }


def should_continue(state: AgentState) -> str:
    """Route based on verification result and retry budget."""
    if state.get("is_verified", False):
        return "human_approval"
    if state.get("attempt_count", 0) < state.get("max_attempts", 3):
        return "analyze_failure"
    return "failed"


def build_agent_graph():
    """Build and compile the LangGraph workflow."""
    builder = StateGraph(AgentState)

    builder.add_node("investigate", investigate_node)
    builder.add_node("retrieve", retrieve_node)
    builder.add_node("plan", plan_node)
    builder.add_node("edit", edit_node)
    builder.add_node("test", test_node)
    builder.add_node("verify", verify_node)
    builder.add_node("analyze_failure", analyze_failure_node)

    builder.add_edge(START, "investigate")
    builder.add_edge("investigate", "retrieve")
    builder.add_edge("retrieve", "plan")
    builder.add_edge("plan", "edit")
    builder.add_edge("edit", "test")
    builder.add_edge("test", "verify")

    builder.add_conditional_edges(
        "verify",
        should_continue,
        {
            "human_approval": END,
            "analyze_failure": "analyze_failure",
            "failed": END,
        },
    )
    builder.add_edge("analyze_failure", "edit")

    return builder.compile()


agent_app = build_agent_graph()
