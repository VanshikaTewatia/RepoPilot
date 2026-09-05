"""Typed Agent State definitions for LangGraph."""

from typing import Any, Dict, List, Optional
from typing_extensions import TypedDict


class AgentState(TypedDict, total=False):
    """Complete state representation passed across LangGraph nodes."""

    task_id: int
    repository_id: int
    workspace_dir: str
    task_description: str
    status: str
    attempt_count: int
    max_attempts: int
    test_target: Optional[str]
    investigation_findings: str
    keyword_matches: List[Dict[str, Any]]
    retrieved_context: List[Dict[str, Any]]
    repair_plan: str
    proposed_patches: List[Dict[str, Any]]
    test_results: Optional[Dict[str, Any]]
    error_analysis: Optional[str]
    is_verified: bool
    messages: List[Dict[str, Any]]
    # How many of proposed_patches actually applied successfully in the most
    # recent edit_node pass (distinct from len(proposed_patches), which only
    # reflects what was *generated*). None when not yet set (e.g. before the
    # first edit attempt). See app.services.agent.graph.finalize_node --
    # verification passing with patches proposed but none applied must never
    # be reported as FIXED.
    applied_patch_count: Optional[int]
    # Task-level outcome classification, distinct from is_verified/status:
    # "FIXED" | "NO_CHANGE_NEEDED" | "UNABLE_TO_VERIFY" | "FAILED".
    # See app.services.agent.graph.finalize_node.
    outcome: Optional[str]
    outcome_detail: Optional[str]
    # Phase 4B-3: evidence-driven baseline reproduction, run once between
    # investigate and retrieve -- see app.services.agent.graph.baseline_node.
    # Baseline reproduction is evidence gathering, never proof of
    # correctness: "REPRODUCED" | "NOT_REPRODUCED" | "UNABLE_TO_REPRODUCE" |
    # "NOT_APPLICABLE". A planner/bridge failure (planning_failed=True, or a
    # bridge PLANNING_FAILED outcome) is always surfaced here as
    # "UNABLE_TO_REPRODUCE", never "NOT_APPLICABLE" -- see
    # app.services.baseline's own planning_failed/BridgeOutcome semantics.
    # finalize_node is the only place this is allowed to affect the task's
    # final outcome, and only to prevent a false FIXED claim.
    baseline_status: Optional[str]
    # The full app.services.baseline.BaselineResult.to_dict() when Phase 4A
    # actually executed a reproduction (None for NOT_APPLICABLE/
    # UNABLE_TO_REPRODUCE cases that never reached execution).
    baseline_result: Optional[Dict[str, Any]]
    # Human-readable explanation of baseline_status, always set regardless
    # of whether execution happened.
    baseline_detail: Optional[str]
    # Phase 5: the exact validated execution specification (a minimal,
    # JSON-safe serialization of the ReproductionInput actually run for
    # baseline reproduction -- see
    # app.services.agent.graph._reproduction_input_to_state_dict) retained
    # ONLY so post_fix_reproduction_node can rerun the IDENTICAL procedure
    # after an edit is applied -- never a new plan, never regenerated
    # commands/image/working_dir. None whenever baseline never reached a
    # real execution (NOT_APPLICABLE / UNABLE_TO_REPRODUCE / planning
    # failure). Deliberately excludes command output -- that lives in
    # baseline_result / post_fix_reproduction_result, each already bounded
    # by Phase 4A.
    reproduction_spec: Optional[Dict[str, Any]]
    # Phase 5: outcome of rerunning reproduction_spec after edit_node
    # applies a candidate fix -- "REPRODUCED" (the previously-established
    # failure is still present) | "NOT_REPRODUCED" (no longer observed) |
    # "UNABLE_TO_REPRODUCE". Only ever set when baseline_status ==
    # "REPRODUCED" and this attempt's own test/verify already passed; see
    # app.services.agent.graph.post_fix_reproduction_node. finalize_node is
    # the only place this is allowed to gate FIXED -- it can prevent FIXED,
    # never manufacture it on its own.
    post_fix_reproduction_status: Optional[str]
    post_fix_reproduction_result: Optional[Dict[str, Any]]
    post_fix_reproduction_detail: Optional[str]
    # Phase 6A: evidence-driven root-cause diagnosis, run once between
    # retrieve and plan (and refreshed on every retry) -- see
    # app.services.agent.graph.diagnose_node. Advisory only: plan_node may
    # use a DIAGNOSED result to inform its Gemini patch prompt, but a
    # missing/failed/insufficient diagnosis must never block or alter patch
    # generation, and diagnosis must never affect finalize_node/
    # should_continue. "DIAGNOSED" | "INSUFFICIENT_EVIDENCE" |
    # "DIAGNOSIS_FAILED" -- see app.services.diagnosis.DiagnosisStatus.
    # DIAGNOSIS_FAILED is a process failure, never a verdict; it must never
    # be treated as equivalent to INSUFFICIENT_EVIDENCE.
    diagnosis_status: Optional[str]
    # The full app.services.diagnosis.Diagnosis.to_dict() when diagnosis
    # actually produced a result (None before diagnose_node first runs).
    diagnosis: Optional[Dict[str, Any]]
    # Human-readable explanation of diagnosis_status, always set once
    # diagnose_node has run.
    diagnosis_detail: Optional[str]
