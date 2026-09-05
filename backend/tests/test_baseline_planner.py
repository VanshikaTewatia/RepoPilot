"""Unit tests for the evidence-driven reproduction planner (Phase 4B-1):
app.services.baseline.planner / app.services.baseline.plan_validator.

Gemini is always mocked -- no real API calls, no network access. Mirrors
the exact mocking convention already used by
tests/test_qa_classifier.py::classify_question (a `real_looking_key`
fixture to opt into the real-call code path, and a `genai.Client` patch
returning a canned JSON payload).
"""

import json
from unittest.mock import MagicMock, patch

import pytest

from app.core.config import settings
from app.services.baseline import (
    BaselineExecutor,
    EvidenceReference,
    ExitCodeSemantics,
    KnownCommand,
    ReproductionExpectation,
    ReproductionInput,
    ReproductionPlan,
    ReproductionType,
    RepositoryEvidence,
    plan_reproduction,
    validate_plan,
)
from app.services.baseline.plan_validator import (
    MAX_PLAN_COMMANDS,
    MAX_ARGV_TOKEN_CHARS,
)
from app.services.verification.engine import VerificationEngine
from app.services.verification.project_analyzer import ProjectInfo


@pytest.fixture
def real_looking_key(monkeypatch):
    """A key that doesn't start with 'test'/'mock', so plan_reproduction
    actually attempts a (mocked) Gemini call instead of short-circuiting."""
    monkeypatch.setattr(settings, "gemini_api_key", "AIzaFakeRealLookingKey123")


def _mock_gemini_response(payload: dict):
    mock_response = MagicMock()
    mock_response.text = json.dumps(payload)
    mock_client = MagicMock()
    mock_client.models.generate_content.return_value = mock_response
    return patch("app.services.baseline.planner.genai.Client", return_value=mock_client)


def _node_project() -> ProjectInfo:
    return ProjectInfo(
        root=".",
        ecosystem="node",
        languages=["JavaScript", "TypeScript"],
        frameworks=["React", "Next.js"],
        build_system="node",
        package_manager="npm",
        test_system="jest",
        evidence=["package.json"],
    )


def _minimal_plan_payload(**overrides) -> dict:
    payload = {
        "applicable": True,
        "reason": (
            "tests/test_cart.py::test_subtotal exercises the reported "
            "regression directly and fails today because of it"
        ),
        "reproduction_type": "test_failure",
        "commands": [["npm", "test"]],
        "working_dir": None,
        "expected_observation": "The existing cart subtotal test fails.",
        "exit_code_semantics": "nonzero_is_reproduced",
        "reproduced_output_pattern": None,
        "not_reproduced_output_pattern": None,
        "confidence": 0.8,
        "evidence_refs": ["package.json"],
        "project_root": ".",
        "ecosystem": "node",
        "image": None,
        "timeout_seconds": 60,
    }
    payload.update(overrides)
    return payload


def _evidence_with_known_npm_test() -> RepositoryEvidence:
    return RepositoryEvidence(
        detected_projects=[_node_project()],
        known_commands=[
            KnownCommand(command=["npm", "test"], description="runs jest", source_file="package.json"),
        ],
        investigation_findings="The cart subtotal calculation lives in src/cart/subtotal.js.",
        evidence_references=[
            EvidenceReference(file_path="package.json", description="declares the test script"),
        ],
    )


# ===========================================================================
# 1. Strong evidence -> valid reproduction plan
# ===========================================================================
@pytest.mark.asyncio
async def test_strong_evidence_produces_valid_applicable_plan(real_looking_key):
    evidence = _evidence_with_known_npm_test()
    with _mock_gemini_response(_minimal_plan_payload()):
        plan = await plan_reproduction("The cart subtotal is wrong for VIP customers", evidence)

    assert plan.applicable is True
    assert plan.reproduction_type == ReproductionType.TEST_FAILURE
    assert plan.commands == [["npm", "test"]]
    assert plan.exit_code_semantics == ExitCodeSemantics.NONZERO_IS_REPRODUCED
    assert plan.expected_observation
    # 5. existing valid-plan behavior: planning genuinely succeeded.
    assert plan.planning_failed is False


# ===========================================================================
# 2. No relevant evidence -> NOT_APPLICABLE (a *genuine* verdict: planning
# succeeded and correctly concluded nothing applies -- planning_failed must
# be False here, distinguishing this from a planner infrastructure failure).
# ===========================================================================
@pytest.mark.asyncio
async def test_no_relevant_evidence_returns_not_applicable(real_looking_key):
    empty_evidence = RepositoryEvidence()
    payload = _minimal_plan_payload(
        applicable=False,
        reason="No repository evidence supports any specific reproduction for this report.",
        reproduction_type="not_applicable",
        commands=[],
        expected_observation=None,
        evidence_refs=[],
    )
    with _mock_gemini_response(payload):
        plan = await plan_reproduction("Something is broken somewhere", empty_evidence)

    assert plan.applicable is False
    assert plan.reproduction_type == ReproductionType.NOT_APPLICABLE
    assert plan.commands == []
    assert plan.planning_failed is False  # a real verdict, not a planner failure


# ===========================================================================
# 3. User's claimed framework conflicts with repository evidence -> planner
# follows repository evidence (both a prompt-contract test and an
# accept-the-evidence-backed-plan test).
# ===========================================================================
@pytest.mark.asyncio
async def test_prompt_marks_repository_evidence_as_authoritative_over_user_wording(real_looking_key):
    evidence = RepositoryEvidence(
        detected_projects=[
            ProjectInfo(root=".", ecosystem="node", languages=["JavaScript"], frameworks=["Vue"], evidence=["package.json"])
        ],
    )
    captured = {}

    def _capture(*args, **kwargs):
        captured["contents"] = kwargs.get("contents")
        captured["system_instruction"] = kwargs["config"]["system_instruction"]
        response = MagicMock()
        response.text = json.dumps(_minimal_plan_payload(
            applicable=False, reason="no evidence", reproduction_type="not_applicable",
            commands=[], expected_observation=None, evidence_refs=[],
        ))
        return response

    mock_client = MagicMock()
    mock_client.models.generate_content.side_effect = _capture
    with patch("app.services.baseline.planner.genai.Client", return_value=mock_client):
        await plan_reproduction("React login is broken", evidence)

    assert "React login is broken" in captured["contents"]
    assert "Vue" in captured["contents"]
    assert "authoritative" in captured["system_instruction"].lower()


@pytest.mark.asyncio
async def test_planner_accepts_evidence_backed_plan_that_ignores_conflicting_user_framework(real_looking_key):
    """Repository evidence shows Vue; the user claimed React. A plan that
    correctly follows the Vue evidence (not React-specific commands) must
    be accepted."""
    evidence = RepositoryEvidence(
        detected_projects=[
            ProjectInfo(root=".", ecosystem="node", languages=["JavaScript"], frameworks=["Vue"], test_system="vitest", evidence=["package.json"])
        ],
        known_commands=[KnownCommand(command=["npm", "run", "test"], description="runs vitest", source_file="package.json")],
        evidence_references=[EvidenceReference(file_path="package.json")],
    )
    payload = _minimal_plan_payload(
        commands=[["npm", "run", "test"]],
        reason="The Vue component's existing vitest suite covers login and fails due to this bug.",
        evidence_refs=["package.json"],
    )
    with _mock_gemini_response(payload):
        plan = await plan_reproduction("React login is broken", evidence)

    assert plan.applicable is True
    assert plan.commands == [["npm", "run", "test"]]


# ===========================================================================
# 4. Existing package.json script is preferred over invented command
# ===========================================================================
@pytest.mark.asyncio
async def test_prompt_presents_known_commands_and_instructs_preference(real_looking_key):
    evidence = _evidence_with_known_npm_test()
    captured = {}

    def _capture(*args, **kwargs):
        captured["contents"] = kwargs.get("contents")
        response = MagicMock()
        response.text = json.dumps(_minimal_plan_payload())
        return response

    mock_client = MagicMock()
    mock_client.models.generate_content.side_effect = _capture
    with patch("app.services.baseline.planner.genai.Client", return_value=mock_client):
        await plan_reproduction("cart bug", evidence)

    assert "KNOWN COMMANDS" in captured["contents"]
    assert "npm" in captured["contents"] and "test" in captured["contents"]
    assert "package.json" in captured["contents"]


@pytest.mark.asyncio
async def test_plan_using_known_command_validates_evidence_ref_successfully(real_looking_key):
    evidence = _evidence_with_known_npm_test()
    with _mock_gemini_response(_minimal_plan_payload()):
        plan = await plan_reproduction("cart bug", evidence)

    assert plan.applicable is True
    assert plan.commands == [["npm", "test"]]  # the known command, not an invented one


# ===========================================================================
# 5. Empty commands rejected
# ===========================================================================
def test_validator_rejects_applicable_plan_with_empty_commands():
    plan = ReproductionPlan(
        applicable=True,
        reason="x",
        reproduction_type=ReproductionType.TEST_FAILURE,
        commands=[],
        expected_observation="something",
    )
    result = validate_plan(plan, RepositoryEvidence())
    assert result.valid is False
    assert any("command" in e for e in result.errors)


# ===========================================================================
# 6. Absolute working_dir rejected
# ===========================================================================
def test_validator_rejects_absolute_working_dir():
    plan = ReproductionPlan(
        applicable=True, reason="x", reproduction_type=ReproductionType.TEST_FAILURE,
        commands=[["npm", "test"]], expected_observation="x",
        working_dir="/etc",
    )
    result = validate_plan(plan, RepositoryEvidence())
    assert result.valid is False
    assert any("working_dir" in e for e in result.errors)


# ===========================================================================
# 7. ../ working_dir rejected
# ===========================================================================
def test_validator_rejects_traversal_working_dir():
    plan = ReproductionPlan(
        applicable=True, reason="x", reproduction_type=ReproductionType.TEST_FAILURE,
        commands=[["npm", "test"]], expected_observation="x",
        working_dir="../../etc",
    )
    result = validate_plan(plan, RepositoryEvidence())
    assert result.valid is False
    assert any("working_dir" in e for e in result.errors)


# ===========================================================================
# 8. Invalid regex rejected
# ===========================================================================
def test_validator_rejects_invalid_reproduced_pattern():
    plan = ReproductionPlan(
        applicable=True, reason="x", reproduction_type=ReproductionType.RUNTIME_BEHAVIOR,
        commands=[["python", "repro.py"]], expected_observation="x",
        reproduced_output_pattern="([unbalanced",
    )
    result = validate_plan(plan, RepositoryEvidence())
    assert result.valid is False
    assert any("reproduced_output_pattern" in e for e in result.errors)


def test_validator_rejects_invalid_not_reproduced_pattern():
    plan = ReproductionPlan(
        applicable=True, reason="x", reproduction_type=ReproductionType.RUNTIME_BEHAVIOR,
        commands=[["python", "repro.py"]], expected_observation="x",
        not_reproduced_output_pattern="[a-z",
    )
    result = validate_plan(plan, RepositoryEvidence())
    assert result.valid is False


# ===========================================================================
# 9. Unsupported exit-code semantics rejected
# ===========================================================================
def test_validator_rejects_unsupported_exit_code_semantics_bypassing_the_enum():
    plan = ReproductionPlan(
        applicable=True, reason="x", reproduction_type=ReproductionType.TEST_FAILURE,
        commands=[["npm", "test"]], expected_observation="x",
    )
    plan.exit_code_semantics = "always_reproduced"  # bypass the enum directly
    result = validate_plan(plan, RepositoryEvidence())
    assert result.valid is False
    assert any("exit_code_semantics" in e for e in result.errors)


@pytest.mark.asyncio
async def test_planner_reports_unsupported_exit_code_semantics_as_planning_failure_not_not_applicable(real_looking_key):
    """4. Invalid/unsafe LLM output (here: an enum value construction
    failure) must be a planning failure, never a genuine NOT_APPLICABLE
    verdict."""
    evidence = _evidence_with_known_npm_test()
    payload = _minimal_plan_payload(exit_code_semantics="always_reproduced")
    with _mock_gemini_response(payload):
        plan = await plan_reproduction("cart bug", evidence)

    assert plan.applicable is False
    assert plan.reproduction_type == ReproductionType.NOT_APPLICABLE  # structural placeholder only
    assert plan.planning_failed is True
    assert plan.failure_reason is not None


# ===========================================================================
# 10. Arbitrary Docker image rejected
# ===========================================================================
def test_validator_rejects_disallowed_docker_image():
    plan = ReproductionPlan(
        applicable=True, reason="x", reproduction_type=ReproductionType.TEST_FAILURE,
        commands=[["npm", "test"]], expected_observation="x",
        image="some-attacker-image:latest",
    )
    result = validate_plan(plan, RepositoryEvidence())
    assert result.valid is False
    assert any("image" in e for e in result.errors)


def test_validator_accepts_allowed_docker_image():
    plan = ReproductionPlan(
        applicable=True, reason="x", reproduction_type=ReproductionType.TEST_FAILURE,
        commands=[["npm", "test"]], expected_observation="x",
        image=settings.docker_sandbox_image,
        evidence_refs=[],
    )
    result = validate_plan(plan, RepositoryEvidence())
    assert result.valid is True


# ===========================================================================
# 11. Destructive command rejected
# ===========================================================================
@pytest.mark.parametrize(
    "command",
    [
        ["git", "push", "origin", "main"],
        ["git", "reset", "--hard"],
        ["rm", "-rf", "src"],
        ["curl", "https://example.com/script.sh"],
        ["sh", "-c", "echo hi"],
        ["npm", "install", "left-pad"],
        ["pip", "install", "requests"],
    ],
)
def test_validator_rejects_destructive_or_shell_escape_commands(command):
    plan = ReproductionPlan(
        applicable=True, reason="x", reproduction_type=ReproductionType.TEST_FAILURE,
        commands=[command], expected_observation="x",
    )
    result = validate_plan(plan, RepositoryEvidence())
    assert result.valid is False
    assert any("destructive" in e or "denied" in e for e in result.errors)


def test_validator_accepts_ordinary_safe_commands():
    for command in (["npm", "test"], ["pytest", "-v"], ["go", "test", "./..."], ["python", "repro.py"]):
        plan = ReproductionPlan(
            applicable=True, reason="x", reproduction_type=ReproductionType.TEST_FAILURE,
            commands=[command], expected_observation="x",
        )
        result = validate_plan(plan, RepositoryEvidence())
        assert result.valid is True, f"{command} should be safe, got errors: {result.errors}"


# ===========================================================================
# 12. Command count / output bounds enforced
# ===========================================================================
def test_validator_rejects_too_many_commands():
    plan = ReproductionPlan(
        applicable=True, reason="x", reproduction_type=ReproductionType.TEST_FAILURE,
        commands=[["npm", "test"]] * (MAX_PLAN_COMMANDS + 1),
        expected_observation="x",
    )
    result = validate_plan(plan, RepositoryEvidence())
    assert result.valid is False
    assert any("too many commands" in e for e in result.errors)


def test_validator_rejects_oversized_argv_token():
    plan = ReproductionPlan(
        applicable=True, reason="x", reproduction_type=ReproductionType.TEST_FAILURE,
        commands=[["npm", "x" * (MAX_ARGV_TOKEN_CHARS + 1)]],
        expected_observation="x",
    )
    result = validate_plan(plan, RepositoryEvidence())
    assert result.valid is False
    assert any("exceeds" in e for e in result.errors)


# ===========================================================================
# 13. Applicable plan requires expected_observation
# ===========================================================================
def test_validator_rejects_applicable_plan_missing_expected_observation():
    plan = ReproductionPlan(
        applicable=True, reason="x", reproduction_type=ReproductionType.TEST_FAILURE,
        commands=[["npm", "test"]], expected_observation=None,
    )
    result = validate_plan(plan, RepositoryEvidence())
    assert result.valid is False
    assert any("expected_observation" in e for e in result.errors)


# ===========================================================================
# 14. NOT_APPLICABLE plan cannot contain executable commands
# ===========================================================================
def test_validator_rejects_not_applicable_plan_with_commands():
    plan = ReproductionPlan(
        applicable=False, reason="x", reproduction_type=ReproductionType.NOT_APPLICABLE,
        commands=[["npm", "test"]],
    )
    result = validate_plan(plan, RepositoryEvidence())
    assert result.valid is False
    assert any("NOT_APPLICABLE plan must not contain" in e for e in result.errors)


# ===========================================================================
# 15. Planner cannot execute commands itself
# ===========================================================================
@pytest.mark.asyncio
async def test_planner_never_invokes_the_sandbox(real_looking_key):
    evidence = _evidence_with_known_npm_test()
    with _mock_gemini_response(_minimal_plan_payload()), patch.object(
        VerificationEngine, "execute_command"
    ) as mock_execute, patch.object(BaselineExecutor, "run") as mock_run:
        await plan_reproduction("cart bug", evidence)

    mock_execute.assert_not_called()
    mock_run.assert_not_called()


# ===========================================================================
# 16. Malformed LLM JSON handled deterministically -- and, per the hardening
# review, this is a PLANNING FAILURE, never a genuine NOT_APPLICABLE verdict
# (requirement 3: "Malformed JSON != NOT_APPLICABLE").
# ===========================================================================
@pytest.mark.asyncio
async def test_malformed_llm_json_is_planning_failure_not_not_applicable(real_looking_key):
    mock_response = MagicMock()
    mock_response.text = "this is not json at all {{{"
    mock_client = MagicMock()
    mock_client.models.generate_content.return_value = mock_response

    with patch("app.services.baseline.planner.genai.Client", return_value=mock_client):
        plan = await plan_reproduction("cart bug", _evidence_with_known_npm_test())  # must not raise

    assert plan.applicable is False
    assert plan.reproduction_type == ReproductionType.NOT_APPLICABLE  # structural placeholder only
    assert plan.planning_failed is True
    assert plan.failure_reason is not None


# ===========================================================================
# 17. LLM returns a valid-looking but unsafe plan -> validator rejects it,
# and this is a PLANNING FAILURE, never a genuine NOT_APPLICABLE verdict
# (requirement 4: "Invalid/unsafe plan != NOT_APPLICABLE").
# ===========================================================================
@pytest.mark.asyncio
async def test_llm_returns_unsafe_plan_and_validator_marks_it_a_planning_failure(real_looking_key):
    payload = _minimal_plan_payload(commands=[["git", "push", "origin", "main"]])
    with _mock_gemini_response(payload):
        plan = await plan_reproduction("cart bug", _evidence_with_known_npm_test())  # must not raise/execute anything

    assert plan.applicable is False
    assert plan.reproduction_type == ReproductionType.NOT_APPLICABLE  # structural placeholder only
    assert plan.planning_failed is True
    assert "validation" in plan.reason.lower() or "rejected" in plan.reason.lower()
    assert plan.failure_reason == plan.reason


# ===========================================================================
# 18. Planner does not manufacture a reproduction when evidence is
# insufficient (even if the LLM tries to) -- also a planning failure, since
# the LLM DID propose something, it was just unsafe/unfounded.
# ===========================================================================
@pytest.mark.asyncio
async def test_planner_rejects_fabricated_plan_against_insufficient_evidence(real_looking_key):
    empty_evidence = RepositoryEvidence()  # nothing known at all
    payload = _minimal_plan_payload(evidence_refs=["src/imaginary/file.py"])
    with _mock_gemini_response(payload):
        plan = await plan_reproduction("cart bug", empty_evidence)

    # the LLM tried to cite a file that was never part of the evidence --
    # the validator must catch this and downgrade, never pass it through.
    assert plan.applicable is False
    assert plan.reproduction_type == ReproductionType.NOT_APPLICABLE
    assert plan.planning_failed is True


# ===========================================================================
# 19. Evidence references must point to known investigation evidence
# ===========================================================================
def test_validator_rejects_evidence_ref_not_in_known_evidence():
    evidence = _evidence_with_known_npm_test()
    plan = ReproductionPlan(
        applicable=True, reason="x", reproduction_type=ReproductionType.TEST_FAILURE,
        commands=[["npm", "test"]], expected_observation="x",
        evidence_refs=["src/totally/made/up.py"],
    )
    result = validate_plan(plan, evidence)
    assert result.valid is False
    assert any("evidence_ref" in e for e in result.errors)


def test_validator_accepts_evidence_ref_with_line_range_citation():
    evidence = RepositoryEvidence(
        evidence_references=[EvidenceReference(file_path="src/cart/subtotal.js", line_start=10, line_end=20)],
    )
    plan = ReproductionPlan(
        applicable=True, reason="x", reproduction_type=ReproductionType.TEST_FAILURE,
        commands=[["npm", "test"]], expected_observation="x",
        evidence_refs=["src/cart/subtotal.js:10-20"],
    )
    result = validate_plan(plan, evidence)
    assert result.valid is True


def test_validator_rejects_evidence_refs_when_no_evidence_object_given():
    plan = ReproductionPlan(
        applicable=True, reason="x", reproduction_type=ReproductionType.TEST_FAILURE,
        commands=[["npm", "test"]], expected_observation="x",
        evidence_refs=["package.json"],
    )
    result = validate_plan(plan, None)
    assert result.valid is False


# ===========================================================================
# 20. Planner confidence is bounded/validated
# ===========================================================================
@pytest.mark.parametrize("bad_confidence", [-0.1, 1.1, 5.0, -3.0])
def test_validator_rejects_out_of_bounds_confidence(bad_confidence):
    plan = ReproductionPlan(
        applicable=True, reason="x", reproduction_type=ReproductionType.TEST_FAILURE,
        commands=[["npm", "test"]], expected_observation="x",
        confidence=bad_confidence,
    )
    result = validate_plan(plan, RepositoryEvidence())
    assert result.valid is False
    assert any("confidence" in e for e in result.errors)


@pytest.mark.parametrize("good_confidence", [0.0, 0.5, 1.0])
def test_validator_accepts_in_bounds_confidence(good_confidence):
    plan = ReproductionPlan(
        applicable=True, reason="x", reproduction_type=ReproductionType.TEST_FAILURE,
        commands=[["npm", "test"]], expected_observation="x",
        confidence=good_confidence,
    )
    result = validate_plan(plan, RepositoryEvidence())
    assert result.valid is True


# ===========================================================================
# Hardening pass: planner infrastructure failure MUST NOT be reported as
# NOT_APPLICABLE (requirement 2: "Gemini/API failure != NOT_APPLICABLE").
# ===========================================================================
@pytest.mark.asyncio
async def test_gemini_unconfigured_is_planning_failure_not_not_applicable():
    # default test conftest key ("test_gemini_key_123") is treated as unconfigured --
    # no network call is even attempted, but this is still not a genuine
    # "no reproduction exists" verdict: planning never actually ran.
    plan = await plan_reproduction("cart bug", _evidence_with_known_npm_test())
    assert plan.applicable is False
    assert plan.reproduction_type == ReproductionType.NOT_APPLICABLE  # structural placeholder only
    assert plan.planning_failed is True
    assert plan.failure_reason is not None


@pytest.mark.asyncio
async def test_gemini_call_raising_mid_call_is_planning_failure_not_not_applicable(real_looking_key):
    """A live Gemini call that fails outright (network error, 503, 429,
    quota exceeded, ...) must never be conflated with a genuine
    NOT_APPLICABLE verdict -- planning simply never completed."""
    mock_client = MagicMock()
    mock_client.models.generate_content.side_effect = RuntimeError("503 Service Unavailable")

    with patch("app.services.baseline.planner.genai.Client", return_value=mock_client):
        plan = await plan_reproduction("cart bug", _evidence_with_known_npm_test())  # must not raise

    assert plan.applicable is False
    assert plan.reproduction_type == ReproductionType.NOT_APPLICABLE  # structural placeholder only
    assert plan.planning_failed is True
    assert "503" in plan.failure_reason


# ===========================================================================
# Empty task description IS a genuine, deterministic NOT_APPLICABLE verdict
# (there is truly nothing to plan a reproduction for) -- distinct from an
# infrastructure failure, so planning_failed must stay False here.
# ===========================================================================
@pytest.mark.asyncio
async def test_empty_task_description_returns_not_applicable():
    plan = await plan_reproduction("", _evidence_with_known_npm_test())
    assert plan.applicable is False
    assert plan.planning_failed is False


# ===========================================================================
# A validated plan maps cleanly onto Phase 4A's ReproductionInput shape
# (structural compatibility only -- Phase 4B-1 never constructs or executes
# a ReproductionInput itself).
# ===========================================================================
def test_validated_plan_fields_are_structurally_compatible_with_reproduction_input():
    plan = ReproductionPlan(
        applicable=True, reason="x", reproduction_type=ReproductionType.TEST_FAILURE,
        commands=[["npm", "test"]], expected_observation="x",
        exit_code_semantics=ExitCodeSemantics.NONZERO_IS_REPRODUCED,
        working_dir=None, timeout_seconds=60,
    )
    assert validate_plan(plan, RepositoryEvidence()).valid is True

    # This mirrors what a Phase 4B-2 integration would do -- not exercised
    # or executed here.
    repro_input = ReproductionInput(
        workspace_path="/some/workspace",
        commands=plan.commands,
        working_dir=plan.working_dir,
        timeout_seconds=plan.timeout_seconds,
        expectation=ReproductionExpectation(
            exit_code_semantics=plan.exit_code_semantics,
            reproduced_output_pattern=plan.reproduced_output_pattern,
            not_reproduced_output_pattern=plan.not_reproduced_output_pattern,
        ),
    )
    assert repro_input.commands == [["npm", "test"]]
