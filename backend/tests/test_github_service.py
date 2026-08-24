"""Unit tests for GitHubService: URL parsing, clone/push confinement, PR creation.

No real network calls are made anywhere in this file -- httpx and GitPython
are mocked at the call boundary. Token-handling assertions verify the token
never appears in a raised error message (defense-in-depth scrubbing).
"""

import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import git
import httpx
import pytest

from app.services.github_service import (
    GitHubError,
    GitHubService,
    is_github_url,
    parse_github_url,
)

FAKE_TOKEN = "ghp_supersecrettoken1234567890"


# ---------------------------------------------------------------------------
# URL parsing / validation
# ---------------------------------------------------------------------------
@pytest.mark.parametrize(
    "url,expected",
    [
        ("https://github.com/octocat/Hello-World", ("octocat", "Hello-World")),
        ("https://github.com/octocat/Hello-World.git", ("octocat", "Hello-World")),
        ("https://github.com/octocat/Hello-World/", ("octocat", "Hello-World")),
        ("https://github.com/my-org/repo_name.py", ("my-org", "repo_name.py")),
    ],
)
def test_parse_github_url_valid(url, expected):
    assert parse_github_url(url) == expected


@pytest.mark.parametrize(
    "bad_url",
    [
        "git@github.com:octocat/Hello-World.git",  # SSH form
        "http://github.com/octocat/Hello-World",  # not https
        "https://gitlab.com/octocat/Hello-World",  # wrong host
        "https://github.com/octocat",  # missing repo segment
        "https://github.com/../../etc/passwd",  # traversal-style
        "not a url at all",
        "",
        "https://github.com/octocat/../secrets",
    ],
)
def test_parse_github_url_rejects_invalid(bad_url):
    with pytest.raises(GitHubError):
        parse_github_url(bad_url)


def test_is_github_url_predicate():
    assert is_github_url("https://github.com/octocat/Hello-World") is True
    assert is_github_url("C:/Users/dev/local_repo") is False
    assert is_github_url(None) is False
    assert is_github_url("") is False


# ---------------------------------------------------------------------------
# validate_repository (mocked httpx)
# ---------------------------------------------------------------------------
def _mock_response(status_code: int, json_data: dict | None = None, text: str = ""):
    resp = MagicMock()
    resp.status_code = status_code
    resp.json.return_value = json_data or {}
    resp.text = text
    return resp


def test_validate_repository_public_success_without_token():
    service = GitHubService()
    payload = {
        "clone_url": "https://github.com/octocat/Hello-World.git",
        "default_branch": "main",
        "private": False,
        "full_name": "octocat/Hello-World",
    }
    with patch("app.services.github_service.httpx.get", return_value=_mock_response(200, payload)) as mock_get:
        meta = service.validate_repository("octocat", "Hello-World", token=None)

    assert meta["clone_url"] == payload["clone_url"]
    assert meta["default_branch"] == "main"
    assert meta["private"] is False
    # No Authorization header sent when no token is provided.
    _, kwargs = mock_get.call_args
    assert "Authorization" not in kwargs["headers"]


def test_validate_repository_private_success_with_token():
    payload = {"clone_url": "https://github.com/acme/secret.git", "default_branch": "develop", "private": True}
    with patch("app.services.github_service.httpx.get", return_value=_mock_response(200, payload)) as mock_get:
        meta = GitHubService().validate_repository("acme", "secret", token=FAKE_TOKEN)

    assert meta["private"] is True
    _, kwargs = mock_get.call_args
    assert kwargs["headers"]["Authorization"] == f"Bearer {FAKE_TOKEN}"


def test_validate_repository_not_found_raises_clear_error():
    with patch("app.services.github_service.httpx.get", return_value=_mock_response(404)):
        with pytest.raises(GitHubError, match="not found"):
            GitHubService().validate_repository("nobody", "nothing", token=None)


def test_validate_repository_forbidden_raises():
    with patch("app.services.github_service.httpx.get", return_value=_mock_response(403)):
        with pytest.raises(GitHubError, match="denied"):
            GitHubService().validate_repository("owner", "repo", token=None)


def test_validate_repository_network_error_scrubs_token():
    with patch(
        "app.services.github_service.httpx.get",
        side_effect=httpx.ConnectError(f"boom token={FAKE_TOKEN}"),
    ):
        with pytest.raises(GitHubError) as exc_info:
            GitHubService().validate_repository("owner", "repo", token=FAKE_TOKEN)
    assert FAKE_TOKEN not in str(exc_info.value)


# ---------------------------------------------------------------------------
# clone_repository (mocked GitPython, real path-confinement logic)
# ---------------------------------------------------------------------------
def test_clone_repository_rejects_destination_outside_workspace_root():
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp) / "workspaces"
        root.mkdir()
        outside = Path(tmp) / "elsewhere" / "clone"

        with patch("app.services.github_service.git.Repo.clone_from") as mock_clone:
            with pytest.raises(GitHubError, match="outside the configured workspace root"):
                GitHubService().clone_repository("https://github.com/a/b.git", outside, root, token=None)
        mock_clone.assert_not_called()


def test_clone_repository_uses_authenticated_url_when_token_present():
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp) / "workspaces"
        root.mkdir()
        dest = root / "repos" / "a_b_12345678"

        with patch("app.services.github_service.git.Repo.clone_from") as mock_clone:
            result = GitHubService().clone_repository(
                "https://github.com/a/b.git", dest, root, token=FAKE_TOKEN
            )

        assert result == dest.resolve()
        called_url = mock_clone.call_args[0][0]
        assert FAKE_TOKEN in called_url  # embedded only for the ephemeral clone call
        assert called_url.startswith("https://x-access-token:")


def test_clone_repository_uses_shallow_depth_one():
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp) / "workspaces"
        root.mkdir()
        dest = root / "repos" / "a_b_shallow"

        with patch("app.services.github_service.git.Repo.clone_from") as mock_clone:
            GitHubService().clone_repository("https://github.com/a/b.git", dest, root, token=None)

        assert mock_clone.call_args.kwargs.get("depth") == 1


def test_clone_repository_public_repo_without_token_uses_plain_url():
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp) / "workspaces"
        root.mkdir()
        dest = root / "repos" / "a_b_87654321"

        with patch("app.services.github_service.git.Repo.clone_from") as mock_clone:
            GitHubService().clone_repository("https://github.com/a/b.git", dest, root, token=None)

        called_url = mock_clone.call_args[0][0]
        assert called_url == "https://github.com/a/b.git"


def test_clone_repository_rejects_existing_destination():
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp) / "workspaces"
        dest = root / "repos" / "already_here"
        dest.mkdir(parents=True)

        with patch("app.services.github_service.git.Repo.clone_from") as mock_clone:
            with pytest.raises(GitHubError, match="already exists"):
                GitHubService().clone_repository("https://github.com/a/b.git", dest, root, token=None)
        mock_clone.assert_not_called()


def test_clone_repository_scrubs_token_on_failure():
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp) / "workspaces"
        root.mkdir()
        dest = root / "repos" / "fail_case"

        with patch(
            "app.services.github_service.git.Repo.clone_from",
            side_effect=git.GitCommandError(
                "clone", 128, stderr=f"fatal: authentication failed for x-access-token:{FAKE_TOKEN}@github.com"
            ),
        ):
            with pytest.raises(GitHubError) as exc_info:
                GitHubService().clone_repository("https://github.com/a/b.git", dest, root, token=FAKE_TOKEN)
        assert FAKE_TOKEN not in str(exc_info.value)


# ---------------------------------------------------------------------------
# push_branch (mocked GitPython)
# ---------------------------------------------------------------------------
def test_push_branch_requires_token():
    with pytest.raises(GitHubError, match="requires GITHUB_TOKEN"):
        GitHubService().push_branch("/some/repo", "repopilot/task-1", "https://github.com/a/b.git", token=None)


def test_push_branch_success_uses_authenticated_url():
    mock_repo_instance = MagicMock()
    with patch("app.services.github_service.git.Repo", return_value=mock_repo_instance):
        GitHubService().push_branch(
            "/some/repo", "repopilot/task-1", "https://github.com/a/b.git", token=FAKE_TOKEN
        )

    pushed_url = mock_repo_instance.git.push.call_args[0][0]
    assert FAKE_TOKEN in pushed_url
    assert mock_repo_instance.git.push.call_args[0][1] == "repopilot/task-1:repopilot/task-1"


def test_push_branch_scrubs_token_on_failure():
    mock_repo_instance = MagicMock()
    mock_repo_instance.git.push.side_effect = git.GitCommandError(
        "push", 128, stderr=f"remote: rejected for https://x-access-token:{FAKE_TOKEN}@github.com/a/b.git"
    )
    with patch("app.services.github_service.git.Repo", return_value=mock_repo_instance):
        with pytest.raises(GitHubError) as exc_info:
            GitHubService().push_branch(
                "/some/repo", "repopilot/task-1", "https://github.com/a/b.git", token=FAKE_TOKEN
            )
    assert FAKE_TOKEN not in str(exc_info.value)


# ---------------------------------------------------------------------------
# create_pull_request (mocked httpx)
# ---------------------------------------------------------------------------
def test_create_pull_request_requires_token():
    with pytest.raises(GitHubError, match="requires GITHUB_TOKEN"):
        GitHubService().create_pull_request(
            "a", "b", "repopilot/task-1", "main", "title", "body", token=None
        )


def test_create_pull_request_success_sends_auth_header():
    payload = {"html_url": "https://github.com/a/b/pull/42", "number": 42}
    with patch(
        "app.services.github_service.httpx.post", return_value=_mock_response(201, payload)
    ) as mock_post:
        result = GitHubService().create_pull_request(
            "a", "b", "repopilot/task-1", "main", "Fix bug", "body text", token=FAKE_TOKEN
        )

    assert result == {"url": "https://github.com/a/b/pull/42", "number": 42}
    _, kwargs = mock_post.call_args
    assert kwargs["headers"]["Authorization"] == f"Bearer {FAKE_TOKEN}"
    assert kwargs["json"]["head"] == "repopilot/task-1"
    assert kwargs["json"]["base"] == "main"


def test_create_pull_request_failure_scrubs_token_and_raises():
    with patch(
        "app.services.github_service.httpx.post",
        return_value=_mock_response(422, text=f"validation failed token={FAKE_TOKEN}"),
    ):
        with pytest.raises(GitHubError) as exc_info:
            GitHubService().create_pull_request(
                "a", "b", "repopilot/task-1", "main", "Fix bug", "body", token=FAKE_TOKEN
            )
    assert FAKE_TOKEN not in str(exc_info.value)
