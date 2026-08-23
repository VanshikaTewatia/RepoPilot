"""GitHub repository operations: URL parsing, cloning, branch push, and PR creation.

All network calls are synchronous (httpx) and all git operations use GitPython,
matching the synchronous-client-wrapped-in-asyncio.to_thread convention already
used by GeminiEmbeddingProvider. A token is always passed in explicitly by the
caller -- this module never reads `settings` directly, so `settings.github_token`
in `config.py` remains the single source of truth for where the token comes from,
and every method here stays trivially mockable without touching global state.
"""

import re
from pathlib import Path
from typing import Optional, Tuple

import git
import httpx

DEFAULT_GITHUB_API_BASE = "https://api.github.com"

# Strict allow-list: only `https://github.com/<owner>/<repo>`, optionally with a
# trailing `.git` and/or slash. Owner/repo charsets follow GitHub's own rules and
# exclude path-traversal characters. SSH URLs (git@...) and any other host are
# rejected outright.
_GITHUB_URL_PATTERN = re.compile(
    r"^https://github\.com/"
    r"(?P<owner>[A-Za-z0-9](?:[A-Za-z0-9-]{0,38})?)/"
    r"(?P<repo>[A-Za-z0-9._-]+?)"
    r"(?:\.git)?/?$"
)


class GitHubError(Exception):
    """Raised for any GitHub validation/clone/push/PR failure.

    Message text is always token-safe -- callers may surface it directly in
    an HTTPException detail or log line.
    """


def _scrub(text: str, token: Optional[str]) -> str:
    """Strip a token value (and any embedded HTTPS credential) from free text.

    Defense in depth: GitPython/httpx exceptions can echo back the URL they
    failed on, which may contain the authenticated clone/push URL. This is
    called on every error path before the message leaves this module.
    """
    if not text:
        return text
    if token:
        text = text.replace(token, "***")
    return re.sub(r"https://[^/@\s]+@", "https://***@", text)


def parse_github_url(url: str) -> Tuple[str, str]:
    """Parse and strictly validate a GitHub HTTPS repository URL.

    Returns (owner, repo). Raises GitHubError for SSH URLs, non-github.com
    hosts, malformed input, or anything containing path-traversal segments.
    """
    if not isinstance(url, str) or not url.strip():
        raise GitHubError("Repository URL is required.")

    match = _GITHUB_URL_PATTERN.match(url.strip())
    if not match:
        raise GitHubError(
            "Invalid GitHub repository URL. Expected format: "
            "https://github.com/<owner>/<repo>"
        )

    owner, repo = match.group("owner"), match.group("repo")
    if ".." in owner or ".." in repo or "/" in owner or "/" in repo:
        raise GitHubError("Invalid GitHub repository URL.")
    return owner, repo


def is_github_url(url: Optional[str]) -> bool:
    """Cheap predicate used to branch approval-flow behavior without re-parsing."""
    if not url:
        return False
    try:
        parse_github_url(url)
        return True
    except GitHubError:
        return False


class GitHubService:
    """Stateless GitHub operations: validate, clone, branch push, open PRs."""

    def __init__(self, api_base_url: str = DEFAULT_GITHUB_API_BASE, timeout: float = 15.0):
        self.api_base_url = api_base_url.rstrip("/")
        self.timeout = timeout

    @staticmethod
    def _headers(token: Optional[str]) -> dict:
        headers = {
            "Accept": "application/vnd.github+json",
            "X-GitHub-Api-Version": "2022-11-28",
        }
        if token:
            headers["Authorization"] = f"Bearer {token}"
        return headers

    def validate_repository(self, owner: str, repo: str, token: Optional[str] = None) -> dict:
        """Confirm the repository exists and is reachable; return clone metadata.

        Works without a token for public repositories. Raises GitHubError
        with a clear, actionable message on 404 (not found, or private
        without a configured token) and 403 (rate-limited/forbidden).
        """
        url = f"{self.api_base_url}/repos/{owner}/{repo}"
        try:
            resp = httpx.get(url, headers=self._headers(token), timeout=self.timeout)
        except httpx.HTTPError as e:
            raise GitHubError(f"Could not reach GitHub: {_scrub(str(e), token)}") from None

        if resp.status_code == 404:
            raise GitHubError(
                f"Repository '{owner}/{repo}' was not found. If it is private, "
                "configure GITHUB_TOKEN with access to it on the server."
            )
        if resp.status_code == 403:
            raise GitHubError(
                "GitHub denied the request (rate-limited or forbidden). "
                "Configure GITHUB_TOKEN on the server, or try again later."
            )
        if resp.status_code != 200:
            raise GitHubError(
                f"GitHub API returned unexpected status {resp.status_code} for '{owner}/{repo}'."
            )

        data = resp.json()
        return {
            "clone_url": data.get("clone_url") or f"https://github.com/{owner}/{repo}.git",
            "default_branch": data.get("default_branch") or "main",
            "private": bool(data.get("private", False)),
            "full_name": data.get("full_name", f"{owner}/{repo}"),
        }

    @staticmethod
    def _authenticated_url(clone_url: str, token: Optional[str]) -> str:
        """Embed a token as an HTTPS credential for one ephemeral git operation.

        Never persisted or logged; constructed fresh for each clone/push call.
        """
        if not token or not clone_url.startswith("https://"):
            return clone_url
        return clone_url.replace("https://", f"https://x-access-token:{token}@", 1)

    def clone_repository(
        self,
        clone_url: str,
        dest_dir: Path,
        workspace_root: Path,
        token: Optional[str] = None,
    ) -> Path:
        """Clone a repository into dest_dir, which must be confined under workspace_root.

        dest_dir must be server-computed (never derived from raw user input),
        mirroring the confinement pattern used by WorkspaceManager.
        """
        dest = Path(dest_dir).resolve()
        root = Path(workspace_root).resolve()
        if root != dest and root not in dest.parents:
            raise GitHubError("Refusing to clone outside the configured workspace root.")
        if dest.exists():
            raise GitHubError(f"Destination already exists: {dest}")

        dest.parent.mkdir(parents=True, exist_ok=True)
        authed_url = self._authenticated_url(clone_url, token)
        try:
            git.Repo.clone_from(authed_url, dest)
        except git.GitCommandError as e:
            raise GitHubError(f"Failed to clone repository: {_scrub(str(e), token)}") from None
        return dest

    def push_branch(
        self,
        repo_path: Path | str,
        branch_name: str,
        clone_url: str,
        token: Optional[str],
    ) -> None:
        """Push a local branch to origin using an ephemeral authenticated URL.

        A token is always required (GitHub requires write auth for pushes
        regardless of repository visibility); the token's identity must have
        push access to the target repository.
        """
        if not token:
            raise GitHubError(
                "Pushing requires GITHUB_TOKEN configured on the server "
                "(the identity must have write access to this repository)."
            )
        repo = None
        authed_url = self._authenticated_url(clone_url, token)
        try:
            repo = git.Repo(Path(repo_path).resolve())
            repo.git.push(authed_url, f"{branch_name}:{branch_name}", force=True)
        except git.GitCommandError as e:
            raise GitHubError(f"Failed to push branch '{branch_name}': {_scrub(str(e), token)}") from None
        finally:
            if repo is not None:
                try:
                    repo.close()
                except Exception:
                    pass

    def create_pull_request(
        self,
        owner: str,
        repo: str,
        head_branch: str,
        base_branch: str,
        title: str,
        body: str,
        token: Optional[str],
    ) -> dict:
        """Open a Pull Request via the GitHub REST API. Returns {"url", "number"}."""
        if not token:
            raise GitHubError(
                "Creating a Pull Request requires GITHUB_TOKEN configured on the server."
            )
        url = f"{self.api_base_url}/repos/{owner}/{repo}/pulls"
        payload = {"title": title, "head": head_branch, "base": base_branch, "body": body}
        try:
            resp = httpx.post(url, headers=self._headers(token), json=payload, timeout=self.timeout)
        except httpx.HTTPError as e:
            raise GitHubError(f"Could not reach GitHub: {_scrub(str(e), token)}") from None

        if resp.status_code not in (200, 201):
            detail = _scrub(resp.text[:500], token)
            raise GitHubError(f"GitHub rejected Pull Request creation ({resp.status_code}): {detail}")

        data = resp.json()
        return {"url": data.get("html_url", ""), "number": data.get("number")}
