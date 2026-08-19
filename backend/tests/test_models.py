"""Unit tests for database models."""

from app.db.models.repository import Repository
from app.db.models.code_chunk import CodeChunk
from app.db.models.task import Task
from app.db.models.interaction import Interaction


def test_repository_model_instantiation():
    """Test Repository model attributes and repr."""
    repo = Repository(
        id=1,
        name="test-repo",
        local_path="/path/to/repo",
        default_branch="main",
        status="pending",
    )
    assert repo.id == 1
    assert repo.name == "test-repo"
    assert repo.local_path == "/path/to/repo"
    assert repr(repo) == "<Repository id=1 name='test-repo' status='pending'>"


def test_code_chunk_model_instantiation():
    """Test CodeChunk model attributes, vector embedding, and metadata."""
    chunk = CodeChunk(
        id=10,
        repository_id=1,
        file_path="src/calculator.py",
        language="python",
        symbol_name="add",
        symbol_type="function",
        start_line=1,
        end_line=5,
        source_code="def add(a, b):\n    return a + b\n",
        content_hash="abc123hash",
        embedding=[0.1] * 3072,
        chunk_metadata={"docstring": None},
    )
    assert chunk.id == 10
    assert chunk.repository_id == 1
    assert chunk.file_path == "src/calculator.py"
    assert chunk.symbol_name == "add"
    assert len(chunk.embedding) == 3072
    assert repr(chunk) == "<CodeChunk id=10 file='src/calculator.py' symbol='add' lines=1-5>"


def test_task_and_interaction_models():
    """Test Task and Interaction models."""
    task = Task(
        id=100,
        repository_id=1,
        title="Fix calculation bug",
        description="Fix edge case in division by zero",
        status="investigating",
        attempts=1,
    )
    assert task.id == 100
    assert task.attempts == 1
    assert "Fix calculation bug" in repr(task)

    interaction = Interaction(
        id=500,
        task_id=100,
        role="assistant",
        content="I have investigated the code.",
        tool_name="read_file",
        tool_args={"file_path": "src/calculator.py"},
        tool_result={"success": True},
    )
    assert interaction.id == 500
    assert interaction.task_id == 100
    assert interaction.role == "assistant"
    assert repr(interaction) == "<Interaction id=500 task_id=100 role='assistant'>"
