"""Transactional output publication and crash recovery."""

from __future__ import annotations

import re
import secrets
import shutil
from collections.abc import Callable
from pathlib import Path
from typing import Literal

from bookextract.config import write_json_atomic
from bookextract.errors import ProcessingError
from bookextract.models import OutputTransaction
from bookextract.output_paths import validate_output_tree
from bookextract.storage import RunStore

OutputCommand = Literal["markdown", "epub"]

_CANDIDATE_RE = re.compile(r"^(markdown|epub)\.candidate\.[0-9a-f]{16}$")
_PREVIOUS_RE = re.compile(r"^(markdown|epub)\.previous$")
_WORK_RE = re.compile(r"^(markdown|epub)\.work\.[0-9a-f]{16}$")


def output_destination(store: RunStore, command: OutputCommand) -> Path:
    return store._path("output", command)


def output_build_dir(store: RunStore) -> Path:
    path = store._path(".output-build")
    path.mkdir(parents=True, exist_ok=True)
    return path


def _transaction_marker_path(store: RunStore, command: OutputCommand) -> Path:
    return output_build_dir(store) / f"{command}.transaction.json"


def _read_transaction(store: RunStore, command: OutputCommand) -> OutputTransaction | None:
    path = _transaction_marker_path(store, command)
    if not path.is_file():
        return None
    return OutputTransaction.model_validate_json(path.read_text(encoding="utf-8"))


def _write_transaction(store: RunStore, transaction: OutputTransaction) -> None:
    write_json_atomic(
        _transaction_marker_path(store, transaction.command),
        transaction.model_dump(mode="json"),
    )


def _remove_transaction(store: RunStore, command: OutputCommand) -> None:
    path = _transaction_marker_path(store, command)
    if path.exists():
        path.unlink()


def _validate_transaction_basename(name: str, command: OutputCommand) -> None:
    if not name or "/" in name or "\\" in name or ".." in name:
        raise ProcessingError(
            code="invalid-run-layout",
            message=f"invalid transaction basename: {name!r}",
        )
    if not (
        _CANDIDATE_RE.match(name)
        or _PREVIOUS_RE.match(name)
        or _WORK_RE.match(name)
    ):
        raise ProcessingError(
            code="invalid-run-layout",
            message=f"unexpected transaction basename: {name!r}",
        )
    if not name.startswith(f"{command}."):
        raise ProcessingError(
            code="invalid-run-layout",
            message=f"transaction basename command mismatch: {name!r}",
        )


def _resolve_build_path(store: RunStore, basename: str, command: OutputCommand) -> Path:
    _validate_transaction_basename(basename, command)
    path = output_build_dir(store) / basename
    if path.is_symlink():
        raise ProcessingError(
            code="invalid-run-layout",
            message=f"transaction path is a symlink: {basename}",
        )
    return path


def _quarantine_path(store: RunStore, src: Path, label: str) -> None:
    if not src.exists():
        return
    recovery_dir = store._next_recovery_dir()
    store._quarantine(recovery_dir, src, label)


def _remove_tree(path: Path) -> None:
    if path.is_symlink():
        path.unlink()
    elif path.is_dir():
        shutil.rmtree(path)
    elif path.exists():
        path.unlink()


def _new_nonce() -> str:
    return secrets.token_hex(8)


def _candidate_name(command: OutputCommand) -> str:
    return f"{command}.candidate.{_new_nonce()}"


def _previous_name(command: OutputCommand) -> str:
    return f"{command}.previous"


def _work_name(command: OutputCommand) -> str:
    return f"{command}.work.{_new_nonce()}"


def _tree_exists(path: Path) -> bool:
    return path.is_dir() and not path.is_symlink()


def _validate_tree(
    path: Path,
    *,
    command: OutputCommand,
    expected_committed_page_count: int | None,
) -> bool:
    if not _tree_exists(path):
        return False
    try:
        validate_output_tree(
            path,
            expected_command=command,
            expected_committed_page_count=expected_committed_page_count,
        )
    except ProcessingError:
        return False
    return True


def _restore_previous(
    store: RunStore,
    command: OutputCommand,
    *,
    previous_path: Path,
    destination: Path,
) -> None:
    if destination.exists():
        _remove_tree(destination)
    shutil.move(str(previous_path), str(destination))


def _publish_candidate(
    store: RunStore,
    command: OutputCommand,
    *,
    candidate_path: Path,
    destination: Path,
) -> None:
    if destination.exists():
        _remove_tree(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.move(str(candidate_path), str(destination))


def _finish_recovery(
    store: RunStore,
    command: OutputCommand,
    *,
    destination: Path,
    candidate_path: Path | None,
    previous_path: Path | None,
    work_path: Path | None,
) -> None:
    if work_path is not None and work_path.exists():
        _remove_tree(work_path)
    if candidate_path is not None and candidate_path.exists():
        _remove_tree(candidate_path)
    if previous_path is not None and previous_path.exists():
        _remove_tree(previous_path)
    _remove_transaction(store, command)
    if destination.exists() and destination.is_symlink():
        raise ProcessingError(
            code="invalid-run-layout",
            message=f"destination symlink rejected: {destination}",
        )


def _recover_candidate_published(
    store: RunStore,
    command: OutputCommand,
    *,
    transaction: OutputTransaction,
    destination: Path,
    head: int,
) -> None:
    previous_path = _resolve_build_path(store, transaction.previous, command)
    candidate_path = _resolve_build_path(store, transaction.candidate, command)
    work_path = (
        _resolve_build_path(store, transaction.work, command)
        if transaction.work is not None
        else None
    )

    if _validate_tree(destination, command=command, expected_committed_page_count=head):
        _finish_recovery(
            store,
            command,
            destination=destination,
            candidate_path=candidate_path,
            previous_path=previous_path,
            work_path=work_path,
        )
        return

    if _validate_tree(previous_path, command=command, expected_committed_page_count=None):
        _quarantine_path(store, destination, f"{command}-destination-invalid")
        _restore_previous(store, command, previous_path=previous_path, destination=destination)
        _finish_recovery(
            store,
            command,
            destination=destination,
            candidate_path=candidate_path,
            previous_path=None,
            work_path=work_path,
        )
        return

    _quarantine_path(store, destination, f"{command}-destination-stale")
    if previous_path.exists():
        _quarantine_path(store, previous_path, f"{command}-previous-invalid")
    if candidate_path.exists():
        _quarantine_path(store, candidate_path, f"{command}-candidate-remnant")
    if work_path is not None and work_path.exists():
        _remove_tree(work_path)
    _remove_transaction(store, command)
    raise ProcessingError(
        code="output-transaction-corruption",
        message=f"unrecoverable {command} output transaction after candidate-published",
    )


def _recover_with_marker(
    store: RunStore,
    command: OutputCommand,
    *,
    transaction: OutputTransaction,
    destination: Path,
    head: int,
) -> None:
    candidate_path = _resolve_build_path(store, transaction.candidate, command)
    previous_path = _resolve_build_path(store, transaction.previous, command)
    work_path = (
        _resolve_build_path(store, transaction.work, command)
        if transaction.work is not None
        else None
    )
    dest_exists = destination.exists()
    prev_exists = previous_path.exists()
    cand_exists = candidate_path.exists()

    if transaction.phase == "building":
        if prev_exists:
            _quarantine_path(store, candidate_path, f"{command}-candidate-unexpected")
            if work_path is not None and work_path.exists():
                _remove_tree(work_path)
            _remove_transaction(store, command)
            raise ProcessingError(
                code="output-transaction-corruption",
                message=f"unexpected previous during {command} building phase",
            )
        if cand_exists:
            _quarantine_path(store, candidate_path, f"{command}-candidate-incomplete")
        if work_path is not None and work_path.exists():
            _remove_tree(work_path)
        _remove_transaction(store, command)
        return

    if transaction.phase == "candidate-valid":
        candidate_valid = cand_exists and _validate_tree(
            candidate_path, command=command, expected_committed_page_count=head
        )
        if dest_exists and cand_exists and not prev_exists:
            if not candidate_valid:
                _quarantine_path(store, candidate_path, f"{command}-candidate-stale")
                if _validate_tree(
                    destination, command=command, expected_committed_page_count=None
                ):
                    _finish_recovery(
                        store,
                        command,
                        destination=destination,
                        candidate_path=None,
                        previous_path=None,
                        work_path=work_path,
                    )
                    return
                _quarantine_path(store, destination, f"{command}-destination-invalid")
                _remove_transaction(store, command)
                return
            previous_name = _previous_name(command)
            shutil.move(str(destination), str(previous_path))
            _write_transaction(
                store,
                transaction.model_copy(
                    update={"phase": "previous-moved", "previous": previous_name}
                ),
            )
            _publish_candidate(
                store,
                command,
                candidate_path=candidate_path,
                destination=destination,
            )
            _write_transaction(
                store,
                transaction.model_copy(
                    update={
                        "phase": "candidate-published",
                        "previous": previous_name,
                    }
                ),
            )
            _finish_recovery(
                store,
                command,
                destination=destination,
                candidate_path=None,
                previous_path=previous_path if previous_path.exists() else None,
                work_path=work_path,
            )
            return
        if dest_exists and not cand_exists and not prev_exists:
            if _validate_tree(
                destination, command=command, expected_committed_page_count=head
            ):
                _finish_recovery(
                    store,
                    command,
                    destination=destination,
                    candidate_path=None,
                    previous_path=None,
                    work_path=work_path,
                )
                return
            _quarantine_path(store, destination, f"{command}-destination-stale")
            _remove_transaction(store, command)
            return
        if not dest_exists and cand_exists and prev_exists:
            _publish_candidate(
                store,
                command,
                candidate_path=candidate_path,
                destination=destination,
            )
            _finish_recovery(
                store,
                command,
                destination=destination,
                candidate_path=None,
                previous_path=previous_path,
                work_path=work_path,
            )
            return
        if not dest_exists and not cand_exists and prev_exists:
            _restore_previous(
                store,
                command,
                previous_path=previous_path,
                destination=destination,
            )
            _finish_recovery(
                store,
                command,
                destination=destination,
                candidate_path=None,
                previous_path=None,
                work_path=work_path,
            )
            return
        if not dest_exists and cand_exists and not prev_exists:
            if not candidate_valid:
                _quarantine_path(store, candidate_path, f"{command}-candidate-stale")
                _remove_transaction(store, command)
                return
            _publish_candidate(
                store,
                command,
                candidate_path=candidate_path,
                destination=destination,
            )
            _finish_recovery(
                store,
                command,
                destination=destination,
                candidate_path=None,
                previous_path=None,
                work_path=work_path,
            )
            return

    if transaction.phase == "previous-moved":
        candidate_valid = cand_exists and _validate_tree(
            candidate_path, command=command, expected_committed_page_count=head
        )
        if not dest_exists and cand_exists and prev_exists:
            if not candidate_valid:
                _quarantine_path(store, candidate_path, f"{command}-candidate-stale")
                if _validate_tree(
                    previous_path, command=command, expected_committed_page_count=None
                ):
                    _restore_previous(
                        store,
                        command,
                        previous_path=previous_path,
                        destination=destination,
                    )
                _finish_recovery(
                    store,
                    command,
                    destination=destination,
                    candidate_path=None,
                    previous_path=None,
                    work_path=work_path,
                )
                return
            _publish_candidate(
                store,
                command,
                candidate_path=candidate_path,
                destination=destination,
            )
            _finish_recovery(
                store,
                command,
                destination=destination,
                candidate_path=None,
                previous_path=previous_path,
                work_path=work_path,
            )
            return
        if dest_exists and not cand_exists and prev_exists:
            if _validate_tree(
                destination, command=command, expected_committed_page_count=head
            ):
                _finish_recovery(
                    store,
                    command,
                    destination=destination,
                    candidate_path=None,
                    previous_path=previous_path,
                    work_path=work_path,
                )
                return
            if _validate_tree(
                previous_path, command=command, expected_committed_page_count=None
            ):
                _quarantine_path(store, destination, f"{command}-destination-stale")
                _restore_previous(
                    store,
                    command,
                    previous_path=previous_path,
                    destination=destination,
                )
                _finish_recovery(
                    store,
                    command,
                    destination=destination,
                    candidate_path=None,
                    previous_path=None,
                    work_path=work_path,
                )
                return
            _remove_transaction(store, command)
            return
        if not dest_exists and not cand_exists and prev_exists:
            _restore_previous(
                store,
                command,
                previous_path=previous_path,
                destination=destination,
            )
            _finish_recovery(
                store,
                command,
                destination=destination,
                candidate_path=None,
                previous_path=None,
                work_path=work_path,
            )
            return

    if transaction.phase == "candidate-published":
        _recover_candidate_published(
            store,
            command,
            transaction=transaction,
            destination=destination,
            head=head,
        )
        return

    raise ProcessingError(
        code="output-transaction-corruption",
        message=f"unhandled {command} transaction phase {transaction.phase!r}",
    )


def _recover_no_marker(store: RunStore, command: OutputCommand, *, head: int) -> None:
    build_dir = output_build_dir(store)
    destination = output_destination(store, command)
    previous_path = build_dir / _previous_name(command)

    for entry in build_dir.iterdir():
        name = entry.name
        if _CANDIDATE_RE.match(name) and name.startswith(f"{command}."):
            _quarantine_path(store, entry, name)
        elif _WORK_RE.match(name) and name.startswith(f"{command}."):
            _quarantine_path(store, entry, name)

    if destination.exists():
        if _validate_tree(destination, command=command, expected_committed_page_count=None):
            if previous_path.exists():
                _quarantine_path(store, previous_path, f"{command}-previous-stale")
        elif previous_path.exists() and _validate_tree(
            previous_path, command=command, expected_committed_page_count=None
        ):
            _quarantine_path(store, destination, f"{command}-destination-invalid")
            _restore_previous(
                store,
                command,
                previous_path=previous_path,
                destination=destination,
            )
        else:
            _quarantine_path(store, destination, f"{command}-destination-invalid")
    elif previous_path.exists():
        _restore_previous(
            store,
            command,
            previous_path=previous_path,
            destination=destination,
        )


def recover_output_transaction(store: RunStore, command: OutputCommand) -> None:
    head = store.read_head().committed_page_count
    destination = output_destination(store, command)
    transaction = _read_transaction(store, command)
    if transaction is None:
        _recover_no_marker(store, command, head=head)
        return
    if transaction.command != command:
        raise ProcessingError(
            code="output-transaction-corruption",
            message="transaction command mismatch",
        )
    _recover_with_marker(
        store,
        command,
        transaction=transaction,
        destination=destination,
        head=head,
    )


def quarantine_invalid_output_destinations(store: RunStore) -> None:
    """Quarantine invalid published output during process without blocking commits."""
    for command in ("markdown", "epub"):
        destination = output_destination(store, command)
        if not destination.exists():
            continue
        if not _validate_tree(
            destination,
            command=command,
            expected_committed_page_count=None,
        ):
            _quarantine_path(store, destination, f"{command}-destination-invalid")


def begin_output_transaction(
    store: RunStore,
    command: OutputCommand,
    *,
    work: bool = False,
) -> tuple[Path, OutputTransaction]:
    build_dir = output_build_dir(store)
    candidate_name = _candidate_name(command)
    candidate_path = build_dir / candidate_name
    if candidate_path.exists():
        shutil.rmtree(candidate_path)
    candidate_path.mkdir(parents=True)

    work_name: str | None = None
    work_path: Path | None = None
    if work:
        work_name = _work_name(command)
        work_path = build_dir / work_name
        if work_path.exists():
            shutil.rmtree(work_path)
        work_path.mkdir(parents=True)

    transaction = OutputTransaction(
        output_transaction_format_version=1,
        command=command,
        phase="building",
        candidate=candidate_name,
        previous=_previous_name(command),
        work=work_name,
    )
    _write_transaction(store, transaction)
    return candidate_path, transaction


def mark_candidate_valid(store: RunStore, transaction: OutputTransaction) -> None:
    _write_transaction(
        store,
        transaction.model_copy(update={"phase": "candidate-valid"}),
    )


def publish_candidate_tree(
    store: RunStore,
    command: OutputCommand,
    *,
    transaction: OutputTransaction,
    candidate_path: Path,
) -> Path:
    destination = output_destination(store, command)
    head = store.read_head().committed_page_count
    if not _validate_tree(
        candidate_path,
        command=command,
        expected_committed_page_count=head,
    ):
        raise ProcessingError(
            code="invalid-run-layout",
            message="candidate failed validation before publication",
        )

    mark_candidate_valid(store, transaction)
    previous_path = _resolve_build_path(store, transaction.previous, command)
    work_path = (
        _resolve_build_path(store, transaction.work, command)
        if transaction.work is not None
        else None
    )

    if destination.exists():
        if previous_path.exists():
            _remove_tree(previous_path)
        shutil.move(str(destination), str(previous_path))
        _write_transaction(
            store,
            transaction.model_copy(update={"phase": "previous-moved"}),
        )

    _publish_candidate(
        store,
        command,
        candidate_path=candidate_path,
        destination=destination,
    )
    _write_transaction(
        store,
        transaction.model_copy(update={"phase": "candidate-published"}),
    )

    if previous_path.exists():
        _remove_tree(previous_path)
    if work_path is not None and work_path.exists():
        _remove_tree(work_path)
    _remove_transaction(store, command)
    return destination


def publish_output(
    store: RunStore,
    command: OutputCommand,
    builder: Callable[[Path, Path | None], None],
    *,
    use_work_directory: bool = False,
) -> Path:
    recover_output_transaction(store, command)
    candidate_path, transaction = begin_output_transaction(
        store,
        command,
        work=use_work_directory,
    )
    work_path = (
        _resolve_build_path(store, transaction.work, command)
        if transaction.work is not None
        else None
    )
    try:
        builder(candidate_path, work_path)
        return publish_candidate_tree(
            store,
            command,
            transaction=transaction,
            candidate_path=candidate_path,
        )
    except BaseException:
        if candidate_path.exists():
            _quarantine_path(store, candidate_path, f"{command}-candidate-failed")
        if work_path is not None and work_path.exists():
            _remove_tree(work_path)
        _remove_transaction(store, command)
        raise
