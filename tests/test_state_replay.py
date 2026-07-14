"""State reduction and commit replay invariant tests."""

from __future__ import annotations

from pathlib import Path

from bookextract.models import (
    BookState,
    PageInterpretation,
    PageType,
    StructuralOpening,
    StructureKind,
    TocEntry,
    TocStatus,
)
from bookextract.state import (
    apply_assessment,
    finalize_toc_if_required,
    load_book_state_from_commits,
)
from bookextract.storage import RunStore
from tests.conftest import make_assessment


def test_finalize_toc_commits_after_non_toc_page() -> None:
    state = BookState(
        toc_status=TocStatus.COLLECTING,
        toc=[TocEntry(title="Ch1", level=1, kind=StructureKind.CHAPTER)],
    )
    assessment = make_assessment(1, PageInterpretation(page_type=PageType.BLANK, blocks=[]))
    updated = finalize_toc_if_required(state, assessment)
    assert updated.toc_status == TocStatus.COMMITTED


def test_apply_assessment_advances_chapter_index() -> None:
    state = BookState(
        toc_status=TocStatus.COMMITTED,
        toc=[TocEntry(title="Ch1", level=1, kind=StructureKind.CHAPTER)],
        next_expected_opening_index=0,
    )
    assessment = make_assessment(
        2,
        PageInterpretation(
            page_type=PageType.CHAPTER_OPENING,
            opening=StructuralOpening(
                kind=StructureKind.CHAPTER,
                title="Ch1",
                level=1,
            ),
            blocks=[],
        ),
    )
    new_state = apply_assessment(state, assessment)
    assert new_state.next_expected_opening_index == 1
    assert new_state.current_section is not None
    assert new_state.current_section.title == "Ch1"


def test_replay_matches_incremental_state(run_dir: Path) -> None:
    from bookextract.state import build_commit_payload

    store = RunStore(run_dir)
    state = BookState()
    for page_index in range(3):
        assessment = make_assessment(
            page_index,
            PageInterpretation(page_type=PageType.BLANK, blocks=[]),
        )
        state = finalize_toc_if_required(state, assessment)
        state = apply_assessment(state, assessment)
        files = build_commit_payload(
            assessment=assessment,
            context_json={},
            interpretation_json=assessment.interpretation.model_dump(mode="json"),
            provenance_json={},
        )
        store.write_commit(page_index + 1, files)

    replayed = load_book_state_from_commits(store, 3)
    assert replayed.processed_page_count == state.processed_page_count
    assert replayed.toc_status == state.toc_status
