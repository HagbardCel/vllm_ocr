# Lean Implementation Plan: Scanned Book PDF to Structured Markdown/EPUB

## 1. Objective

Build a lean, local-first tool that converts scanned books from PDF into structured outputs such as Markdown and EPUB.

The first version should prioritize:

- clarity;
- simplicity;
- maintainability;
- extensibility;
- deterministic behavior after each page assessment;
- explicit failure when later content contradicts previously accepted structure.

The tool should use a local vision-language model served through `llama.cpp`, likely a Gemma or Qwen 3.6 multimodal model.

The first version may support only a subset of books. Unsupported or contradictory structures should fail explicitly rather than be handled through complex reconciliation or backtracking.

---

## 2. High-level architecture

The first version should behave as a sequential, stateful, fail-fast book compiler:

```text
PDF
  ↓
render next page
  ↓
construct compact authoritative context
  ↓
local VLM produces structured PageInterpretation
  ↓
Pydantic validation
  ↓
deterministic structural validation
  ├── mismatch → save diagnostic and stop
  └── valid → commit page and update BookState
  ↓
checkpoint
  ↓
next page
  ↓
Markdown
  ↓
Pandoc → EPUB
```

The initially supported book class should be deliberately constrained:

- predominantly single-column prose;
- recognizable front matter and table of contents;
- conventional chapter and section hierarchy;
- same-page footnotes;
- full-page or clearly bounded illustrations;
- no complicated tables, marginalia, parallel texts, or irregular scholarly apparatus.

---

## 3. Core architectural decisions

### 3.1 Authoritative sequential state

Every accepted page assessment becomes authoritative.

Later pages may use it to disambiguate content, but may not silently contradict it.

A contradiction produces a structured error such as:

```text
StructuralMismatchError
```

The first version should not include:

- alternative hypotheses;
- confidence aggregation;
- automatic backtracking;
- multi-parser reconciliation;
- dependency graphs;
- databases;
- review interfaces.

### 3.2 One VLM call per page

Start with a single VLM call per page that performs:

- page-type classification;
- transcription;
- block identification;
- heading-level assignment;
- table-of-contents extraction;
- footnote extraction;
- image and caption identification;
- paragraph-continuation detection.

This is not necessarily the final architecture, but it provides the clearest baseline.

The `PageInterpreter` abstraction should later allow replacing this with:

- two-stage classify-then-extract processing;
- Docling;
- Marker;
- OCR plus a text-only LLM;
- multiple parsers with reconciliation.

### 3.3 Structured output instead of Markdown output

The VLM should return a typed `PageInterpretation`.

Markdown should be generated later by deterministic application code.

The model should not be responsible for:

- Markdown syntax;
- EPUB semantics;
- asset paths;
- cross-page state mutation;
- final document rendering.

### 3.4 Context as a projection

The VLM should not receive the complete prior transcription.

Instead, it should receive a compact `PageContext` containing only information relevant to the current structural decision.

---

## 4. Recommended technology stack

| Concern | Choice |
|---|---|
| Language | Python 3.12+ |
| Validation and models | Pydantic v2 |
| HTTP | `httpx` |
| PDF rendering | `pypdfium2` |
| Image processing | Pillow |
| CLI | Typer |
| Testing | pytest |
| Packaging | `uv` |
| VLM serving | `llama-server` |
| Markdown to EPUB | Pandoc |
| Persistence | JSON files and image assets |

Avoid initially:

- LangChain;
- workflow engines;
- SQLAlchemy;
- FastAPI;
- asynchronous job queues;
- plugin-discovery frameworks;
- graph databases.

Direct HTTP calls to `llama-server` are easier to inspect and debug.

---

## 5. Package structure

Use a moderately flat package structure:

```text
bookextract/
├── __init__.py
├── cli.py
├── config.py
├── models.py
├── errors.py
├── pdf.py
├── context.py
├── validation.py
├── state.py
├── pipeline.py
├── storage.py
│
├── interpretation/
│   ├── base.py
│   ├── vlm.py
│   └── prompts.py
│
├── inference/
│   ├── base.py
│   └── llamacpp.py
│
└── rendering/
    ├── markdown.py
    ├── assets.py
    └── epub.py

tests/
├── unit/
│   ├── test_state.py
│   ├── test_validation.py
│   ├── test_context.py
│   └── test_markdown.py
├── contract/
│   └── test_llamacpp_client.py
├── golden/
│   ├── sample_book/
│   └── test_golden_pages.py
└── integration/
    └── test_local_model.py
```

Two abstraction boundaries matter most:

1. `VisionModelClient`: communicates with a model server.
2. `PageInterpreter`: converts a page and context into a structured page interpretation.

Everything else can initially remain concrete and direct.

---

## 6. Domain model

### 6.1 Base model

```python
from __future__ import annotations

from enum import StrEnum
from pathlib import Path
from typing import Annotated, Literal, Protocol, TypeVar

from pydantic import BaseModel, ConfigDict, Field


class DomainModel(BaseModel):
    model_config = ConfigDict(
        extra="forbid",
        validate_assignment=True,
    )
```

`extra="forbid"` is important because unexpected fields or schema drift should fail visibly.

### 6.2 Page and block types

```python
class PageType(StrEnum):
    COVER = "cover"
    TITLE = "title"
    COPYRIGHT = "copyright"
    DEDICATION = "dedication"
    TOC = "toc"
    PREFACE = "preface"
    CHAPTER_OPENING = "chapter_opening"
    BODY = "body"
    FIGURE_PAGE = "figure_page"
    APPENDIX = "appendix"
    BIBLIOGRAPHY = "bibliography"
    INDEX = "index"
    BLANK = "blank"
    OTHER = "other"


class TextRole(StrEnum):
    BOOK_TITLE = "book_title"
    SUBTITLE = "subtitle"
    AUTHOR = "author"
    CHAPTER_TITLE = "chapter_title"
    HEADING = "heading"
    PARAGRAPH = "paragraph"
    BLOCKQUOTE = "blockquote"
    EPIGRAPH = "epigraph"
    LIST_ITEM = "list_item"
    CAPTION = "caption"
    RUNNING_HEADER = "running_header"
    RUNNING_FOOTER = "running_footer"
    PAGE_NUMBER = "page_number"
    OTHER = "other"
```

A closed enum is appropriate for version 0.1.

Namespaced strings can be introduced later when multiple parser families are supported.

### 6.3 Optional geometry and presentation

Geometry and style are useful, but should not be mandatory:

```python
class BoundingBox(DomainModel):
    # Normalized coordinates in the range [0, 1].
    x0: float = Field(ge=0.0, le=1.0)
    y0: float = Field(ge=0.0, le=1.0)
    x1: float = Field(ge=0.0, le=1.0)
    y1: float = Field(ge=0.0, le=1.0)


class RelativeSize(StrEnum):
    SMALL = "small"
    BODY = "body"
    LARGE = "large"
    EXTRA_LARGE = "extra_large"


class Alignment(StrEnum):
    LEFT = "left"
    CENTER = "center"
    RIGHT = "right"
    JUSTIFIED = "justified"
    UNKNOWN = "unknown"


class TextStyle(DomainModel):
    font_family: str | None = None
    font_size_pt: float | None = None
    relative_size: RelativeSize | None = None

    bold: bool | None = None
    italic: bool | None = None
    small_caps: bool | None = None
    alignment: Alignment | None = None
```

A VLM will probably populate qualitative attributes such as `relative_size`, boldness, and alignment more reliably than exact point sizes.

A future layout-based backend can provide exact measurements.

### 6.4 Inline text and footnote references

Use structured inline content instead of embedding Markdown syntax:

```python
class TextRun(DomainModel):
    kind: Literal["text"] = "text"
    text: str


class FootnoteReference(DomainModel):
    kind: Literal["footnote_reference"] = "footnote_reference"
    label: str


InlineContent = Annotated[
    TextRun | FootnoteReference,
    Field(discriminator="kind"),
]
```

Example:

```json
[
  {"kind": "text", "text": "The proposal was controversial"},
  {"kind": "footnote_reference", "label": "7"},
  {"kind": "text", "text": "."}
]
```

This avoids character-offset problems and keeps Markdown out of the domain model.

### 6.5 Content blocks

```python
class TextBlock(DomainModel):
    kind: Literal["text"] = "text"
    role: TextRole
    content: list[InlineContent]

    heading_level: int | None = Field(default=None, ge=1, le=6)

    continues_previous: bool = False
    continues_on_next_page: bool = False

    bbox: BoundingBox | None = None
    style: TextStyle | None = None


class FootnoteBlock(DomainModel):
    kind: Literal["footnote"] = "footnote"
    label: str
    content: list[InlineContent]

    bbox: BoundingBox | None = None
    style: TextStyle | None = None


class FigureBlock(DomainModel):
    kind: Literal["figure"] = "figure"

    label: str | None = None
    caption: str | None = None

    bbox: BoundingBox | None = None

    # Assigned after the model response.
    asset_path: Path | None = None


class SeparatorBlock(DomainModel):
    kind: Literal["separator"] = "separator"


ContentBlock = Annotated[
    TextBlock | FootnoteBlock | FigureBlock | SeparatorBlock,
    Field(discriminator="kind"),
]
```

Do not ask the model to generate block IDs.

Assign deterministic IDs after validation:

```text
p001-b001
p001-b002
p002-b001
```

### 6.6 Table of contents and metadata

```python
class TocEntry(DomainModel):
    title: str
    level: int = Field(ge=1, le=6)
    printed_page_label: str | None = None


class ExtractedMetadata(DomainModel):
    title: str | None = None
    subtitle: str | None = None
    authors: list[str] = Field(default_factory=list)
    language: str | None = None
```

### 6.7 VLM response

Separate model-generated information from application-generated metadata:

```python
class PageInterpretation(DomainModel):
    page_type: PageType
    printed_page_label: str | None = None

    metadata: ExtractedMetadata | None = None
    toc_entries: list[TocEntry] = Field(default_factory=list)
    blocks: list[ContentBlock] = Field(default_factory=list)

    warnings: list[str] = Field(default_factory=list)
```

The model should not generate:

- PDF page index;
- image path;
- model name;
- token counts;
- timestamps;
- asset filenames.

These are application facts.

```python
class ModelRunInfo(DomainModel):
    backend: str
    model: str
    prompt_version: str

    prompt_tokens: int | None = None
    completion_tokens: int | None = None
    elapsed_ms: float | None = None


class PageAssessment(DomainModel):
    page_index: int
    image_path: Path

    interpretation: PageInterpretation
    model_run: ModelRunInfo
```

---

## 7. Authoritative state

### 7.1 Chapter state

```python
class HeadingSummary(DomainModel):
    level: int
    text: str
    page_index: int


class ChapterState(DomainModel):
    title: str
    toc_index: int | None = None
    started_on_page: int

    headings: list[HeadingSummary] = Field(default_factory=list)

    open_paragraph_tail: str | None = None
```

Do not store the full chapter text in `BookState`.

Accepted page assessments already contain it.

### 7.2 TOC lifecycle

```python
class TocStatus(StrEnum):
    NOT_SEEN = "not_seen"
    COLLECTING = "collecting"
    COMMITTED = "committed"
    ABSENT = "absent"
```

### 7.3 Book state

```python
class BookState(DomainModel):
    metadata: ExtractedMetadata = Field(default_factory=ExtractedMetadata)

    toc_status: TocStatus = TocStatus.NOT_SEEN
    toc: list[TocEntry] = Field(default_factory=list)
    next_expected_toc_index: int = 0

    current_chapter: ChapterState | None = None

    previous_page_type: PageType | None = None
    processed_page_count: int = 0
```

This state should remain small and contain only information required for later decisions.

---

## 8. Context passed to the VLM

```python
class PageContext(DomainModel):
    book_title: str | None = None
    authors: list[str] = Field(default_factory=list)

    toc_status: TocStatus
    expected_toc_entries: list[TocEntry] = Field(default_factory=list)

    current_chapter_title: str | None = None
    current_chapter_headings: list[HeadingSummary] = Field(
        default_factory=list
    )

    open_paragraph_tail: str | None = None
    previous_page_type: PageType | None = None
```

The context builder should be a pure function:

```python
def build_page_context(state: BookState) -> PageContext:
    expected = []

    if state.toc_status == TocStatus.COMMITTED:
        start = state.next_expected_toc_index
        expected = state.toc[start : start + 3]

    chapter = state.current_chapter

    return PageContext(
        book_title=state.metadata.title,
        authors=state.metadata.authors,
        toc_status=state.toc_status,
        expected_toc_entries=expected,
        current_chapter_title=chapter.title if chapter else None,
        current_chapter_headings=(
            chapter.headings[-12:] if chapter else []
        ),
        open_paragraph_tail=(
            chapter.open_paragraph_tail if chapter else None
        ),
        previous_page_type=state.previous_page_type,
    )
```

The model receives:

- the next three expected TOC entries;
- recent chapter headings;
- only the tail of an open paragraph;
- authoritative metadata;
- the prior page type.

---

## 9. Key interfaces

### 9.1 Vision model client

```python
ResponseT = TypeVar("ResponseT", bound=DomainModel)


class VisionModelClient(Protocol):
    async def generate_structured(
        self,
        *,
        image_path: Path,
        prompt: str,
        response_model: type[ResponseT],
    ) -> tuple[ResponseT, ModelRunInfo]:
        ...
```

This interface knows nothing about books.

Possible future implementations:

- `LlamaCppVisionClient`;
- `MlxVisionClient`;
- remote OpenAI-compatible client;
- deterministic fixture client for tests.

### 9.2 Page interpreter

```python
class PageInterpreter(Protocol):
    async def interpret(
        self,
        *,
        image_path: Path,
        context: PageContext,
    ) -> tuple[PageInterpretation, ModelRunInfo]:
        ...
```

Initial implementation:

```python
class VlmPageInterpreter:
    def __init__(
        self,
        client: VisionModelClient,
        prompt_builder: PagePromptBuilder,
    ) -> None:
        self._client = client
        self._prompt_builder = prompt_builder

    async def interpret(
        self,
        *,
        image_path: Path,
        context: PageContext,
    ) -> tuple[PageInterpretation, ModelRunInfo]:
        prompt = self._prompt_builder.build(context)

        return await self._client.generate_structured(
            image_path=image_path,
            prompt=prompt,
            response_model=PageInterpretation,
        )
```

A future `DoclingPageInterpreter` can implement the same interface without using a VLM.

---

## 10. llama.cpp integration

Use `llama-server` through its OpenAI-compatible chat-completions endpoint.

For multimodal input, send:

- a text prompt;
- a local `file://` image URL;
- a JSON schema generated from the Pydantic response model.

### 10.1 Illustrative server command

```bash
llama-server \
  -m ~/data/models/book-vlm/model.gguf \
  --mmproj ~/data/models/book-vlm/mmproj.gguf \
  --alias book-vlm \
  -c 32768 \
  --media-path /absolute/path/to/book-run \
  --host 127.0.0.1 \
  --port 8080
```

The exact Qwen or Gemma model should be treated as configuration.

The architecture should not depend on a particular model family.

### 10.2 Client implementation

```python
import time
from pathlib import Path
from typing import TypeVar

import httpx
from pydantic import BaseModel


T = TypeVar("T", bound=BaseModel)


class LlamaCppVisionClient:
    def __init__(
        self,
        *,
        base_url: str,
        model: str,
        timeout_seconds: float,
        prompt_version: str,
        disable_thinking: bool = True,
    ) -> None:
        self._base_url = base_url.rstrip("/")
        self._model = model
        self._timeout = timeout_seconds
        self._prompt_version = prompt_version
        self._disable_thinking = disable_thinking

    async def generate_structured(
        self,
        *,
        image_path: Path,
        prompt: str,
        response_model: type[T],
    ) -> tuple[T, ModelRunInfo]:
        schema = response_model.model_json_schema()

        payload = {
            "model": self._model,
            "temperature": 0,
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": prompt,
                        },
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": image_path.resolve().as_uri(),
                            },
                        },
                    ],
                }
            ],
            "response_format": {
                "type": "json_schema",
                "schema": schema,
            },
        }

        if self._disable_thinking:
            payload["chat_template_kwargs"] = {
                "enable_thinking": False,
            }

        started = time.monotonic()

        async with httpx.AsyncClient(timeout=self._timeout) as client:
            response = await client.post(
                f"{self._base_url}/v1/chat/completions",
                json=payload,
            )
            response.raise_for_status()

        elapsed_ms = (time.monotonic() - started) * 1000
        body = response.json()

        raw_content = body["choices"][0]["message"]["content"]
        result = response_model.model_validate_json(raw_content)

        usage = body.get("usage", {})

        return result, ModelRunInfo(
            backend="llama.cpp",
            model=self._model,
            prompt_version=self._prompt_version,
            prompt_tokens=usage.get("prompt_tokens"),
            completion_tokens=usage.get("completion_tokens"),
            elapsed_ms=elapsed_ms,
        )
```

Pin the `llama.cpp` build used for a run.

Keep `enable_thinking` configurable because model-template behavior can vary.

---

## 11. Prompt design

Use one stable prompt template with a version identifier.

### 11.1 Prompt sections

```text
1. Role and fidelity requirements
2. Authoritative book context
3. Page classification rules
4. Block extraction rules
5. Footnote rules
6. Figure rules
7. Continuation rules
8. Output field semantics
```

### 11.2 Faithful transcription

```text
Transcribe the visible text faithfully.
Do not modernize spelling.
Do not improve grammar.
Do not silently complete damaged text.
Do not insert text merely because it is expected from the context.
```

### 11.3 Context usage

```text
The supplied book context contains previously accepted facts.
Use it to disambiguate headings and chapter transitions.

However, report what is visibly present on the current page.
Do not alter observed text merely to make it agree with the context.
A later deterministic validator will detect contradictions.
```

This prevents the VLM from hallucinating the expected chapter title.

### 11.4 Page classification

The prompt should define each page type explicitly.

Important distinctions include:

- `chapter_opening`: contains the beginning of a new chapter;
- `body`: ordinary content within the current chapter;
- `figure_page`: predominantly an illustration or plate;
- `toc`: actual table-of-contents entries, not a list of figures or index.

### 11.5 Block ordering

```text
Return blocks in normal human reading order.
Exclude running headers, running footers and printed page numbers from
ordinary paragraph content, but return them using their dedicated roles.
```

### 11.6 Heading levels

```text
Use level 1 for chapter titles.
Use level 2 for the highest heading level within a chapter.
Do not skip a level unless the visible book structure clearly requires it.
Use the TOC and prior headings as evidence.
```

### 11.7 Footnotes

```text
Represent each inline footnote marker as a footnote_reference segment.
Represent the corresponding note body as a footnote block.
Preserve the printed label exactly.
```

### 11.8 Figures

```text
Return a figure block for a meaningful illustration, photograph,
diagram, map or plate.

Provide a normalized bounding box when the figure occupies only part
of the page. A bounding box is optional for a full-page plate.
Do not generate a caption when none is printed.
```

---

## 12. Sequential processing algorithm

```python
async def process_book(
    *,
    source: PdfPageSource,
    interpreter: PageInterpreter,
    store: RunStore,
    config: ProcessingConfig,
) -> BookDocument:
    state, document = store.load_or_initialize()

    for page_index in range(state.processed_page_count, source.page_count):
        image_path = source.render_page(
            page_index,
            dpi=config.render_dpi,
        )

        context = build_page_context(state)

        interpretation, model_run = await interpreter.interpret(
            image_path=image_path,
            context=context,
        )

        assessment = PageAssessment(
            page_index=page_index,
            image_path=image_path,
            interpretation=interpretation,
            model_run=model_run,
        )

        state = finalize_toc_if_required(
            state=state,
            next_page=assessment,
        )

        validate_assessment(
            assessment=assessment,
            state=state,
            config=config,
        )

        assessment = extract_figure_assets(
            assessment=assessment,
            config=config,
        )

        state = apply_assessment(
            assessment=assessment,
            state=state,
        )

        document.pages.append(assessment)

        store.commit(
            state=state,
            document=document,
            assessment=assessment,
            context=context,
        )

    validate_complete_book(document=document, state=state)
    return document
```

Processing order:

```text
interpret
→ finalize prior TOC if appropriate
→ validate current page
→ extract assets
→ update state
→ commit atomically
```

---

## 13. State transitions

### 13.1 TOC state machine

```text
NOT_SEEN
    ├── TOC page → COLLECTING
    └── first chapter/body page → ABSENT

COLLECTING
    ├── TOC page → append entries
    └── non-TOC page → validate and COMMIT before validating page

COMMITTED
    └── later TOC page → error

ABSENT
    └── later TOC page → error
```

When committing the TOC, validate:

- it is not empty;
- titles are not empty;
- levels start plausibly;
- hierarchy does not jump by more than one;
- numeric page labels are mostly monotonic;
- duplicate entries are either explicitly allowed or rejected.

### 13.2 Chapter opening

On a `CHAPTER_OPENING` page:

1. locate exactly one `CHAPTER_TITLE` block;
2. compare it with the next expected TOC entry;
3. close the preceding chapter;
4. create a new `ChapterState`;
5. advance `next_expected_toc_index`.

Use deterministic title normalization:

```python
def normalize_title(value: str) -> str:
    value = value.casefold()
    value = normalize_unicode(value)
    value = remove_leader_dots(value)
    value = normalize_whitespace(value)
    value = strip_terminal_punctuation(value)
    return value
```

Do not initially use embeddings or LLM matching for TOC titles.

### 13.3 Paragraph continuation

Allow at most one paragraph to remain open across a page boundary.

Rules:

- `continues_previous=True` requires an open paragraph;
- normally only the first body paragraph may continue the prior page;
- `continues_on_next_page=True` opens continuation state;
- a chapter opening cannot begin while a prior paragraph remains open.

Store only the last few hundred characters in `BookState`.

Retain the full text in the document model.

---

## 14. Fail-fast validation

```python
class StructuralMismatchError(RuntimeError):
    def __init__(
        self,
        *,
        code: str,
        page_index: int,
        message: str,
        expected: object | None = None,
        observed: object | None = None,
    ) -> None:
        self.code = code
        self.page_index = page_index
        self.expected = expected
        self.observed = observed
        super().__init__(message)
```

Initial error codes:

| Code | Meaning |
|---|---|
| `invalid-model-output` | HTTP response or Pydantic validation failed |
| `unexpected-toc` | TOC encountered after TOC state was finalized |
| `invalid-toc-hierarchy` | TOC levels are structurally inconsistent |
| `missing-chapter-title` | Chapter-opening page lacks a chapter title |
| `multiple-chapter-titles` | More than one chapter-title block |
| `toc-chapter-mismatch` | Observed chapter does not match expected TOC entry |
| `unexpected-chapter` | Chapter found after TOC entries were exhausted |
| `heading-level-jump` | Heading hierarchy skips an unsupported level |
| `continuation-without-source` | Page claims to continue a closed paragraph |
| `unclosed-paragraph` | Structural boundary reached with open paragraph |
| `unresolved-footnote` | Reference has no corresponding note |
| `orphan-footnote` | Note body has no reference |
| `figure-crop-unavailable` | Inline figure has no usable bounding box |
| `toc-not-exhausted` | End of PDF reached before required entries appeared |

### Technical retries

Retry only for technical failures:

- timeout;
- temporary server error;
- malformed JSON;
- truncated response.

Do not automatically retry semantic contradictions:

- wrong chapter title;
- unsupported hierarchy;
- missing footnote;
- figure without usable crop.

---

## 15. Figure extraction

After structural validation:

```text
FigureBlock with bbox
    → crop from rendered high-resolution page
    → write assets/figure-p012-b004.png
    → set asset_path

FigureBlock without bbox on FIGURE_PAGE
    → use complete page image

FigureBlock without bbox on ordinary BODY page
    → figure-crop-unavailable error
```

Keep figure recognition separate from asset extraction.

A future parser may provide more precise geometry or embedded images.

---

## 16. Checkpoint and run layout

Use files instead of a database:

```text
runs/my-book/
├── run.json
├── state.json
├── document.json
├── error.json
│
├── pages/
│   ├── page-0001.png
│   ├── page-0002.png
│   └── ...
│
├── assessments/
│   ├── page-0001.json
│   └── ...
│
├── contexts/
│   ├── page-0001.json
│   └── ...
│
├── responses/
│   ├── page-0001.json
│   └── ...
│
├── assets/
│   ├── figure-p012-b004.png
│   └── ...
│
└── output/
    ├── book.md
    └── book.epub
```

`run.json` should record:

```json
{
  "schema_version": "0.1",
  "source_pdf_sha256": "...",
  "model": "book-vlm",
  "llama_cpp_version": "...",
  "prompt_version": "page-v1",
  "render_dpi": 240,
  "temperature": 0
}
```

### Atomic commit

For each accepted page:

1. write the assessment to a temporary file;
2. write the new state to a temporary file;
3. write the new document to a temporary file;
4. rename the files atomically.

Only increment `processed_page_count` in the committed state.

---

## 17. Markdown renderer

The renderer should not infer structure.

```python
class MarkdownRenderer:
    def render(self, document: BookDocument) -> str:
        ...
```

Mapping:

| Domain content | Markdown |
|---|---|
| Book title | YAML metadata or `#` |
| Chapter title | `#` |
| Heading level 2 | `##` |
| Paragraph | Plain paragraph |
| Blockquote | `>` |
| Footnote reference | `[^label]` |
| Footnote body | `[^label]: ...` |
| Figure | `![caption](assets/path.png)` |
| Separator | `---` |
| Running header/footer | Omitted |
| Page number | Omitted |

Optionally preserve source page markers:

```html
<!-- source-page: 42 -->
```

This is useful for debugging and may be disabled for final output.

---

## 18. EPUB generation

Do not build EPUB packages manually in version 0.1.

```text
BookDocument
    ↓ deterministic Markdown renderer
book.md + assets
    ↓ Pandoc
book.epub
```

Wrap Pandoc behind a narrow interface:

```python
class EpubRenderer(Protocol):
    def render(
        self,
        *,
        markdown_path: Path,
        output_path: Path,
        metadata_path: Path | None = None,
    ) -> None:
        ...
```

This allows replacing Pandoc later without modifying parsing or state management.

---

## 19. Configuration

Use one Pydantic settings object loaded from TOML:

```python
class ProcessingConfig(DomainModel):
    source_pdf: Path
    run_directory: Path

    render_dpi: int = 240

    llama_base_url: str = "http://127.0.0.1:8080"
    llama_model: str = "book-vlm"
    request_timeout_seconds: float = 300

    prompt_version: str = "page-v1"
    disable_thinking: bool = True

    strict_model_warnings: bool = False
    require_figure_crops: bool = True
    require_same_page_footnotes: bool = True

    markdown_page_markers: bool = True
```

Avoid large numbers of switches.

Add a setting only when real books require differing behavior.

---

## 20. CLI

Keep the CLI small:

```bash
bookextract init book.pdf --run runs/book
bookextract process --run runs/book
bookextract status --run runs/book
bookextract render-markdown --run runs/book
bookextract render-epub --run runs/book
bookextract inspect-error --run runs/book
bookextract reset-after 17 --run runs/book
```

`reset-after 17` should:

1. delete committed assessments after page 17;
2. reconstruct state by replaying accepted assessments through page 17.

This is much simpler than general dependency tracking.

A later command may support manually replacing an assessment:

```bash
bookextract replace-assessment 17 corrected-page-0017.json
```

---

## 21. Testing strategy

### 21.1 Unit tests

Most tests should not use a real model.

#### State reducer tests

Test sequences such as:

```text
title → toc → toc → chapter opening → body
title → chapter opening without TOC
chapter 1 → chapter 2
```

#### Validator tests

Each mismatch code should have at least one focused test.

#### Renderer tests

Construct small `BookDocument` objects manually and assert exact Markdown output.

#### Context tests

Ensure that context:

- exposes expected TOC entries;
- limits heading history;
- includes the open paragraph tail;
- does not include unnecessary full text.

### 21.2 Contract tests

Run `LlamaCppVisionClient` against a fake HTTP server and verify:

- image-path encoding;
- JSON schema inclusion;
- timeout handling;
- malformed response handling;
- token-usage extraction.

### 21.3 Golden page tests

Maintain a small manually reviewed set:

```text
title page
copyright page
two TOC pages
chapter opening
ordinary body page
body page with subheading
footnote page
full-page illustration
inline image and caption
```

For each page, store:

```text
context.json
expected-assessment.json
```

The core tests should replay these without running the model.

### 21.4 Real-model integration tests

Mark separately:

```python
@pytest.mark.local_model
```

Assertions should focus on structure:

- response passes schema;
- page type is correct;
- expected chapter title appears;
- footnote reference and body labels match;
- figure is detected.

Do not require byte-for-byte identical output across model versions.

---

## 22. Model evaluation

Do not hard-code the architecture around Gemma or Qwen.

Benchmark models through the same `VisionModelClient`.

Suggested candidates:

- the existing Qwen3.6 setup;
- Qwen3.6 27B or 35B-A3B with verified vision support;
- Gemma 4 31B;
- Gemma 4 26B-A4B.

Use a representative benchmark derived from actual books.

| Metric | Measurement |
|---|---|
| Schema reliability | Valid responses per page |
| Text fidelity | Manually inspected character errors |
| Page type | Exact accuracy |
| Heading role | Precision and recall |
| Heading level | Exact accuracy |
| TOC extraction | Entry and level accuracy |
| Footnotes | Reference/body matching accuracy |
| Figures | Detection and crop usability |
| Performance | Seconds and tokens per page |

Keep constant between model runs:

- prompt;
- schema;
- DPI;
- temperature;
- `llama.cpp` build;
- quantization.

Choose the default model based on performance on actual books.

---

## 23. Implementation milestones

### Milestone 1 — One-page vertical slice

Deliver:

- PDF page rendering;
- llama.cpp client;
- `PageInterpretation`;
- schema-constrained response;
- saved prompt, context, raw response, and assessment.

Acceptance criterion:

```text
One page image → valid PageInterpretation JSON
```

### Milestone 2 — Sequential checkpointed traversal

Deliver:

- `BookState`;
- `PageContext`;
- processing loop;
- atomic checkpoints;
- crash recovery.

Acceptance criterion:

```text
A 20-page PDF can be processed sequentially and resumed after interruption.
```

### Milestone 3 — Front matter and TOC

Deliver:

- TOC collection;
- TOC commit point;
- title normalization;
- chapter matching;
- structured mismatch error.

Acceptance criterion:

```text
A chapter title contradicting the committed TOC stops processing and
produces error.json with expected and observed values.
```

### Milestone 4 — Hierarchical text and Markdown

Deliver:

- headings;
- paragraphs;
- cross-page continuation;
- running-header/footer suppression;
- Markdown renderer.

Acceptance criterion:

```text
A conventional prose chapter produces readable Markdown with consistent
heading levels and paragraph boundaries.
```

### Milestone 5 — Footnotes

Deliver:

- inline footnote-reference segments;
- footnote blocks;
- same-page reference validation;
- Markdown footnote output.

Acceptance criterion:

```text
Every footnote reference has exactly one body and every body has a reference.
```

### Milestone 6 — Images

Deliver:

- figure detection;
- normalized bounding boxes;
- image cropping;
- captions;
- Markdown assets.

Acceptance criterion:

```text
Full-page figures and bounded inline figures appear in the correct document order.
```

### Milestone 7 — EPUB

Deliver:

- metadata file;
- Pandoc adapter;
- EPUB output.

Acceptance criterion:

```text
The same accepted BookDocument renders to Markdown and EPUB without
additional structural inference.
```

### Milestone 8 — Second interpretation approach

Only after the first pipeline works, implement one alternative:

```python
class DoclingPageInterpreter(PageInterpreter):
    ...
```

or:

```python
class TwoStageVlmPageInterpreter(PageInterpreter):
    ...
```

This validates whether the abstraction works in practice.

---

## 24. Explicit version-0.1 exclusions

Do not include:

- automatic re-evaluation of earlier pages;
- competing interpretations;
- confidence aggregation;
- arbitrary cross-page footnotes;
- endnotes spanning chapters;
- complex tables;
- equations;
- marginalia;
- multi-column scholarly layouts;
- handwritten annotations;
- generated image descriptions;
- automatic spelling correction;
- multi-model voting;
- web interface;
- visual block editor.

Each unsupported case should either be retained as `OTHER` or fail with a defined error.

---

## 25. Definition of done

Version 0.1 is successful when it can:

1. process pages sequentially through a local llama.cpp VLM;
2. pass authoritative book and chapter context forward;
3. extract a typed page representation;
4. commit a TOC and validate later chapters against it;
5. retain optional geometry and typography;
6. handle headings, paragraphs, same-page footnotes, figures, and captions;
7. stop with a precise structural error on contradiction;
8. resume after interruption;
9. produce deterministic Markdown;
10. convert the Markdown to EPUB;
11. replace the VLM interpreter later without changing the processing loop.

The central maintainability rule is:

> The model observes and describes a page. Pure application code validates, commits, and renders the book.
