---
name: Lean VLM Book Extractor
overview: "FROZEN — v0.1. Implementation-ready. Environment-before-smoke, exact Pandoc profile, ContextBudgetResult schema, render_contract. Runtime: Pydantic, HTTPX, PyMuPDF. Milestones 0–7."
todos:
  - id: branch-setup
    content: git checkout main && git pull --ff-only && git switch -c implement/bookextract-v0.1
    status: pending
  - id: m0-schema-spike
    content: "M0: persist TokenCountingContract + ThinkingControlContract, input_tokens probe, thinking tests"
    status: pending
  - id: scaffold
    content: Scaffold pyproject.toml, ProcessingConfig, argparse CLI, Ruff + mypy
    status: pending
  - id: m1-models-pdf-vlm
    content: "M1: InterpretationResult, budget probes, frozen contracts in client, one-page commit"
    status: pending
  - id: m2-pipeline-storage
    content: "M2: atomic failures, validate_commit, inference-environment, recovery"
    status: pending
  - id: m3-toc-sections
    content: "M3: TOC filter, metadata policy, structural opening validation"
    status: pending
  - id: m4-markdown
    content: "M4: RenderedPublicationBlock, NFC canonicalization, MarkdownRenderer"
    status: pending
  - id: m5-footnotes
    content: "M5: document-order footnote refs, document-end definitions"
    status: pending
  - id: m6-figures
    content: "M6: bbox transform, FigureCropPolicy, commit assets/"
    status: pending
  - id: m7-epub
    content: "M7: PublicationDocument UUID, source_map, Pandoc/EPUBCheck, publish"
    status: pending
  - id: tests
    content: Frozen contracts, budget probes, thinking smoke, identity NFC, replay
    status: pending
isProject: false
---

# Lean Local VLM Book Extraction — Implementation Plan (FROZEN)

Normative spec for v0.1.

**Status: FROZEN.** Copy to `docs/plans/implementation_plan_v0.1.md` at M0 scaffold. Begin on `implement/bookextract-v0.1`.

```text
Runtime:  Pydantic | HTTPX (sync) | PyMuPDF
Dev:      pytest | Ruff | mypy (+ pydantic plugin)
External: llama-server | Pandoc | EPUBCheck
Manager:  uv (commit uv.lock)
```

## Normative document hierarchy

1. `implementation_plan_v0.1.md` governs implementation details, persistence formats, milestone acceptance criteria, and v0.1 scope.
2. `lean_local_vlm_book_extraction_plan.md` supplies architectural definitions not repeated here (domain model §6, prompt design §11, validation codes §14).
3. Where the documents conflict, `implementation_plan_v0.1.md` takes precedence.
4. Prior chat revisions and drafts are non-normative.

## Git workflow

```bash
git checkout main
git pull --ff-only
git switch -c implement/bookextract-v0.1
```

---

## Run directory layout

```text
runs/book/
├── run.json                      # run_format_version: 1
├── source-location.json
├── inference-location.json
├── inference-environment.json    # write-once; includes M0 frozen contracts
├── head.json                     # head_format_version: 1
├── state.json                    # state_cache_version: 1
├── pages/
├── commits/
├── failures/
├── diagnostics/                  # preflight thinking-smoke artifacts
├── recovery/
├── .output-build/
└── output/
```

Location JSON files: write temp → `os.replace()`.

### Lifecycle

**`init`**: hash PDF; freeze config in `run.json`; write location files; no `inference-environment.json`.

**First `process`** (environment created atomically after all calibration succeeds):

```text
preflight → path binding → file hashing
→ token-count calibration (discovery; fallback allowed)
→ thinking-control calibration (discovery)
→ construct complete InferenceEnvironment
→ write inference-environment.json.tmp → os.replace()
→ process pages
```

Failed probe must not leave a partial or prematurely frozen environment.

**Later `process`**:

```text
preflight → parse/validate server response
→ verify model-path binding
→ fingerprint configured files
→ compare durable server identity and file fingerprints
→ verify frozen token/thinking contract metadata
→ run thinking smoke
→ process pages
```

Error precedence:

```text
different build/template/model → inference-environment-drift (exit 13)
same environment, smoke violates contract → thinking-control-contract-drift (exit 13)
same environment, smoke timeout/transient 5xx → thinking-smoke-unavailable (exit 10)
```

**`relocate-inference-files`**: before first process → update location only; after → fingerprint + compare + update location.

---

## Parser-independent pipeline

`pipeline.py` never imports `wire.py` or `conversion.py`.

```text
for each page:
    render → PageInput → PageContext
    → PageInterpreter.interpret() → InterpretationResult
    → assign IDs → validate → reduce → commit
```

### Interpretation boundary

```python
@dataclass(frozen=True, slots=True)
class PendingArtifact:
    logical_name: str
    media_type: str
    content: bytes
    filename: str | None = None   # single-segment basename only


@dataclass(frozen=True, slots=True)
class InterpretationResult:
    interpretation: PageInterpretation
    provenance: InterpretationProvenance
    artifacts: tuple[PendingArtifact, ...] = ()
    failed_attempts: tuple[InferenceAttempt, ...] = ()
```

```text
LlamaCppVisionClient.generate_structured()
    → InferenceResult[VlmPageResponse]  (see Appendix B)

VlmPageInterpreter.interpret()
    → InterpretationResult + PendingArtifacts

pipeline → validate → RunStore
```

### Failure flow

```python
@dataclass(frozen=True, slots=True)
class InferenceFailureContext:
    prompt: bytes
    request_summary: bytes
    schema_ref: bytes
    page_image_sha256: str
    wire_request_sha256: str | None


class InferenceError(BookExtractError):
    code: str
    retryable: bool
    attempts_exhausted: bool
    context: InferenceFailureContext
    attempts: tuple[InferenceAttempt, ...]
```

Pre-request: `attempts=()`, no `attempts/` under failure dir.

---

## Server preflight

### v0.1 contract

> Exactly **one loaded model**, unambiguous `/props`, matching alias, verified absolute model-path binding. Router mode not deliberately supported; need not be detected if conditions hold.

```text
GET /health → GET /v1/models → GET /props
```

Reject `unsupported-multi-model-server` when `/v1/models` returns >1 model.

### LlamaWireModel

```python
class LlamaWireModel(BaseModel):
    model_config = ConfigDict(extra="ignore", strict=True)

class LlamaAssistantMessage(LlamaWireModel):
    content: str | None = None
    reasoning_content: str | None = None
```

Missing/wrong required fields → `unsupported-llama-server-contract`. Additive fields ignored. Raw response retained.

### Model path binding

```python
if not server_model_path.is_absolute():
    raise ProcessingError(code="unverifiable-server-model-path")
if server_model_path.resolve() != configured_model_path.resolve():
    raise ProcessingError(code="server-model-path-mismatch")
```

### InferenceEnvironment (write-once)

```json
{
  "inference_environment_format_version": 1,
  "server": {},
  "model_file": {},
  "projector_file": {},
  "token_counting_contract": {},
  "thinking_control_contract": {}
}
```

```python
class InferenceEnvironment(DomainModel):
    inference_environment_format_version: int = 1
    server: ServerInferenceIdentity
    model_file: FileFingerprint | None = None
    projector_file: FileFingerprint | None = None
    model_binding_verified: bool
    projector_binding: Literal["server-verified", "operator-asserted", "unavailable"]
    fingerprints_complete: bool = False
    token_counting_contract: TokenCountingContract
    thinking_control_contract: ThinkingControlContract
```

Persist `ServerInferenceIdentity`. Transient `ServerInvocationCapabilities` per preflight.

### M0 frozen contracts (discriminated unions)

**Token counting** — persisted after first-process discovery only:

```python
class MultimodalInputTokensContract(DomainModel):
    mode: Literal["chat-input-tokens-multimodal"]
    model_alias: str
    llama_cpp_build: str
    chat_template_sha256: str

class TextOnlyInputTokensContract(DomainModel):
    mode: Literal["chat-input-tokens-text-only"]
    image_token_policy: Literal["configured-reserve"]
    model_alias: str
    llama_cpp_build: str
    chat_template_sha256: str

class ApplyTemplateTokenizeContract(DomainModel):
    mode: Literal["apply-template-tokenize"]
    image_token_policy: Literal["configured-reserve"]
    model_alias: str
    llama_cpp_build: str
    chat_template_sha256: str

class EstimateOnlyContract(DomainModel):
    mode: Literal["estimate-only"]
    estimation_version: int = 1
    model_alias: str
    llama_cpp_build: str
    chat_template_sha256: str

```python
from typing import Annotated

TokenCountingMode = Literal[
    "chat-input-tokens-multimodal",
    "chat-input-tokens-text-only",
    "apply-template-tokenize",
    "estimate-only",
]

TokenCountingContract = Annotated[
    MultimodalInputTokensContract
    | TextOnlyInputTokensContract
    | ApplyTemplateTokenizeContract
    | EstimateOnlyContract,
    Field(discriminator="mode"),
]
```
```

**First-process discovery:** probe candidates in order; fallback allowed; persist only after all calibration tests pass.

**Later-process execution:** use persisted mode only; no fallback selection. Transient timeout/502/503/504 → retry once. Unsupported or malformed → `ProcessingError(code="token-counting-contract-drift")`.

**Thinking control** — sole authority for non-thinking operation (not in `ExtractionConfig`):

```python
class ThinkingControlContract(DomainModel):
    contract_format_version: int = 1
    enable_thinking: Literal[False] = False
    reasoning_format: Literal["none", "deepseek", "deepseek-legacy"]
    applied_template_probe_supported: bool
    reasoning_content_expected: Literal[False] = False
    model_alias: str
    llama_cpp_build: str
    chat_template_sha256: str
```

Invariant: **a v0.1 run always requires verified non-thinking operation.**

**First-process discovery:** calibrate contract; failure → `unsupported-thinking-control`.

**Later-process enforcement:** smoke must satisfy frozen contract; failure → `thinking-control-contract-drift`.

Production payload:

```python
payload = {
    ...
    "chat_template_kwargs": {"enable_thinking": thinking_contract.enable_thinking},
    "reasoning_format": thinking_contract.reasoning_format,
}
```

Enforce on response:

```python
if response.message.reasoning_content:
    raise InferenceError(code="unexpected-reasoning-content", retryable=False, ...)
```

### M0 thinking-control tests (four-part)

1. Applied-template probe — only if `applied_template_probe_supported`.
2. Raw unconstrained probe — `reasoning_format` from contract; inspect raw text.
3. Parsed unconstrained probe — inspect `reasoning_content`.
4. Production-schema probe — valid `VlmPageResponse`; JSON-only content.

Failure during first-process calibration → `unsupported-thinking-control`.

### Runtime thinking smoke (every `process`, not per page)

One small non-thinking smoke probe per processing invocation. Atomic write:

```text
diagnostics/.preflight-0003.tmp/
  → diagnostics/preflight-0003/
      preflight.json
      thinking-smoke-request.json
      thinking-smoke-response.json
      result.json
```

Does not increment `InterpretationProvenance.attempts`.

| Outcome | Error | Exit |
|---|---|---|
| Smoke violates frozen non-thinking behavior | `thinking-control-contract-drift` | 13 |
| Timeout / transient 5xx after retry | `InferenceError(code="thinking-smoke-unavailable", retryable=True)` | 10 |

### Static feasibility

```python
if context_size <= max_tokens + context_safety_margin_tokens:
    raise ProcessingError(code="static-context-impossible")
```

---

## Context budget (inside LlamaCppVisionClient)

Uses frozen `token_counting_contract`. Do not base64-encode for **cheap estimates**.

```text
cheap estimate: no base64 construction

exact input_tokens count near limit:
    construct complete production body once (including base64 image)
    hash it → count it → reuse same bytes for completion
```

```python
effective_image_tokens = (
    config.server_image_max_tokens
    if config.server_image_max_tokens is not None
    else config.reserved_image_tokens
)
```

### Exact-byte reuse (`input_tokens` modes)

When frozen mode uses `/v1/chat/completions/input_tokens`:

```python
wire_body = serialize_wire_request(payload)

token_count = count_input_tokens(
    body=wire_body,
    content_type="application/json",
)

if budget_fits(token_count):
    response = client.post(
        "/v1/chat/completions",
        content=wire_body,
        headers={"Content-Type": "application/json"},
    )
```

Budget rejection after exact serialization: `stage="serialized"`, `wire_request_sha256` present, `attempts=()`.

### M0 decision tree (produces `TokenCountingContract`)

```text
1. Probe POST /v1/chat/completions/input_tokens.
2. Compare text-only vs with-image vs resolutions.
3. Multimodal-inclusive → required = input_tokens + max_tokens + safety_margin
4. Text-only → required = input_tokens + effective_image_tokens + max_tokens + safety_margin
5. Unavailable → apply-template + tokenize + image reserve → estimate-only if needed
```

### Budget-probe policy (not completion retries)

Auxiliary probes (`input_tokens`, `apply-template`, `tokenize`) do **not** appear in `attempts/` and do **not** increment `InterpretationProvenance.attempts`.

**First-process discovery:**

| Outcome | Action |
|---|---|
| 404 / unsupported | Next fallback candidate |
| Valid response | Use returned count |
| Malformed response | Mark mechanism unsupported; fallback |
| Timeout / 502 / 503 / 504 | Retry once; then fallback |
| 400 from production request shape | Fail calibration; do not silently estimate |
| All exact mechanisms unavailable | `estimate-only` unless too close to limit |
| Estimate ambiguous near limit | `InferenceError(code="context-budget-indeterminate")` |

**Later-process execution (frozen contract):**

Endpoint failure rows apply to endpoint-based modes only (`chat-input-tokens-*`, `apply-template-tokenize`). `estimate-only` invokes `estimation_version` directly — no endpoint can drift.

| Outcome | Error | Exit |
|---|---|---|
| Valid response per persisted mode | Use returned count | — |
| 404, unsupported shape, malformed successful response | `token-counting-contract-drift` | 13 |
| Valid response inconsistent with frozen calibration | `token-counting-contract-drift` | 13 |
| Timeout / connection / 502/503/504 after one retry | `context-budget-probe-unavailable` | 10 |
| Estimate ambiguous near limit | `context-budget-indeterminate` | 10 |

```python
class ContextBudgetResult(DomainModel):
    counted_input_tokens: int
    image_tokens_reserved: int
    output_tokens_reserved: int
    safety_margin_tokens: int
    context_size: int
    counting_mode: TokenCountingMode
    exact_for_projected_input: bool
    multimodal_count_included: bool

    @property
    def required_tokens(self) -> int:
        return (
            self.counted_input_tokens
            + self.image_tokens_reserved
            + self.output_tokens_reserved
            + self.safety_margin_tokens
        )
```

For multimodal-inclusive mode, `image_tokens_reserved` must be **0** (avoid double-counting).

Persist in `request-summary.json`:

```json
{
  "context_budget": {
    "counting_mode": "chat-input-tokens-multimodal",
    "counted_input_tokens": 5274,
    "image_tokens_reserved": 0,
    "output_tokens_reserved": 8192,
    "safety_margin_tokens": 512,
    "context_size": 16384,
    "exact_for_projected_input": true,
    "multimodal_count_included": true
  }
}
```

Completion retries apply only to `POST /v1/chat/completions` (Appendix B).

Page failure → `InferenceError(code="context-budget-exceeded")` exit **10**.

---

## Wire request serialization

```python
wire_body = serialize_wire_request(payload)
client.post(..., content=wire_body, headers={"Content-Type": "application/json"})
```

## Staged RequestSnapshot

```python
stage: Literal["planned", "serialized"]
wire_request_sha256: str | None = None
```

---

## Commit manifest and validation

Exhaustive manifest; nested paths in manifest; single-segment `PendingArtifact.filename`.

`reject_symlink_components()` on every path component. Enumerate with `followlinks=False`.

`validate_commit()`: manifest ↔ all regular files (except `manifest.json`); no symlinks; contiguous sequence.

### head.json

```json
{"head_format_version": 1, "committed_page_count": 7}
```

Recovery: `recovery/recovery-NNNN/`; `.page-NNNN.tmp/` → rename → head last (Appendix H).

---

## Failure persistence (complete, atomic)

```text
failures/page-0017/failure-0001/
├── context.json, page-input.json, prompt.txt, schema-ref.json
├── request-summary.json
├── attempts/…          # omitted when attempts == ()
└── error.json
```

`.failure-NNNN.tmp/` → rename → replace `latest.json`.

---

## Publication identifier (fully output-semantic)

### Identity invariant

```text
identical visible metadata, text, headings, footnotes, figures,
and semantic render settings → identical publication UUID
```

### Unicode normalization

Apply NFC to all visible publication strings before canonical hashing:

```python
import unicodedata

def normalize_publication_string(value: str) -> str:
    return unicodedata.normalize("NFC", value)
```

Apply recursively through `PublicationDocument` projection:

```python
def normalize_publication_document_nfc(doc: PublicationDocument) -> PublicationDocument:
    ...
```

### Canonical identity (exact algorithm)

```python
def canonical_json_bytes(value: BaseModel) -> bytes:
    data = value.model_dump(mode="json", exclude_none=False)
    return json.dumps(
        data, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")

normalized_document = normalize_publication_document_nfc(publication_document)

publication_fingerprint = hashlib.sha256(
    canonical_json_bytes(normalized_document)
).hexdigest()

publication_uuid = uuid5(NAMESPACE_URL, f"bookextract:{publication_fingerprint}")
publication_identifier = f"urn:uuid:{publication_uuid}"
```

### Semantic render profiles (v0.1)

```python
class MarkdownSemanticProfile(DomainModel):
    profile_format_version: int = 1
    footnote_style: Literal["document-end"] = "document-end"
    heading_style: Literal["atx"] = "atx"

class EpubSemanticProfile(DomainModel):
    profile_format_version: int = 1
    from_format: Literal["markdown+footnotes"] = "markdown+footnotes"
    to_format: Literal["epub3"] = "epub3"
    split_level: Literal[1] = 1
    include_toc: bool
```

### Rendered blocks → projection

```python
class PublicationDocument(DomainModel):
    identity_format_version: int = 1
    metadata: PublicationMetadata
    blocks: list[PublicationBlock]
    markdown_semantic_profile: MarkdownSemanticProfile
    epub_semantic_profile: EpubSemanticProfile
```

Built after merge, heading resolution, footnote resolution, final metadata, figure materialization.

Excluded from identity: internal IDs, `domain_schema_version`, page structure, extraction mechanics.

Footnotes: document-order refs; visible labels only. Figures: `asset_sha256`.

Ordered `source_map` in `output/manifest.json` by `publication_block_index`.

Page markers: HTML comments only; nonsemantic.

---

## Error hierarchy

```python
RUN_STATE_CODES = {
    "invalid-run-layout", "source-hash-mismatch", "config-drift",
    "schema-drift", "inference-environment-drift", "render-environment-drift",
    "static-context-impossible",
    "unsupported-multi-model-server", "unsupported-llama-server-contract",
    "unverifiable-server-model-path", "server-model-path-mismatch",
    "unsupported-thinking-control", "thinking-control-contract-drift",
    "token-counting-contract-drift", "invalid-commit-artifact-path",
    "unsupported-pandoc-defaults",
}
```

`invalid-artifact-path` → RunStore (exit 14). `invalid-commit-artifact-path` → startup (exit 13).

Page-level `InferenceError`: `context-budget-exceeded`, `context-budget-indeterminate`, `context-budget-probe-unavailable`, `thinking-smoke-unavailable`, `page-image-too-large`, `unexpected-reasoning-content` → exit **10**.

Exit: 0; 2 argparse; 1 unexpected; 10 inference; 11 structural; 12 external; 13 run-state; 14 PDF/storage.

---

## Package layout

```text
bookextract/
├── cli.py, config.py, models.py, errors.py, artifacts.py
├── wire.py, schema.py, conversion.py
├── pdf.py, assets.py, canonical.py
├── context.py, validation.py, state.py, pipeline.py, storage.py
├── interpretation/{base,vlm,prompts}.py
├── inference/{base,llamacpp}.py
└── rendering/{markdown,epub,publication}.py

schemas/vlm-page-response-v1.json
resources/pandoc-epub-v1.json
docs/plans/implementation_plan_v0.1.md
```

---

## Milestones

| M | Deliverables |
|---|---|
| **M0** | Copy plan; persist M0 contracts; `input_tokens` probe; thinking tests + smoke |
| **M1** | `InterpretationResult`; budget probes; frozen contracts in client; commit |
| **M2** | Atomic failures; `validate_commit`; environment; recovery |
| **M3** | TOC; metadata (Appendix F) |
| **M4** | `PublicationDocument`; NFC; Markdown |
| **M5** | Footnotes (Appendix G) |
| **M6** | Figures; coordinates (Appendix E) |
| **M7** | UUID; Pandoc/EPUBCheck (Appendix D); publish |

---

## Pre-implementation checklist

All contracts defined in:

- the main body of `implementation_plan_v0.1.md`;
- Appendices A–H;
- architecture sections explicitly incorporated by the Normative document hierarchy.

- [x] Normative hierarchy + appendices A–H
- [x] Environment compare before thinking smoke; smoke-unavailable exit 10
- [x] Pandoc exact-key validation; profile from parsed values
- [x] `ContextBudgetResult` aligned with persisted JSON; discriminated union
- [x] `render_contract` in `run.json`; `render-environment-drift`
- [x] Frozen config table; M0 contracts; retry invariants; NFC UUID algorithm

---

## Definition of done

1. M0–M7 on `implement/bookextract-v0.1`; merged to `main`.
2. Self-contained `implementation_plan_v0.1.md` with appendices.
3. M0 contracts persisted and used for entire run.
4. Budget probes audited in `request-summary.json`; completions in `attempts/` only.
5. Output-semantic UUID with NFC normalization.
6. Assessment commits authoritative; replay invariants hold.

---

# Appendix A — Configuration

Nested `ProcessingConfig` loaded from TOML at CLI time; frozen subset in `run.json` at `init`.

```python
class ExtractionConfig(DomainModel):
    render_dpi: int = 240
    render_annotations: bool = False
    model_alias: str
    prompt_version: str
    wire_schema_version: str = "vlm-page-response-v1"
    temperature: float = 0.0
    seed: int = 0
    max_tokens: int = 8192
    require_figure_crops: bool = True
    require_same_page_footnotes: bool = True
    reserved_image_tokens: int = 2048
    context_safety_margin_tokens: int = 512
    server_image_max_tokens: int | None = None

class InferenceFingerprintConfig(DomainModel):
    require_complete_fingerprint: bool = False

class ProcessOptions(DomainModel):
    llama_base_url: str = "http://127.0.0.1:8080"
    request_timeout_seconds: float = 300
    max_attempts: int = 2
    retry_backoff_seconds: float = 1.0
    retry_after_max_seconds: float = 60.0

class MarkdownRenderConfig(DomainModel):
    include_page_markers: bool = True

class EpubRenderConfig(DomainModel):
    pandoc_executable: str = "pandoc"
    epubcheck_executable: str = "epubcheck"
    epubcheck_jar_path: Path | None = None
    include_toc: bool = True

class ProcessingConfig(DomainModel):
    extraction: ExtractionConfig
    fingerprint: InferenceFingerprintConfig = Field(default_factory=InferenceFingerprintConfig)
    process: ProcessOptions = Field(default_factory=ProcessOptions)
    markdown: MarkdownRenderConfig = Field(default_factory=MarkdownRenderConfig)
    epub: EpubRenderConfig = Field(default_factory=EpubRenderConfig)
```

Fixed render constants (also frozen in `run.json`):

```python
class RenderContract(DomainModel):
    render_contract_format_version: int = 1
    colorspace: Literal["RGB"] = "RGB"
    alpha: Literal[False] = False
    pymupdf_version: str
```

At `process`, compare `pymupdf.__version__` to frozen value. Mismatch → `render-environment-drift` (exit 13). Existing commits remain authoritative; check prevents mixing render environments within one run.

CLI: `argparse`; mutually exclusive `--through-page` / `--max-pages`; `process --model` / `--projector` overrides.

### Frozen vs invocation-local configuration

| Configuration | Persistence and drift policy |
|---|---|
| `ExtractionConfig` (entire model) | Frozen in `run.json`; drift → `config-drift` |
| `InferenceFingerprintConfig.require_complete_fingerprint` | Frozen in `run.json` |
| Prompt and wire-schema hashes | Frozen in `run.json` |
| RGB/alpha/render constants | Frozen as `render_contract` in `run.json`; drift → `render-environment-drift` |
| `ProcessOptions.llama_base_url` | Invocation-local; may change |
| Timeouts and retry timing | Invocation-local; may change |
| Pandoc/EPUBCheck executable paths | Invocation-local |
| `MarkdownRenderConfig.include_page_markers` | Output option; may change (nonsemantic) |
| `EpubRenderConfig.include_toc` | Output-semantic; may change; affects publication fingerprint |
| `epubcheck_jar_path` | Invocation-local |

Drift check at `process` startup:

```python
frozen_extraction = ExtractionConfig.model_validate(run_json["extraction"])

if current_config.extraction != frozen_extraction:
    raise ProcessingError(code="config-drift")
```

### Persistence schemas

**`run.json`** (immutable after init):

```json
{
  "run_format_version": 1,
  "source": {"sha256": "...", "size_bytes": 0, "page_count": 0},
  "extraction": {},
  "fingerprint_policy": {"require_complete_fingerprint": false},
  "render_contract": {
    "render_contract_format_version": 1,
    "colorspace": "RGB",
    "alpha": false,
    "pymupdf_version": "1.24.9"
  },
  "prompt_sha256": "...",
  "wire_schema_sha256": "...",
  "created_at": "2026-07-14T00:00:00Z"
}
```

No `source.path` in `run.json` — path lives only in `source-location.json`.

**`source-location.json`**:

```json
{
  "source_location_format_version": 1,
  "pdf_path": "/absolute/path/book.pdf"
}
```

**`inference-location.json`**:

```json
{
  "inference_location_format_version": 1,
  "model_file_path": "/path/model.gguf",
  "projector_file_path": "/path/mmproj.gguf"
}
```

**`state.json`**: `{"state_cache_version": 1, "committed_page_count": N, "state": {}}`

**`failures/page-NNNN/latest.json`**:

```json
{
  "latest_failure_pointer_format_version": 1,
  "failure_directory": "failure-0003"
}
```

---

# Appendix B — Inference types and completion retries

```python
@dataclass(frozen=True, slots=True)
class InferenceAttempt:
    attempt_number: int
    succeeded: bool
    status_code: int | None = None
    response_body: bytes | None = None      # failed attempts only
    content_type: str | None = None
    error_code: str | None = None
    error_message: str | None = None
    elapsed_ms: float | None = None
    finish_reason: str | None = None

    def __post_init__(self) -> None:
        if self.attempt_number < 1:
            raise ValueError("attempt_number must be positive")
        if self.succeeded:
            if self.error_code is not None:
                raise ValueError("successful attempt cannot have an error code")
            if self.error_message is not None:
                raise ValueError("successful attempt cannot have an error message")
            if self.response_body is not None:
                raise ValueError("successful body belongs in final_raw_body")
        elif self.error_code is None:
            raise ValueError("failed attempt requires an error code")


@dataclass(frozen=True, slots=True)
class InferenceResult(Generic[T]):
    value: T
    final_raw_body: bytes
    request_snapshot: RequestSnapshot
    attempts: tuple[InferenceAttempt, ...]
    # Invariants:
    # - attempts numbered contiguously from 1
    # - exactly one successful attempt (the final one)
    # - all preceding attempts failed
    # - final_raw_body is nonempty
```

**`InferenceError` invariants:**

```text
- all recorded attempts failed
- attempts contiguous from 1
- attempts_exhausted=True only when retryable failure reached max_attempts
- pre-request failures: attempts=(), attempts_exhausted=False
```

**Retry classification** (completions only; one logical interpretation per page):

| Failure | Retry | `InferenceAttempt.error_code` |
|---|---|---|
| Connection failure or request timeout | Yes | `transport-timeout` |
| HTTP 429, 502, 503, 504 | Yes | `http-retryable` |
| Other HTTP 5xx | Once | `http-server-error` |
| Disconnected or truncated HTTP body | Yes | `response-body-truncated` |
| `finish_reason=length` | No | `output-token-limit` |
| Complete body that is not valid JSON | No | `invalid-json` |
| Valid JSON failing `VlmPageResponse` validation | No | `invalid-structured-response` |
| Structurally valid response with semantic contradiction | No | (structural; not retried) |
| Empty response from apparent transport failure | Yes | `empty-response` |
| Complete successful response with empty content | No | `empty-response` |

Do not retry complete malformed responses from deterministic schema-constrained requests.

Backoff: `delay = retry_backoff_seconds * (2 ** (attempt - 1))`. HTTP 429: respect `Retry-After`, cap `retry_after_max_seconds`.

Memory: failed attempts carry `response_body`; successful body only in `final_raw_body`.

`LlamaCppVisionClient` owns completion retry loop exclusively.

---

# Appendix C — RenderedPage and coordinates

```python
class RenderedPage(DomainModel):
    image_path: Path
    width_px: int
    height_px: int
    page_rect: RectPoints
    rotation_degrees: int
    image_sha256: str
    image_size_bytes: int
```

`MAX_PAGE_IMAGE_BYTES = 40 * 1024 * 1024`.

Coordinate pipeline (architecture §6.3):

```text
VLM BBox1000 [0,1000]
  → pixel coords on rendered image
  → display rect (accounting for rotation)
  → derotation to cropbox space
  → PyMuPDF crop for figure assets
```

Base64 data URI transport to llama-server; never persist base64 in artifacts.

---

# Appendix D — Pandoc and EPUBCheck

Frozen base defaults `resources/pandoc-epub-v1.json` (v0.1 contents):

```json
{
  "from": "markdown+footnotes",
  "to": "epub3",
  "split-level": 1
}
```

At runtime validate exact keys and values:

```python
REQUIRED_BASE_DEFAULT_KEYS = {"from", "to", "split-level"}

if set(base) != REQUIRED_BASE_DEFAULT_KEYS:
    raise ProcessingError(code="unsupported-pandoc-defaults")

effective_profile = EpubSemanticProfile(
    from_format=base["from"],
    to_format=base["to"],
    split_level=base["split-level"],
    include_toc=config.include_toc,
)
```

Literal fields reject unsupported values (e.g. `commonmark`, `epub2`). Raw file hash → provenance only.

```python
build = {**base, "toc": config.include_toc, "metadata": publication_metadata}
```

```python
build_directory = run_dir / ".output-build"
command = [
    config.pandoc_executable,
    "--defaults", str((build_directory / "pandoc-build.json").resolve()),
    "book.md", "--resource-path", ".", "--output", "book.epub",
]
subprocess.run(command, cwd=build_directory, check=True, capture_output=True, text=True)
```

EPUB semantic profile (identity) — see main body `MarkdownSemanticProfile` / `EpubSemanticProfile`.

EPUBCheck:

```python
if config.epubcheck_jar_path:
    ["java", "-jar", str(jar), "--json", str(report), str(epub)]
else:
    [config.epubcheck_executable, "--json", str(report), str(epub)]
```

Optional: `SOURCE_DATE_EPOCH` from `run.json` `created_at` for reproducible timestamps.

---

# Appendix E — Figures (M6)

After structural validation: crop from rendered page using transformed bbox; write `assets/figure-pNNN-bNNN.png` in commit; manifest nested path.

`FigureBlock` without bbox on ordinary body page → `figure-crop-unavailable` (structural error, exit 11).

Publication identity uses `asset_sha256`, not path or `asset_id`.

---

# Appendix F — TOC and metadata (M3)

TOC state machine (architecture §13.1):

```text
AWAITING → COLLECTING → COMMITTED → FINALIZED
```

- First TOC page enters COLLECTING; commit validates hierarchy before accepting.
- Later TOC after FINALIZED → `unexpected-toc`.
- Chapter openings compared to next expected TOC entry (normalized title); mismatch → `toc-chapter-mismatch`.

Metadata merge policy:

- Title page / copyright metadata proposals collected during extraction.
- `final_publication_metadata` selected once at render time.
- Tie-breaking (deterministic):
  1. Higher confidence.
  2. Title-page proposal before copyright-page proposal.
  3. Earlier source page.
  4. Earlier proposal order.
- Hashed in `PublicationDocument.metadata`; per-page proposals excluded from identity.

---

# Appendix G — Footnotes (M5)

- Inline markers as `footnote_reference` segments in wire response.
- Same-page note bodies validated when `require_same_page_footnotes=True`.
- Document-end definitions in Markdown (`[^n]: ...`).
- Publication identity: footnote refs by **document order** (1, 2, 3…); internal `note_id` excluded.
- Visible rendered labels included when they appear in output.

---

# Appendix H — Recovery sequence

On startup / before processing:

```text
1. Read head.json; validate format.
2. For page-0001 … page-{committed_page_count:04d}: validate_commit().
3. Allocate recovery/recovery-NNNN/.
4. Quarantine commits/.page-*.tmp/ into incident.
5. Quarantine orphan commits > committed_page_count.
6. Missing/invalid accepted commit → invalid-run-layout.
7. Rebuild state.json from accepted commits.
8. Quarantine stale .output-build/.
```

Commit write:

```text
write commits/.page-NNNN.tmp/ (complete manifest + all files)
rename → commits/page-NNNN/
write head.json.tmp → os.replace(head.json)
```

Do not auto-advance head to orphan commits in v0.1.
