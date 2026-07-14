# bookextract

Local-first tool for converting scanned book PDFs to structured Markdown and EPUB using a local VLM (llama.cpp).

## Requirements

- Python 3.11+
- [uv](https://github.com/astral-sh/uv) for dependency management
- llama-server (llama.cpp) with a vision model
- Pandoc and EPUBCheck (for EPUB output)

## Install

```bash
uv sync --extra dev
```

## Usage

```bash
bookextract init book.pdf --run runs/mybook --config config.toml --model /path/to/model.gguf
bookextract process --run runs/mybook
bookextract render-markdown --run runs/mybook
bookextract render-epub --run runs/mybook
```

See [docs/plans/implementation_plan_v0.1.md](docs/plans/implementation_plan_v0.1.md) for the normative v0.1 specification.
