# Local BMAD specification workflow

Selected `bmad-spec` resources and shared helpers were copied without modification from a local BMAD 6.11.0 installation. File hashes and upstream attribution are in [provenance.json](provenance.json). Project configuration was authored for Lynceus; no settings or project content from the source project were imported.

Upstream: [BMad Method](https://github.com/bmad-code-org/BMAD-METHOD). License: [MIT](LICENSE).

This is not a full installer-managed BMAD deployment and does not register editor slash commands. An assistant can load [the skill](skills/bmad-spec/SKILL.md) directly. The shared Python helpers work with Python 3.11+ or `uv run --offline` with an available interpreter.

The first run used headless express extraction from the supplied conversation. It produced a kernel and delivery companions; `stories.yaml` was not generated because this skill reserves dispatch story breakdown for interactive checkpoint selection. The delivery work packages remain implementable without dispatch metadata.
