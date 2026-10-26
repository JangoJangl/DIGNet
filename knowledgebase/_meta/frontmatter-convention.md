---
title: YAML frontmatter convention
type: meta
status: active
created: 2026-10-25
updated: 2026-10-25
tags: [meta, convention]
related: ["[[_templates/note-template]]"]
---

# YAML frontmatter convention

Every note starts with a YAML frontmatter block. Obsidian shows it as *Properties*, and it can be queried with the Dataview community plugin.

## Fields

| Field | Required | Type | Allowed values / format | Meaning |
|---|---|---|---|---|
| `title` | yes | string | free text | Human readable title (the file name is the slug) |
| `type` | yes | enum | `moc`, `state`, `problem-log`, `idea-log`, `research`, `roadmap`, `reference`, `changelog`, `meta`, `template` | What kind of note this is. `moc` = map of content / index |
| `status` | yes | enum | `draft`, `active`, `done`, `obsolete` | Lifecycle of the note |
| `created` | yes | date | `YYYY-MM-DD` | Creation date |
| `updated` | yes | date | `YYYY-MM-DD` | Last meaningful content change |
| `tags` | yes | list | lowercase kebab-case, no `#` | Topic tags, e.g. `point-cloud`, `diffusion`, `sim-to-real` |
| `related` | no | list of wikilinks | `"[[path/note]]"` (quoted) | Notes that should be read together |
| `sources` | no | list | repo paths (`diffusion/training.py`) or URLs | Where the content comes from |
| `confidence` | research notes | enum | `high`, `medium`, `low` | How well the claims are verified. `high` = read the primary source or the code, `medium` = secondary source or summary, `low` = recollection, needs checking |
| `effort` | ideas / roadmap | string | `S` (< 0.5 day), `M` (0.5 to 2 days), `L` (2 to 5 days) | Rough effort estimate |

## Rules

1. Dates are ISO 8601 (`2026-10-25`). Update `updated` whenever content changes.
2. Wikilinks inside YAML are quoted strings: `related: ["[[state/problems]]"]`.
3. Tags are lowercase kebab-case without `#` in the frontmatter.
4. One note per subject. Problems and ideas are kept in one log note each ([[state/problems]], [[state/ideas]]) with stable IDs (`P-01`, `I-01`) so that other notes can link to a heading, e.g. `[[state/problems#P-03 Batch size is forced to one grasp]]`.
5. Code references use repo-relative paths with an optional line: `diffusion/networks/diffusion/point_diffusion.py:340`.
6. Research claims carry a `confidence` and a source link. Anything that has not been checked against a primary source is marked `low`.

## Example

```yaml
---
title: Data pipeline
type: state
status: active
created: 2026-10-25
updated: 2026-10-25
tags: [data, acronym, point-cloud]
related: ["[[state/architecture]]", "[[state/problems]]"]
sources: [diffusion/environments/diffusion_grasp/dataloader.py]
---
```

## Dataview example

```dataview
TABLE type, status, updated FROM "" WHERE type = "research" SORT updated DESC
```
