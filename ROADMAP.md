# Product Roadmap

This roadmap prioritizes features that increase researcher trust, speed, and workflow integration.

## 4 key feature updates

1. **Fault-tolerant no-LLM mode (with explicit opt-out)**
   - Add a deterministic fallback pipeline (regex/rules-based extraction + scoring heuristics) when LLMs are unavailable.
   - Add a user-facing toggle to disable LLM usage at runtime.
   - Add per-run metadata (`llm_used: true/false`, fallback reason) in outputs and traces.
   - Value: guarantees baseline functionality under outages, quota limits, or privacy-sensitive workflows.

2. **Interactive paper chat in Streamlit**
   - Add a simple chat UI so users can ask follow-up questions after papers are found/scored.
   - Ground answers in extracted paper text and show citation anchors to sections/pages where possible.
   - Keep chat state tied to selected paper(s) and current research focus.
   - Value: moves the app from static scoring to interactive analysis and faster researcher decision-making.

3. **Evidence-first output contract**
   - Enforce that every key claim (summary/evaluation/chat response) maps to source evidence.
   - Require confidence labels and “insufficient evidence” states instead of overconfident output.
   - Value: increases trust and reduces hallucination risk in research workflows.

4. **Research workspace export + reproducibility bundles**
   - Export qualified papers and evaluations to BibTeX/Zotero CSV/structured notes.
   - Include run config (mode, prompts/rules, timestamps, model or fallback path) for reproducibility.
   - Value: shortens path from discovery to writing while preserving auditability.

## 5 nice-to-haves

1. **Side-by-side paper comparison board**
   - Compare 2-4 papers on methods, datasets, limitations, novelty, and reproducibility in one view.

2. **Custom scoring profiles per project**
   - Let users weight criteria (novelty, rigor, domain fit, recency) and save presets by research track.

3. **Methodology and risk flags**
   - Auto-highlight concerns such as small sample size, weak baselines, missing ablations, or unclear evaluation setup.

4. **Multi-format ingestion**
   - Support arXiv URLs, DOI links, and reference lists in addition to PDF uploads.

5. **Continuous discovery watches**
   - Save topics and re-check periodically for new candidate papers with change alerts.

## Feature request template for GitHub Issue or PR

Feature requests are welcome via **GitHub Issues** or **PRs**.
To help us evaluate quickly, copy this template:

```md
## Feature request

### Problem
What researcher pain point are you solving?

### Proposed feature
What should the app do?

### Research workflow impact
How does this improve trust, speed, or quality of research decisions?

### Example use case
Describe a realistic scenario (inputs, expected output).

### Acceptance criteria
- [ ] Clear behavior 1
- [ ] Clear behavior 2
- [ ] Clear behavior 3

### Optional notes
Links to papers, screenshots, or related tools.
```
