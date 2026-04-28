# Product Roadmap

This roadmap prioritizes features that increase researcher trust, speed, and workflow integration.

## 5 key feature updates

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

3. **Methodology and risk flags**
   - Auto-highlight concerns such as small sample size, weak baselines, missing ablations, or unclear evaluation setup.
   - Tag each flag with short evidence snippets so users can inspect the source quickly.
   - Value: helps researchers avoid weak evidence and improves paper selection quality.

4. **Citation-use examples (practical research application guidance)**
   - For each relevant paper, generate at least 3 concrete ways the user can use it in their work.
   - Cover distinct angles: method reuse, result benchmarking, and limitation-based research gaps.
   - Example: for a credit scoring paper, suggest how to adopt the method, compare against its reported outcomes, or extend a stated limitation.
   - Value: answers “how can I use this?” not just “is this relevant?”, turning relevance into actionable research steps.

5. **Evidence-first output contract**
   - Enforce that every key claim (summary/evaluation/chat response) maps to source evidence.
   - Require confidence labels and “insufficient evidence” states instead of overconfident output.
   - Value: increases trust and reduces hallucination risk in research workflows.

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
