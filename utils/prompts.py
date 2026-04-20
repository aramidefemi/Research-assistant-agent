SUMMARISE_PROMPT = """You are a research assistant helping a team evaluate academic papers.

Given the following paper text, extract:
1. A concise summary (3-5 sentences)
2. Key findings (bullet points)
3. Methodology used

Paper text:
{pdf_text}

Respond in this exact format:
SUMMARY:
<your summary here>

KEY_FINDINGS:
<bullet point findings here>

METHODOLOGY:
<methodology description here>
"""

EVALUATE_SCORE_FIT_PROMPT = """You are a research relevance evaluator.

Research Focus:
{research_focus}

Paper Summary:
{summary}

Key Findings:
{key_findings}

Methodology:
{methodology}

Based on the research focus, assign a relevance score and fit verdict only.

Respond in this exact format:
SCORE: <a number from 0.0 to 1.0>
FIT: <YES or NO — YES if score >= 0.6>
"""

EVALUATE_REASON_PROMPT = """You are a research relevance evaluator.

Research Focus:
{research_focus}

Paper Summary:
{summary}

Key Findings:
{key_findings}

Methodology:
{methodology}

Already determined (do not contradict):
SCORE: {score:.2f}
FIT: {fit_label}

Explain briefly why this paper does or does not fit the research focus.

Respond in this exact format:
REASON: <2-3 sentences>
"""

EVALUATION_MATRIX_PROMPT = """You are extracting a structured evidence matrix for a research paper.

Research Focus:
{research_focus}

Paper Summary:
{summary}

Key Findings:
{key_findings}

Methodology:
{methodology}

Paper Text Excerpt:
{pdf_text_excerpt}

Extraction policy:
- Use only information supported by the provided content.
- If unknown/unclear, use "N/A".
- Return valid JSON only.

Respond in this exact format:
SOURCE_PROFILE_JSON: <minified JSON object>

The JSON object must contain exactly these keys:
{{
  "authors": "",
  "date_of_research": "",
  "country_of_origin": "",
  "purpose_aims": "",
  "research_questions": "",
  "data_used_method_collection_sample_size": "",
  "methods_tools_used": "",
  "method_and_data_collection_limitations": "",
  "results": "",
  "contribution": "",
  "limitation_of_research_outcomes": "",
  "future_perspectives": ""
}}
"""

DISCOVERY_SCORE_FIT_PROMPT = """You are qualifying candidate journal papers for a student research topic.

Research Topic:
{topic}

Candidate Paper Metadata:
TITLE: {title}
ABSTRACT: {abstract}
VENUE: {venue}
YEAR: {year}
CITED_BY_COUNT: {cited_by_count}

Score the candidate only for topical relevance.

Respond in this exact format:
SCORE: <a number from 0.0 to 1.0>
FIT: <YES or NO>
"""

DISCOVERY_QUALITY_REASON_PROMPT = """You are qualifying candidate journal papers for a student research topic.

Research Topic:
{topic}

Candidate Paper Metadata:
TITLE: {title}
ABSTRACT: {abstract}
VENUE: {venue}
YEAR: {year}
CITED_BY_COUNT: {cited_by_count}

Already determined (do not contradict):
SCORE: {score:.2f}
FIT: {fit_label}

Respond in this exact format:
QUALITY: <YES or NO>
REASON: <2-3 concise sentences>
"""

DISCOVERY_SOURCE_PROFILE_PROMPT = """You are extracting a structured evidence matrix for a journal paper candidate.

Research Topic:
{topic}

Candidate Paper Metadata:
TITLE: {title}
ABSTRACT: {abstract}
VENUE: {venue}
YEAR: {year}
CITED_BY_COUNT: {cited_by_count}

Extraction policy:
- Use only information supported by metadata above.
- If unknown/unclear, use "N/A".
- Return valid JSON only.

Respond in this exact format:
SOURCE_PROFILE_JSON: <minified JSON object>

The JSON object must contain exactly these keys:
{{
  "authors": "",
  "date_of_research": "",
  "country_of_origin": "",
  "purpose_aims": "",
  "research_questions": "",
  "data_used_method_collection_sample_size": "",
  "methods_tools_used": "",
  "method_and_data_collection_limitations": "",
  "results": "",
  "contribution": "",
  "limitation_of_research_outcomes": "",
  "future_perspectives": ""
}}
"""

DISCOVERY_ABSTRACT_TRIAGE_PROMPT = """You are triaging candidate journal papers for a student research topic.

Research Topic:
{topic}

Candidates (index, title, abstract):
{candidates_block}

Task:
1) Read only title + abstract.
2) Rank candidates by likely topical fit and usefulness (best first).
3) If all candidates look weak/off-topic, ask for a refetch.

Respond in this exact format:
ORDER: <comma-separated 0-based indexes in best-first order, e.g. 2,0,1>
REFETCH: <YES or NO>
REASON: <1-2 concise sentences>
"""
