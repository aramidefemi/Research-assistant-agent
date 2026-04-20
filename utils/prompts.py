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

Now assess only scholarly quality and explain briefly.

Respond in this exact format:
QUALITY: <YES or NO>
REASON: <2-3 concise sentences>
"""
