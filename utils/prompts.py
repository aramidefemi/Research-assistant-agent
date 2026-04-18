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

EVALUATE_PROMPT = """You are a research relevance evaluator.

Research Focus:
{research_focus}

Paper Summary:
{summary}

Key Findings:
{key_findings}

Methodology:
{methodology}

Based on the research focus, evaluate how relevant this paper is.

Respond in this exact format:
SCORE: <a number from 0.0 to 1.0>
FIT: <YES or NO — YES if score >= 0.6>
REASON: <2-3 sentence explanation of why this paper does or doesn't fit the research focus>
"""
