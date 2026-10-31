from dotenv import load_dotenv
load_dotenv()

from langchain.agents import create_agent
from langchain_mistralai import ChatMistralAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from tools import tavily_search, scrape_url

llm = ChatMistralAI(model="ministral-8b-latest", temperature=0)
parser = StrOutputParser()

planner_prompt = ChatPromptTemplate.from_messages([
    ("system", "You are an expert research planner. Create focused, diverse and specific research questions."),
    ("human", """Break the topic below into exactly {n} research questions.

Topic: {topic}

Rules:
- Cover different aspects such as background, current developments, evidence, practical applications, risks and future outlook where relevant.
- Avoid duplicate or overly broad questions.
- Make questions specific enough to find reliable sources.
- Return one question per line, numbered 1., 2., 3. and nothing else.""")
])
planner_chain = planner_prompt | llm | parser

def build_search_agent():
    return create_agent(
        model=llm,
        tools=[tavily_search],
        system_prompt="""You are a web research search agent.
Always call the tavily_search tool exactly once for the user's research query.
Search for relevant, reliable and recent sources.
Prefer original research papers, government reports, universities and established organizations.
Look for sources that directly answer the query, not only sources discussing the general topic.
Do not invent URLs, titles, dates or source information.
After the tool call, reply with only the word Done."""
    )

def build_reader_agent():
    return create_agent(
        model=llm,
        tools=[scrape_url],
        system_prompt="""You are a web reader agent.
Always call the scrape_url tool exactly once for the URL provided by the user.
Do not invent, summarize or assume article content.
After the tool call, reply with only the word Done."""
    )

summarizer_prompt = ChatPromptTemplate.from_messages([
    ("system", """You are an expert research analyst.
Extract useful information from the supplied webpage or PDF content.
Preserve the original meaning and do not add unsupported information."""),
    ("human", """Summarize the following extracted webpage or PDF content for research purposes.

Article:
{article}

Return:
1. Main Topic
2. Five Important Findings
3. Important Statistics and Dates
4. Supporting Evidence
5. Risks or Limitations

Rules:
- Preserve important numbers, percentages and dates accurately.
- Do not invent facts, statistics, dates or quotations.
- Distinguish facts from opinions.
- Mention the source's limitations when available.
- Keep the summary within 300 words.
- If information is missing, mention it.""")
])
summarizer_chain = summarizer_prompt | llm | parser

fact_prompt = ChatPromptTemplate.from_messages([
    ("system", """You are a strict fact-checking research analyst.
Your task is to verify claims made in a research report against the original source evidence supplied to you.
Do not trust the report, its citations or its wording automatically.
Never use outside knowledge or assumptions."""),
    ("human", """Research report to verify:
{report}

Original numbered source evidence:
{sources}

Your tasks:
1. Identify up to 15 important factual claims in the report.
2. Check each claim against the original content of the source number cited for that claim.
3. Check whether the cited source actually supports the exact statement.
4. Check every statistic, percentage, date, comparison and named study carefully.
5. If a claim has no valid citation, mark it Unverifiable.
6. If the cited source only partly supports the claim, mark it Partially Supported.
7. If the source directly disagrees with the claim, mark it Contradicted.
8. Do not use one source to claim that another source supports something.
9. Identify claims that appear in the report but cannot be found in the supplied evidence.
10. Identify disagreements between sources.

Use only these four statuses:
Supported
Partially Supported
Contradicted
Unverifiable

Respond in this exact format, one claim per line:

Claim | cited source number | status | evidence or reason

At the end provide:

Unsupported or missing citations:
- ...

Conflicts:
- ...

Important:
- Supported means directly supported by the supplied source content, not independently proven to be true.
- Clearly identify when evidence comes from a search snippet rather than extracted webpage or PDF content.
- Do not treat a source title as evidence.
- Do not invent statistics, dates, quotations, studies or evidence.
- If a source does not contain the information, say so.
- Keep the evidence concise and faithful to the original source.""")
])
fact_chain = fact_prompt | llm | parser

writer_prompt = ChatPromptTemplate.from_messages([
    ("system", """You are an expert research writer.
Write a detailed, structured and factual research report using only the supplied research material.
Never invent facts, statistics, studies, dates, quotations or references."""),
    ("human", """Write an initial research report on the topic below.

Topic:
{topic}

Research Questions:
{questions}

Research Gathered:
{research}

Structure the report as:
1. Title
2. Introduction
3. Key Findings (minimum 3 well-explained points)
4. Conflicting or Uncertain Points
5. Conclusion

Rules:
- Cite factual statements using the source numbers provided, such as [1] or [2][3].
- Use only information present in the supplied research.
- Every factual claim must have a source citation.
- Do not invent statistics, percentages, dates, studies or source details.
- Do not use general knowledge to fill gaps in the evidence.
- Do not create unsupported calculations or comparisons.
- Clearly state when the available sources do not provide enough evidence.
- Distinguish search snippets from extracted webpage or PDF evidence.
- Mention publication dates only when supplied.
- Do not write a References section. It will be added automatically.
- Write in a detailed, professional and easy-to-understand style.""")
])
writer_chain = writer_prompt | llm | parser

revision_prompt = ChatPromptTemplate.from_messages([
    ("system", """You are an expert research editor.
Revise research reports using strict fact-checking results.
Accuracy and traceable evidence are more important than report length."""),
    ("human", """Revise the initial research report using the fact-checking results.

Topic:
{topic}

Initial Report:
{draft}

Fact Check Results:
{facts}

Original Research:
{research}

Rules:
- Correct or remove every Contradicted claim.
- Remove or clearly qualify every Unverifiable claim.
- Rewrite Partially Supported claims so they state only what the evidence supports.
- Keep Supported claims only when their cited source actually supports them.
- Remove unsupported statistics, percentages, dates, studies and comparisons.
- Never invent replacement evidence or citations.
- Do not use general knowledge to fill evidence gaps.
- Preserve useful explanations that are supported by the sources.
- Keep source numbers consistent with the supplied research.
- Include an Uncertainties section if important evidence is missing or conflicting.
- Do not include the References section. It will be added automatically.
- Return only the complete revised report.""")
])
revision_writer_chain = revision_prompt | llm | parser

critic_prompt = ChatPromptTemplate.from_messages([
    ("system", """You are a strict and constructive research report evaluator.
Evaluate the revised report using the supplied source evidence and fact-checking results.
Do not assume a citation is correct just because a source number appears in the report."""),
    ("human", """Review the revised research report below.

Number of sources available: {source_count}

Report:
{report}

Fact Check Results:
{facts}

Original Source Evidence:
{sources}

Evaluate:
1. Factual accuracy
2. Source relevance and coverage
3. Citation quality
4. Evidence supporting important claims
5. Conflicting or uncertain information
6. Report structure and readability

For important factual claims:
- Check whether the cited source actually supports the statement.
- Identify unsupported statistics, dates and percentages.
- Identify citations that do not support the associated statement.
- Check whether fact-checking corrections were applied.
- Check whether uncertain claims are presented as established facts.
- Do not accept general knowledge as source evidence.

Respond in this exact format:

Score: X/10

Source Coverage: X of {source_count} sources cited

Strengths:
- ...

Areas to Improve:
- ...

One line verdict:
...""")
])
critic_chain = critic_prompt | llm | parser