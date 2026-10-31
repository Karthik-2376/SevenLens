from dotenv import load_dotenv
load_dotenv()

import os
import re
import time
import streamlit as st
import pipeline
from pipeline import DEPTH


st.set_page_config(page_title="Seven Lens",layout="wide")
missing = [
    key for key in ("MISTRAL_API_KEY", "TAVILY_API_KEY")
    if not os.getenv(key)
]

if missing:
    st.error(f"Missing API keys in .env: {', '.join(missing)}")
    st.stop()

STEPS = {
    1: "Planning research questions",
    2: "Searching the web",
    3: "Reading sources",
    4: "Summarizing sources",
    5: "Writing the first draft",
    6: "Fact-checking the draft",
    7: "Revising the report",
    8: "Critic review",
    9: "Saving the Report"
}

def get_score(feedback):
    match = re.search(r"Score:\s*\**\s*(\d+(?:\.\d+)?)\s*/\s*10",feedback,re.IGNORECASE)
    return float(match.group(1)) if match else None

def run_research(topic, depth):
    original_header = pipeline.header
    start_time = time.time()
    with st.status("Research in progress...", expanded=True) as status:
        progress = st.progress(0)
        def update_progress(message):
            match = re.search(r"STEP\s+(\d+)", message, re.IGNORECASE)
            if match:
                step = int(match.group(1))
                label = STEPS.get(step, "Processing")
                progress.progress(min(step / 9, 1.0))
                status.update(label=f"Step {step}/9: {label}")
                st.write(f"Step {step}/9: {label}")

        pipeline.header = update_progress
        try:
            result = pipeline.run_research_pipeline(topic, depth)
            result["elapsed"] = time.time() - start_time
            progress.progress(1.0)
            status.update(label="Research completed successfully",state="complete",expanded=False)
            return result
        except Exception as error:
            status.update(label="Research failed",state="error",expanded=True)
            st.error(f"Error: {error}")
            return None
        finally:
            pipeline.header = original_header


st.title("SevenLens")
st.subheader("Multi-Agent AI Research and Intelligence System")
st.write(
    "Research any topic using AI agents. "
    "Gather information, verify findings, and generate a research report."
)

st.sidebar.header("Research Settings")

depth = st.sidebar.selectbox(
    "Select research depth",
    ["quick", "standard", "deep"],
    index=1
)

config = DEPTH[depth]

# st.sidebar.write(f"Research questions: {config['questions']}")
# st.sidebar.write(f"Maximum sources: {config['pages']}")

st.sidebar.divider()
st.sidebar.subheader("Research Workflow")

for step, description in STEPS.items():
    st.sidebar.write(f"{step}. {description}")

st.header("Start a New Research")
with st.form("research_form"):
    topic = st.text_input(
        "Enter your research topic",
        placeholder="Example: Impact of generative AI on healthcare diagnostics"
    )

    submitted = st.form_submit_button("Start Research",type="primary")

if submitted:
    if not topic.strip():
        st.warning("Please enter a research topic.")
    else:
        result = run_research(topic.strip(), depth)
        if result:
            st.session_state["result"] = result


if "result" in st.session_state:
    result = st.session_state["result"]
    st.divider()
    st.header("Research Results")
    st.write(f"**Topic:** {result['topic']}")
    score = get_score(result["feedback"])
    col1, col2, col3 = st.columns(3)
    col1.metric("Sources Used", len(result["sources"]))
    col2.metric(
        "Critic Score",
        f"{score}/10" if score is not None else "Not available"
    )

    col3.metric(
        "Execution Time",
        f"{result.get('elapsed', 0):.1f} seconds"
    )

    st.download_button(
        "Download Research Report",
        data=result["report"],
        file_name="seven_lens_report.md",
        mime="text/markdown"
    )

    tab1, tab2, tab3, tab4, tab5 = st.tabs(
        ["Research Report", "Sources", "Fact Check", "Critic Review", "Process"]
    )

    with tab1:
        st.markdown(result["report"])
        if result.get("file"):
            st.caption(f"Saved report: {result['file']}")
    with tab2:
        st.subheader("Sources Used")
        for source in result["sources"]:
            with st.expander(
                f"[{source['id']}] {source['title']}"
            ):
                st.write(f"**URL:** {source['url']}")
                st.write(f"**Source:** {source.get('source', 'Unknown')}")
                st.write(
                    f"**Published:** "
                    f"{source.get('published_date') or 'Not available'}"
                )
                st.write(
                    f"**Evidence type:** {source['evidence_type']}"
                )
                st.write("**Summary:**")
                st.write(source.get("summary", "No summary available"))

                st.link_button(
                    "Open Source",
                    source["url"]
                )

    with tab3:
        st.subheader("Fact Check Results")
        st.markdown(result["facts"])
    with tab4:
        st.subheader("Critic Evaluation")
        st.markdown(result["feedback"])
    with tab5:
        st.subheader("Research Questions")
        for index, question in enumerate(result["questions"], 1):
            st.write(f"{index}. {question}")

        with st.expander("View Initial Draft"):
            st.markdown(result["draft_report"])
else:
    st.info("Enter a research topic above to get started.")
