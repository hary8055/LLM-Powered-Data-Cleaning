import streamlit as st
import pandas as pd
import re
import google.generativeai as genai

# ---------- CONFIG ----------
st.set_page_config(page_title="LLM-Powered Data Cleaner", layout="wide")
genai.configure(api_key=st.secrets["GEMINI_API_KEY"])

# ---------- SESSION STATE ----------
if "prompt_ready" not in st.session_state:
    st.session_state.prompt_ready = False
if "edited_prompt" not in st.session_state:
    st.session_state.edited_prompt = ""
if "cleaned_df" not in st.session_state:
    st.session_state.cleaned_df = None
if "last_reply" not in st.session_state:
    st.session_state.last_reply = ""

# ---------- MODEL ----------
model = genai.GenerativeModel("gemini-1.5-flash")

# ---------- CLEANING FUNCTION ----------
def clean_llm_lines(reply):
    lines = reply.strip().split("\n")
    cleaned = []
    for line in lines:
        line = line.strip()
        if not line or len(line) > 100:
            continue
        cleaned_line = re.sub(r"^[-•\d]+\.*\s*", "", line)
        if cleaned_line:
            cleaned.append(cleaned_line)
    return cleaned

# ---------- UI ----------
st.title("🧹 LLM-Powered Data Cleaning Assistant")

uploaded_file = st.file_uploader("📁 Upload your CSV file", type=["csv"])

if uploaded_file:
    df = st.session_state.cleaned_df.copy() if st.session_state.cleaned_df is not None else pd.read_csv(uploaded_file)

    st.subheader("📊 Data Preview")
    st.dataframe(df.head(), use_container_width=True)

    use_full_csv = st.checkbox("📄 Use up to 150 Rows in Prompt", value=True)
    data_sample = df.head(150) if use_full_csv else df.head(10)
    data_csv = data_sample.to_csv(index=False)

    task = st.selectbox("🛠️ Select Cleaning Task", [
        "Schema Inference",
        "Impute Missing",
        "Normalize Column",
        "Outlier Detection",
        "Generate Cleaning Script",
        "Duplicate Detection",
        "Semantic Column Renaming",
        "Error Explanation & Justification",
        "Multi-Column Consistency Check",
        "Natural Language Data Query"
    ])

    task_mode = st.radio("🧠 Choose task mode:", ["Suggest strategy", "Output values for application"])

    col1 = st.selectbox("Column 1", df.columns)
    col2 = None
    if task == "Multi-Column Consistency Check":
        col2 = st.selectbox("Column 2", df.columns[df.columns != col1])
    query_text = ""
    if task == "Natural Language Data Query":
        query_text = st.text_input("Ask your question about the data")

    null_like_values = ["UNKNOWN", "N/A", "ERROR", "", None]
    is_null = data_sample[col1].isin(null_like_values) | data_sample[col1].isnull()
    missing_sample = data_sample[is_null]
    missing_indices = missing_sample.index
    missing_count = len(missing_indices)

    selected_data = data_sample[[col1]].to_csv(index=False)
    multi_col_data = data_sample[[col1, col2]].dropna().to_csv(index=False) if col2 else data_csv

    # ---------- PROMPT GENERATION ----------
    prompt = ""
    if task == "Schema Inference":
        prompt = f"Given the dataset below, infer a JSON schema:\n{data_csv}"
    elif task == "Impute Missing":
        if task_mode == "Suggest strategy":
            prompt = f"The column '{col1}' has missing values. Suggest suitable imputation strategies and explain their pros and cons using this sample:\n{selected_data}"
        else:
            prompt = (
                f"The column '{col1}' has {missing_count} missing entries in the sample below.\n"
                f"ONLY return exactly {missing_count} imputed values — one per line. No explanations, no labels.\n\n"
                f"{selected_data}"
            )
    elif task == "Normalize Column":
        prompt = f"Normalize values in column '{col1}':\n{selected_data}"
    elif task == "Outlier Detection":
        prompt = f"Detect outliers in column '{col1}':\n{selected_data}"
    elif task == "Generate Cleaning Script":
        prompt = f"Write a Python script to clean the dataset:\n{data_csv}"
    elif task == "Duplicate Detection":
        prompt = f"Find duplicate values in column '{col1}':\n{selected_data}"
    elif task == "Semantic Column Renaming":
        prompt = f"Suggest better column names for:\n{data_csv}"
    elif task == "Error Explanation & Justification":
        prompt = f"Explain errors and how to fix them in column '{col1}':\n{selected_data}"
    elif task == "Multi-Column Consistency Check" and col2:
        prompt = f"Check consistency between '{col1}' and '{col2}':\n{multi_col_data}"
    elif task == "Natural Language Data Query" and query_text:
        prompt = f"Use this dataset to answer: '{query_text}'\n\n{data_csv}"

    if st.button("🚀 Generate Prompt"):
        st.session_state.prompt_ready = True
        st.session_state.edited_prompt = prompt

    if st.session_state.prompt_ready:
        st.markdown("### ✏️ Review and Edit Prompt")
        st.session_state.edited_prompt = st.text_area("Edit prompt before running:", value=st.session_state.edited_prompt, height=300)

        if st.button("✅ Confirm and Run LLM"):
            with st.spinner("🤖 LLM is generating..."):
                response = model.generate_content(st.session_state.edited_prompt)
                reply = response.text
                st.session_state.last_reply = reply

            st.subheader("💡 LLM Suggestion")
            st.code(reply, language="python" if "script" in reply else "markdown")

            cleaned_values = clean_llm_lines(reply)

            if task == "Impute Missing" and task_mode == "Output values for application":
                if len(cleaned_values) < missing_count:
                    st.warning(f"⚠️ Only {len(cleaned_values)} usable values found. Expected {missing_count}.")
                else:
                    df_updated = df.copy()
                    for i, idx in enumerate(missing_indices):
                        df_updated.at[idx, col1] = cleaned_values[i]
                    st.session_state.cleaned_df = df_updated
                    st.success("✅ Imputed values applied.")
            elif task in ["Normalize Column", "Duplicate Detection", "Outlier Detection"] and task_mode == "Output values for application":
                if len(cleaned_values) != len(data_sample[col1]):
                    st.warning("⚠️ Number of cleaned values does not match sample size.")
                else:
                    df_updated = df.copy()
                    for i, idx in enumerate(data_sample[col1].index):
                        df_updated.at[idx, col1] = cleaned_values[i]
                    st.session_state.cleaned_df = df_updated
                    st.success("✅ Cleaned values applied.")

            if st.session_state.cleaned_df is not None:
                st.markdown("### 👀 Updated Data Preview")
                st.dataframe(st.session_state.cleaned_df.head(), use_container_width=True)
                st.download_button("📥 Download Updated CSV", data=st.session_state.cleaned_df.to_csv(index=False), file_name="cleaned_output.csv")

            st.session_state.prompt_ready = False
