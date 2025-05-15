import streamlit as st
import pandas as pd
import re
import google.generativeai as genai

#config
st.set_page_config(page_title="LLM-Powered Data Cleaner", layout="wide")
genai.configure(api_key=st.secrets["GEMINI_API_KEY"])
model = genai.GenerativeModel("gemini-1.5-flash")

for key in ("original_df", "updated_df", "logs"):
    if key not in st.session_state:
        st.session_state[key] = None if key.endswith("_df") else []
for key in ("prompt_ready", "edited_prompt", "last_reply", "last_cleaned", "last_task"):
    if key not in st.session_state:
        st.session_state[key] = None

# helper
def clean_identifier_fields(df):
    df = df.copy()
    logs = []
    if "Transaction ID" in df:
        m = df["Transaction ID"].isnull().sum()
        df["Transaction ID"] = df["Transaction ID"].fillna(method="ffill")
        df["Transaction ID"] = df["Transaction ID"].fillna(lambda x: f"TXN{1000 + x.index}")
        logs.append(f"Filled {m} missing Transaction IDs.")
    if "Customer Name" in df:
        m = df["Customer Name"].isnull().sum()
        df["Customer Name"] = df["Customer Name"].fillna("Unknown")
        logs.append(f"Filled {m} missing Customer Names.")
    if "Email" in df:
        m = df["Email"].isnull().sum()
        def fix_email(e):
            if pd.isna(e) or e.strip() == "":
                return "missing@example.com"
            if "@" not in e and "." in e:
                return e.replace(" ", "") + "@unknown.com"
            return e if re.match(r"[^@\s]+@[^@\s]+\.[^@\s]+", e) else "invalid@example.com"
        df["Email"] = df["Email"].apply(fix_email)
        logs.append(f"Cleaned {m} missing/malformed Emails.")
    if "Phone" in df:
        m = df["Phone"].isnull().sum()
        df["Phone"] = df["Phone"].fillna("000-000-0000")
        logs.append(f"Filled {m} missing Phone numbers.")
    if "Address" in df:
        m = df["Address"].isnull().sum()
        df["Address"] = df["Address"].fillna("Missing Address")
        logs.append(f"Filled {m} missing Addresses.")
    return df, logs

def clean_llm_lines(reply):
    text = reply.strip()
    if text.startswith("```") and text.endswith("```"):
        text = text[3:-3].strip()
    lines = text.splitlines()
    cleaned = []
    for ln in lines:
        ln = ln.strip()
        if not ln:
            continue
        ln = re.sub(r"^(\d+[\.\)]\s*|[-•\*]\s*)", "", ln)
        cleaned.append(ln)
    return cleaned

#ui
st.title("🧹 LLM-Powered Data Cleaner")

uploaded = st.file_uploader("📁 Upload your CSV file", type="csv")
if uploaded and st.session_state.original_df is None:
    df0 = pd.read_csv(uploaded)
    st.session_state.original_df = df0
    st.session_state.updated_df = df0.copy()
    st.session_state.logs = []
    st.success("File uploaded!")

if st.session_state.updated_df is not None:
    df = st.session_state.updated_df
else:
    df = st.session_state.original_df

if df is not None:
    st.subheader("🔍 Data Preview (first 150 rows)")
    st.dataframe(df.head(150), use_container_width=True)

    st.subheader("🧬 Fix Identifiers")
    if st.button("🧪 Run Identifier Fix"):
        fixed, new_logs = clean_identifier_fields(df)
        st.session_state.updated_df = fixed
        st.session_state.logs.extend(new_logs)
        st.success("Identifier fixes applied!")
        df = fixed

    st.subheader("✅ Review Fixed Data")
    st.dataframe(df.head(150), use_container_width=True)

    st.subheader("🛠️ Choose Cleaning Task")
    task = st.selectbox("Task", [
        "Impute Missing", "Normalize Column", "Outlier Detection",
        "Duplicate Detection", "Schema Inference", "Generate Cleaning Script"
    ])

    st.subheader("🧠 Mode")
    mode = st.radio("Mode", ["Suggest strategy", "Output values for application"])

    col1 = st.selectbox("Select Column", df.columns)
    sample = df.head(150)
    is_null = sample[col1].isnull() | sample[col1].isin(["", "NA", "UNKNOWN"])
    missing_idxs = sample[is_null].index.tolist()
    missing_count = len(missing_idxs)

    prompt = ""
    if task == "Impute Missing":
        if mode == "Suggest strategy":
            prompt = (
                f"You are a data-cleaning assistant. The column '{col1}' has missing or malformed entries "
                f"(null, empty, 'NA', 'UNKNOWN'). Suggest strategies to impute only those missing values. "
                f"Do not alter entries that appear valid or match the data type."
                f"Sample:{sample[[col1]].to_csv(index=False)}"
            )
        else:
            prompt = (
                f"You are a data-cleaning assistant. The column '{col1}' contains {missing_count} missing entries. "
                f"Return only the imputed values for these null rows, maintaining the value format seen in valid rows. "
                f"For numeric columns, keep imputation statistically consistent. No explanation. One value per line."
                f"{sample[[col1]].to_csv(index=False)}"
            )
    elif task == "Normalize Column":
        if col1.lower() == "payment method":
            if mode == "Suggest strategy":
                prompt = (
                    f"The column '{col1}' contains inconsistent payment method labels (e.g. 'cc', 'CREDIT CARD', 'Paypal', 'PayPal '). "
                    f"Propose a mapping from known variants to standard labels like 'Credit Card', 'PayPal', 'Bank Transfer', etc. "
                    f"Do not suggest changing valid entries already matching these."
                    f"Sample:{sample[[col1]].to_csv(index=False)}"
                )
            else:
                prompt = (
                    f"Normalize only incorrect or inconsistent values in '{col1}' based on known payment types. "
                    f"Keep valid standard forms (e.g. 'Credit Card', 'PayPal') unchanged. Output one cleaned value per row. No explanation."
                    f"{sample[[col1]].to_csv(index=False)}"
                )
        else:
            if mode == "Suggest strategy":
                prompt = (
                    f"You are a data-cleaning assistant. The column '{col1}' contains inconsistently formatted entries. "
                    f"Propose a normalization strategy that transforms only the inconsistent ones. Leave valid entries untouched."
                    f"Sample:{sample[[col1]].to_csv(index=False)}"
                )
            else:
                prompt = (
                    f"In column '{col1}', normalize inconsistent values only. Return one value per row. Preserve already standard entries. "
                    f"Do not modify already valid categories. No explanation."
                    f"{sample[[col1]].to_csv(index=False)}"
                )
    elif task == "Outlier Detection":
        prompt = (
            f"You are a data-cleaning assistant. Detect **only** true outliers in '{col1}'—"
            f"values that are statistically anomalous or violate domain rules. No valid extremes. Return one per line."
            f"{sample[[col1]].to_csv(index=False)}"
        )
    elif task == "Duplicate Detection":
        prompt = (
            f"You are a data-cleaning assistant. Find **exact** duplicate rows based on '{col1}'. "
            f"Return row indices of duplicates, one per line, no explanation."
            f"{sample[[col1]].to_csv(index=False)}"
        )
    elif task == "Schema Inference":
        prompt = (
            f"You are a data schema assistant. Infer a JSON schema for the dataset fields and types."
            f"{sample.to_csv(index=False)}"
        )

    if st.button("🚀 Generate Prompt"):
        st.session_state.prompt_ready = True
        st.session_state.edited_prompt = prompt

    if st.session_state.prompt_ready:
        st.subheader("✏️ Review & Edit Prompt")
        st.session_state.edited_prompt = st.text_area(
            "", st.session_state.edited_prompt, height=200
        )
        if st.button("✅ Run LLM Task"):
            with st.spinner("🤖 Running LLM..."):
                reply = model.generate_content(st.session_state.edited_prompt).text
            st.subheader("💡 LLM Output")
            st.code(reply)
            cleaned = clean_llm_lines(reply)
            st.session_state.last_reply = reply
            st.session_state.last_cleaned = cleaned
            st.session_state.last_task = {
                "type": task,
                "mode": mode,
                "column": col1,
                "indexes": missing_idxs if task == "Impute Missing" else sample.index.tolist()
            }
            st.success("LLM output received. Scroll down to review & apply.")

if st.session_state.last_cleaned:
    task = st.session_state.last_task["type"]
    col = st.session_state.last_task["column"]
    idxs = st.session_state.last_task["indexes"]
    cleaned = st.session_state.last_cleaned

    st.subheader("🧾 Review Proposed Changes")
    preview = df.copy()
    for i, idx in enumerate(idxs[:len(cleaned)]):
        preview.at[idx, col] = cleaned[i]
    st.dataframe(preview.head(150), use_container_width=True)

    confirm = st.checkbox("✅ I reviewed and confirm apply changes")
    if confirm and st.button("💾 Apply Changes"):
        updated = st.session_state.updated_df.copy() if st.session_state.updated_df is not None else df.copy()
        for i, idx in enumerate(idxs[:len(cleaned)]):
            updated.at[idx, col] = cleaned[i]
        st.session_state.updated_df = updated
        st.session_state.logs.append(f"{task} applied to '{col}'")
        st.success("Changes applied to dataset.")
        st.session_state.last_cleaned = None
        st.session_state.last_task = None

if st.session_state.updated_df is not None:
    st.subheader("📥 Download Cleaned CSV & Logs")
    csv_bytes = st.session_state.updated_df.to_csv(index=False).encode("utf-8")
    st.download_button("⬇️ Download Cleaned CSV", data=csv_bytes, file_name="cleaned_output.csv")
    if st.session_state.logs:
        logs_text = "\n".join(st.session_state.logs)
        st.download_button("📄 Download Logs", data=logs_text, file_name="cleaning_log.txt")