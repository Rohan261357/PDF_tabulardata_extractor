
import streamlit as st
import pandas as pd
import pdfplumber
import json
import re
from openai import OpenAI

# Function to extract text from PDF
def pdf_to_llm_input(pdf_file):
    with pdfplumber.open(pdf_file) as pdf:
        text = ''
        for page in pdf.pages:
            page_text = page.extract_text()
            if page_text:
                text += page_text + '\n'
    return text

# Function to extract tables using OpenAI
def llm_table_extraction(pdf_text, api_key):
    client = OpenAI(api_key=api_key)

    prompt = f"""ANALYZE THIS PDF CONTENT AND RETURN ALL TABLES IN PROPER JSON FORMAT:
{pdf_text}

STRICT FORMATTING RULES:
1. Output MUST be valid JSON only
2. Use this exact structure:
{{
  "tables": [
    {{
      "title": "Table Title",
      "columns": ["Column1", "Column2"],
      "rows": [
        ["Row1Val1", "Row1Val2"],
        ["Row2Val1", "Row2Val2"]
      ]
    }}
  ]
}}
3. Preserve ALL original values exactly
4. Escape special characters properly
5. Ensure JSON is properly closed"""

    response = client.chat.completions.create(
        model="gpt-4",
        messages=[
            {"role": "system", "content": "You are a JSON formatting expert. Only return valid JSON."},
            {"role": "user", "content": prompt}
        ],
        temperature=0
    )

    raw_output = response.choices[0].message.content
    json_str = re.sub(r'^```json\n?|\n```$', '', raw_output, flags=re.IGNORECASE)

    try:
        data = json.loads(json_str)
        if "tables" not in data:
            raise ValueError("Missing 'tables' key in JSON response")
        return data
    except (json.JSONDecodeError, ValueError) as e:
        st.error(f"JSON Error: {str(e)}")
        st.text_area("Raw LLM Response", raw_output)
        return None

# Function to convert tables to DataFrames
def llm_to_dataframes(llm_output):
    dfs = []
    for idx, table in enumerate(llm_output["tables"], start=1):
        df = pd.DataFrame(table["rows"], columns=table["columns"])
        dfs.append((table["title"], df))
    return dfs

# Streamlit App
st.set_page_config(page_title="PDF Table Extractor", layout="wide")
st.title("📄 PDF Table Extractor with LLM")

api_key = st.text_input("Enter your OpenAI API Key", type="password")
pdf_file = st.file_uploader("Upload PDF File", type="pdf")

if pdf_file and api_key:
    with st.spinner("Extracting text from PDF..."):
        pdf_text = pdf_to_llm_input(pdf_file)

    if st.button("Extract Tables"):
        with st.spinner("Calling LLM to extract tables..."):
            llm_output = llm_table_extraction(pdf_text, api_key)

        if llm_output:
            dataframes = llm_to_dataframes(llm_output)
            st.success(f"Extracted {len(dataframes)} tables!")

            for title, df in dataframes:
                st.subheader(title)
                st.dataframe(df)
                csv = df.to_csv(index=False).encode('utf-8')
                st.download_button(
                    label="Download CSV",
                    data=csv,
                    file_name=f"{title.replace(' ', '_')}.csv",
                    mime="text/csv"
                )







