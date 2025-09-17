#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
CV Parser (local script)
- Removes Flask API components
- Uses OpenAI instead of IBM watsonx
- Hardcoded variables in main instead of CLI args

Env vars required:
    OPENAI_API_KEY  # your OpenAI key
"""
import os
import re
import json
from pathlib import Path
from random import randint

# --------- File extraction deps ---------
from pdfminer.high_level import extract_text
import docx2txt
# import textractplus as tp

# --------- OpenAI client ---------
try:
    # OpenAI Python SDK (v1.x)
    from openai import OpenAI
    _USE_RESPONSES_API = False
except Exception:
    OpenAI = None

# ---------------- Utilities ----------------
ProgramDefinition = """
Created on Thu May 2 2024 15:43:00 

@author: OmarDasser

This script accepts a file path to a CV (in PDF, DOCX, or PPTX format)
and parses it into a JSON object containing distinct sections such as "Assignment History,"
"Education," "Work Experience," and "Key Skills." The program processes the document to
extract and categorize its content, and writes the structured JSON representation to disk.
""".strip()


def normalize_text_word(text: str) -> str:
    text = text.replace('\n\n\n', '\n')
    text = text.replace('\n\n', '\n')
    text = text.replace('\n\n', '\n')
    text = text.replace('\\', '/')
    return text


def normalize_text(text: str) -> str:
    text = re.sub(r'\f', '', text)
    text = re.sub(r'\n{2,}', '\n', text)
    text = re.sub(r'\s*\n\s*', '\n', text)
    url_pattern = r'https?://\S+|www\.\S+'
    text = re.sub(url_pattern, '', text)
    text = re.sub(r'[\[\]]', '', text)
    text = re.sub('\n', '', text)
    text = re.sub('\W+', '', text)
    text = text.strip()
    return text


def extract_text_from_pdf(filename: str) -> str:
    text = extract_text(filename)
    return normalize_text(text)


def extract_text_from_word(filename: str) -> str:
    text = docx2txt.process(filename)
    return normalize_text_word(text)


# def extract_text_from_ppt(filename: str) -> str:
#     text = tp.process(filename).decode('utf-8')
#     return normalize_text_word(text)


def parse_file(filename: str) -> str:
    ext = filename.split('.')[-1].lower()
    if ext == 'pdf':
        return extract_text_from_pdf(filename)
    elif ext in ('doc', 'docx'):
        return extract_text_from_word(filename)
    # elif ext in ('ppt', 'pptx'):
    #     return extract_text_from_ppt(filename)
    else:
        raise ValueError(f"Unsupported file extension: {ext}")


# ---------------- OpenAI call ----------------
OPENAI_PROMPT_TEMPLATE = """
Analyze the CV provided and extract the following information into a structured JSON format: 
Name: The candidate's full name; 
Profile: A brief summary of the 'Profile' section where the candidate introduces themselves; 
Education: List degrees and institutions, including degree earned and institution name; 
Languages: Languages the candidate speaks and the level; 
Key Skills: List of key skills mentioned as simple titles; 
Key Courses and Training: Relevant courses or training as simple titles; 
Digital Credentials: Any digital credentials or badges earned as simple titles (Include this category only when it exists in the CV); 
Work Experience: For each role, include Position, Company, Duration, and a summarized description of contributions and achievements; 
Assignment History: Details of projects including Project name, Role, Client, Duration, and a summary of contributions. 
Ensure each category is accurately filled with relevant information extracted from the CV, with summaries for 'Profile', work experience, and assignment contributions being concise. 
Output this information exclusively in JSON format, without any introductory or concluding remarks.

Input: {normalized_text}
""".strip()


def call_openai(normalized_text: str, model: str = "gpt-4o-mini") -> str:
    if OpenAI is None:
        raise RuntimeError("OpenAI SDK is not installed. Run: pip install openai>=1.0.0")

    client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))

    prompt = OPENAI_PROMPT_TEMPLATE.format(normalized_text=normalized_text)

    resp = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": "You are a precise information extraction engine. Respond ONLY with valid JSON."},
            {"role": "user", "content": prompt},
        ],
        response_format={"type": "json_object"},
        temperature=0,
        max_tokens=4000,
    )

    content = resp.choices[0].message.content
    return content


# ---------------- Post-processing ----------------

def standardize_keys(data):
    if isinstance(data, dict):
        new_dict = {}
        for key, value in data.items():
            if key in ["Contribution", "Summary"]:
                key = "Description"
            processed_value = standardize_keys(value)
            if key not in ['Year', 'Field', 'Location', 'AdditionalInfo']:
                new_dict[key] = processed_value
        return new_dict
    elif isinstance(data, list):
        return [standardize_keys(item) for item in data]
    else:
        return data


def process_json_data(input_json: str):
    data = json.loads(input_json)
    standardized_data = standardize_keys(data)
    if "Key Skills" in standardized_data and isinstance(standardized_data["Key Skills"], list):
        standardized_data["Key Skills"] = [
            t for t in standardized_data["Key Skills"] if 'Languages' not in t and 'English' not in t
        ]
    return standardized_data


# ---------------- Main ----------------

def main():
    # Hardcoded variables
    # files = [f for f in os.listdir('CVs')]
    # index_rand = randint(0,len(files))
    # file = files[index_rand]
    # f = file.split('.')[0]
    # cv_path = Path(f"CVs/{file}")  
    cv_path = Path(f"CVs/cv_test19.pdf")  # <-- change this to your CV file path
    model = "gpt-4o-mini"
    # out_path = Path(f"Archive/{f}.json")
    out_path = Path(f"Archive/cv_test19.json")

    if not cv_path.exists():
        raise FileNotFoundError(f"File not found: {cv_path}")

    print(ProgramDefinition)

    text = parse_file(str(cv_path))
    normalized_text = text
    raw_response = call_openai(normalized_text, model=model)

    data = process_json_data(raw_response)

    out_dir = out_path.parent
    out_dir.mkdir(exist_ok=True, parents=True)

    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=4, ensure_ascii=False)

    print(f"✅ Saved parsed JSON to: {out_path}")


if __name__ == "__main__":
    main()