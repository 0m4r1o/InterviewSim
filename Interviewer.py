# Interviewer.py
# Requires:
#   pip install -U langchain-community openai pyttsx3 SpeechRecognition
#   (plus a PyAudio install suitable for your OS if you use the mic)
#
# Environment:
#   setx OPENAI_API_KEY "sk-..."   (Windows)
#   or export OPENAI_API_KEY="sk-..." (Linux/macOS)

from langchain_community.chat_models import ChatOpenAI
import pyttsx3
import speech_recognition as sr
import json
import os
from typing import Any, Dict, Optional, Union
from datetime import datetime
ProgramDefinition = """
Created on Thu May 2 2025 16:10:00 

@author: OmarDasser

This script conducts AI-assisted mock interviews using a candidate's CV
and a job offer description. It generates tailored interview questions,
records and transcribes spoken responses, and evaluates the answers against
the job requirements. The program integrates language models for question
generation and scoring, text-to-speech for asking questions aloud, and 
speech recognition for capturing responses. Final evaluations, including
scores, strengths, weaknesses, and recommendations, are saved as structured 
JSON reports for later review.
""".strip()

# ----------------------------
# LLM setup
# ----------------------------
def create_llm(model: str = "gpt-4o-mini", temperature: float = 0.2) -> ChatOpenAI:
    """
    Keep your key in the environment:
      setx OPENAI_API_KEY "..."
    """
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError("OPENAI_API_KEY is not set in environment variables.")
    return ChatOpenAI(
        temperature=temperature,
        model=model,
        verbose=False,
        openai_api_key=api_key
    )


# ----------------------------
# TTS / ASR helpers (optional)
# ----------------------------
from openai import OpenAI
import os, tempfile, uuid, sys
from pathlib import Path


client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY","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"))

from pathlib import Path
import tempfile, uuid, time, os
import pygame


def speak(text: str) -> None:
    path = f"audios/tts_{uuid.uuid4().hex}.mp3"
    try:
        # synthesize to a unique temp MP3 file
        with client.audio.speech.with_streaming_response.create(
            model="gpt-4o-mini-tts",
            voice="alloy",
            input=text
        ) as resp:
            resp.stream_to_file(str(path))

        # play it with pygame
        pygame.mixer.init()                    # init audio
        pygame.mixer.music.load(str(path))     # load mp3
        pygame.mixer.music.play()              # play (non-blocking)
        while pygame.mixer.music.get_busy():   # block until it finishes
            time.sleep(0.1)
    except Exception as e:
        print("TTS error:", e)
    finally:
        try:
            if path.exists():
                path.unlink()
        except Exception:
            pass





import tempfile
import sounddevice as sd
import wave, tempfile
from typing import Optional
import os


import numpy as np
import sounddevice as sd
import wave, tempfile

def listen(threshold=350, silence_secs=5, max_secs=30):
    """
    Record audio until user stops talking (silence detection).
    - threshold: amplitude threshold to consider as silence
    - silence_secs: how long silence must last before stopping
    - max_secs: safety cutoff
    """
    samplerate = 16000
    blocksize = int(0.5 * samplerate)  # 0.5 second blocks
    channels = 1

    print("🎤 Speak now (auto-stop on silence)...")

    recording = []
    silence_count = 0
    max_blocks = int(max_secs * samplerate / blocksize)

    with sd.InputStream(samplerate=samplerate, channels=channels, dtype="int16") as stream:
        for _ in range(max_blocks):
            block, _ = stream.read(blocksize)
            recording.append(block)

            # check loudness
            volume = np.abs(block).mean()
            if volume < threshold:
                silence_count += 1
            else:
                silence_count = 0

            if silence_count * 0.5 >= silence_secs:
                break

    audio = np.concatenate(recording, axis=0)

    # save temp wav
    tmp_path = tempfile.mktemp(suffix=".wav")
    with wave.open(tmp_path, "wb") as wf:
        wf.setnchannels(channels)
        wf.setsampwidth(2)
        wf.setframerate(samplerate)
        wf.writeframes(audio.tobytes())

    # send to transcription
    with open(tmp_path, "rb") as f:
        tr = client.audio.transcriptions.create(
            model="gpt-4o-mini-transcribe",
            file=f
        )
    print(f"🎤 Response: {tr.text}")
    return tr.text

import random
import re

def humanize_question(text: str) -> str:
    # Common fillers/disfluencies
    fillers = ["hmm", "euuh", "like", "you know", "let’s see","I mean", "ah", "uh", "so", "well"]
    
    # Split by commas so we can inject around pauses
    parts = re.split(r'([,;])', text)
    new_parts = []
    
    for i, part in enumerate(parts):
        part = part.strip()
        if not part:
            continue
        
        # 30% chance to prepend a filler before a segment
        if random.random() < 0.3:
            filler = random.choice(fillers)
            part = f"{filler}, {part}"
        
        # 20% chance to append a filler at the end of a segment
        if random.random() < 0.2:
            filler = random.choice(fillers)
            part = f"{part}, {filler}"
        
        new_parts.append(part)
    
    # Sometimes add one at the very beginning
    if random.random() < 0.4:
        new_parts[0] = random.choice(fillers) + ", " + new_parts[0]
    
    # Sometimes add one at the very end
    if random.random() < 0.4:
        new_parts[-1] = new_parts[-1] + ", " + random.choice(fillers)
    
    return " ".join(new_parts)







# ----------------------------
# Robust JSON extraction
# ----------------------------
def _extract_first_json_blob(text: str) -> Optional[str]:
    """
    Extract the first top-level JSON object from model text output.
    Handles stray text or Markdown fences.
    """
    start = text.find("{")
    end = text.rfind("}")
    if start != -1 and end != -1 and end > start:
        candidate = text[start:end + 1]
        try:
            json.loads(candidate)
            return candidate
        except Exception:
            # Try removing code fences
            candidate2 = candidate.replace("```json", "").replace("```", "").strip()
            try:
                json.loads(candidate2)
                return candidate2
            except Exception:
                return None
    return None


# ----------------------------
# Core: Prompt + Question generation
# ----------------------------
def GetQuestions(cv: Dict[str, Any],
                 job_offer_raw: Union[Dict[str, Any], str],
                 num_questions: int = 12,
                 model: str = "gpt-4o-mini") -> Dict[str, Any]:
    """
    Generate structured interview questions tailored to the given CV and job offer.
    - cv: dict from cv1.json (expects "Name" or similar)
    - job_offer_raw: dict {"Job Offer": "..."} or a raw string.
    Returns a dict:
    {
      "role": "interviewer",
      "candidate_name": "...",
      "job_offer": "...",
      "mix": {"technical": 0.5, "behavioral": 0.3, "situational": 0.2},
      "questions": [ ... ]
    }
    """

    # Candidate name from your CV shape
    candidate_name = (
        cv.get("Name")
        or cv.get("name")
        or cv.get("full_name")
        or "Candidate"
    )

    # Extract job offer as a string
    if isinstance(job_offer_raw, dict) and "Job Offer" in job_offer_raw:
        job_offer_text = str(job_offer_raw["Job Offer"]).strip()
    elif isinstance(job_offer_raw, str):
        job_offer_text = job_offer_raw.strip()
    else:
        raise ValueError("JobOffer must be a string or a dict with key 'Job Offer'.")

    # Build the instruction-heavy prompt
    # NOTE: Double braces {{ }} are used to escape literal braces in f-strings.
    prompt = f"""
You are a senior hiring manager and expert interviewer. You will receive two items:
1) CANDIDATE_CV: the candidate's CV in JSON form.
2) JOB_OFFER: a free-text job description.

Your task: generate exactly {num_questions} interview questions tightly tailored to BOTH the candidate and the job offer. can you generate some filler words in the question to sound human here's an example (hmm, euuh, like, you know, let’s see,I mean, ah, uh, so, well ) add some others id you'd like when you see it to be adequate, make sure to use it more than once

CONSTRAINTS:
- Mix: ~50% technical, ~30% behavioral, ~20% situational/case.
- Prioritize competencies explicitly implied by the job offer (e.g., RHEL, AIX, enterprise server ops).
- Calibrate difficulty to the candidate's experience inferred from the CV.
- Avoid generic filler; every question must clearly connect to either the CV or the job offer.
- Include 2 short probing follow-ups per question.
- Return STRICT JSON ONLY. No preamble, no markdown, no commentary.

OUTPUT JSON SCHEMA:
{{
  "role": "interviewer",
  "candidate_name": "{candidate_name}",
  "job_offer": "{job_offer_text}",
  "mix": {{"technical": 0.5, "behavioral": 0.3, "situational": 0.2}},
  "questions": [
    {{
      "id": <int starting at 1>,
      "type": "technical|behavioral|situational",
      "competency": "<skill or soft-skill>",
      "difficulty": <integer 1-5>,
      "question": "<concise, role-specific question>",
      "follow_ups": ["<short probe 1>", "<short probe 2>"],
      "rationale": "<why this question fits this candidate and role>",
      "expected_good_answer": ["<bullet point>", "<bullet point>", "<bullet point>"]
    }}
  ]
}}

CANDIDATE_CV (JSON):
{json.dumps(cv, ensure_ascii=False, indent=2)}

JOB_OFFER (TEXT):
{job_offer_text}

Return ONLY the JSON described above.
""".strip()

    llm = create_llm(model=model, temperature=0.2)
    result = llm.invoke(prompt)
    raw_text = getattr(result, "content", str(result))

    # Parse strictly
    try:
        return json.loads(raw_text)
    except Exception:
        blob = _extract_first_json_blob(raw_text)
        if not blob:
            raise ValueError("LLM did not return valid JSON. Raw output:\n" + raw_text)
        return json.loads(blob)

def EvaluateInterview(questions: list,
                      responses: list,
                      cv: Optional[Dict[str, Any]] = None,
                      job_offer: Optional[Union[Dict[str, Any], str]] = None,
                      model: str = "gpt-4o-mini") -> Dict[str, Any]:
    """
    Evaluate an interview given parallel lists of questions and responses.
    Optionally provide the CV and job_offer for context to calibrate the rubric.

    Returns a dict like:
    {
      "overall_score": 1-10,
      "summary": {
        "strengths": ["..."],
        "weaknesses": ["..."],
        "recommendations": ["..."]
      },
      "per_question": [
        {
          "id": 1,
          "question": "...",
          "answer": "...",
          "score": 1-10,
          "evidence": ["bullet", "..."],
          "comment": "short qualitative note"
        }
      ]
    }
    """

    # Safety: align lengths and coerce None answers to empty strings
    n = min(len(questions), len(responses))
    pairs = []
    for i in range(n):
        q = questions[i] if isinstance(questions[i], str) else str(questions[i])
        a_raw = responses[i]
        a = "" if a_raw is None else (a_raw if isinstance(a_raw, str) else str(a_raw))
        pairs.append({"id": i + 1, "question": q, "answer": a})

    # Normalize job_offer to plain text (if provided)
    job_offer_text = None
    if job_offer is not None:
        if isinstance(job_offer, dict) and "Job Offer" in job_offer:
            job_offer_text = str(job_offer["Job Offer"]).strip()
        elif isinstance(job_offer, str):
            job_offer_text = job_offer.strip()
        else:
            job_offer_text = json.dumps(job_offer, ensure_ascii=False)

    # Build evaluation prompt
    cv_json = json.dumps(cv, ensure_ascii=False, indent=2) if cv is not None else "{}"
    pairs_json = json.dumps(pairs, ensure_ascii=False, indent=2)
    job_offer_str = job_offer_text if job_offer_text is not None else "N/A"

    prompt = f"""
You are a seasoned hiring manager. Evaluate the following interview answers.
Use the job offer (if present) to weight domain alignment, and the CV (if present) to calibrate seniority.

SCORING RUBRIC (apply consistently):
- 9–10 Exceptional: precise, complete, senior-level reasoning, strong evidence of experience, clear communication.
- 7–8  Strong: mostly correct, minor gaps, good structure and relevant examples.
- 5–6  Adequate: partially correct, lacks depth or clarity, some misconceptions.
- 3–4  Weak: significant gaps, shallow or off-topic answers.
- 1–2  Poor: incorrect or non-answers.

Consider for each answer:
- Technical correctness / problem-solving / tradeoffs (if technical)
- Relevance to job offer requirements
- Clarity, structure, and use of concrete examples
- Ownership, collaboration, and communication signals (for behavioral)
- Realistic incident handling and prioritization (for situational)

Return STRICT JSON ONLY. No preamble, no markdown.

OUTPUT JSON SCHEMA:
{{
  "overall_score": <integer 1-10>,
  "summary": {{
    "strengths": ["<bullet>", "<bullet>"],
    "weaknesses": ["<bullet>", "<bullet>"],
    "recommendations": ["<actionable next step>", "<actionable next step>"]
  }},
  "per_question": [
    {{
      "id": <int>,
      "question": "<verbatim question>",
      "answer": "<verbatim answer>",
      "score": <integer 1-10>,
      "evidence": ["<short bullet citing what was good/bad>", "..."],
      "comment": "<1–3 sentence justification>"
    }}
  ]
}}

JOB_OFFER (TEXT OR N/A):
{job_offer_str}

CANDIDATE_CV (JSON OR empty):
{cv_json}

INTERVIEW (PAIR LIST):
{pairs_json}

Return ONLY the JSON described above.
""".strip()

    llm = create_llm(model=model, temperature=0.0)  # deterministic scoring
    result = llm.invoke(prompt)
    raw_text = getattr(result, "content", str(result))

    # Parse strictly, reusing your helper
    try:
        return json.loads(raw_text)
    except Exception:
        blob = _extract_first_json_blob(raw_text)
        if not blob:
            raise ValueError("LLM did not return valid JSON. Raw output:\n" + raw_text)
        return json.loads(blob)


# ----------------------------
# Example main
# ----------------------------
if __name__ == "__main__":
    # Adjust paths if needed
    CV_PATH = "cv1.json"
    JOB_PATH = "JobOffer.json"
    print(ProgramDefinition)

    with open(CV_PATH, "r", encoding="utf-8") as f:
        cv = json.load(f)

    with open(JOB_PATH, "r", encoding="utf-8") as f:
        job_offer_raw = json.load(f)  # expects {"Job Offer": "..."} or a string

    try:
        questions_pack = GetQuestions(cv, job_offer_raw, num_questions=2, model="gpt-4o-mini")
        # print(json.dumps(questions_pack, ensure_ascii=False, indent=2))

        # Optionally speak first question:
        # if questions_pack.get("questions"):
        #     speak("Here is your first interview question.")
        #     speak(questions_pack["questions"][0]["question"])
        responses = []
        Qs = []
        for q in questions_pack["questions"]:
            Q = q["question"]
            Q_human = humanize_question(Q)  # <-- add fillers
            print(f"Question : {Q_human}")
            speak(Q_human)
            Qs.append(Q)
            response = listen()
            responses.append(response)

        Result = EvaluateInterview(Qs,responses,cv,job_offer_raw)
        # print(Result)
        Name = cv['Name']
        timestamp_str = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")  # safer for filenames
        with open(f"Interviews/Interview_{Name}_{timestamp_str}.json", "w") as of:
            json.dump(Result, of, indent=4)
            print(json.dumps(str(Result),indent=4))

    except Exception as e:
        print("Error while generating questions:", e)
