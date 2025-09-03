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


# ----------------------------
# LLM setup
# ----------------------------
def create_llm(model: str = "gpt-3.5-turbo", temperature: float = 0.2) -> ChatOpenAI:
    """
    Keep your key in the environment:
      setx OPENAI_API_KEY "..."
    """
    api_key = os.environ.get("OPENAI_API_KEY","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")
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
def speak(text: str) -> None:
    try:
        engine = pyttsx3.init()
        engine.setProperty('rate', 160)
        engine.setProperty('volume', 1)
        engine.say(text)
        engine.runAndWait()
    except Exception as e:
        print("TTS error:", e)


def listen() -> Optional[str]:
    try:
        recognizer = sr.Recognizer()
        mic = sr.Microphone()
    except Exception as e:
        print("Microphone init error:", e)
        return None

    print("🎤 Say something...")
    with mic as source:
        recognizer.adjust_for_ambient_noise(source)
        audio = recognizer.listen(source)

    try:
        text = recognizer.recognize_google(audio)
        print("You said:", text)
        return text
    except sr.UnknownValueError:
        print("❌ Could not understand audio")
        return None
    except sr.RequestError:
        print("⚠️ Speech Recognition API unavailable")
        return None


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
                 model: str = "gpt-3.5-turbo") -> Dict[str, Any]:
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

Your task: generate exactly {num_questions} interview questions tightly tailored to BOTH the candidate and the job offer.

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
                      model: str = "gpt-3.5-turbo") -> Dict[str, Any]:
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

    with open(CV_PATH, "r", encoding="utf-8") as f:
        cv = json.load(f)

    with open(JOB_PATH, "r", encoding="utf-8") as f:
        job_offer_raw = json.load(f)  # expects {"Job Offer": "..."} or a string

    try:
        questions_pack = GetQuestions(cv, job_offer_raw, num_questions=1, model="gpt-3.5-turbo")
        print(json.dumps(questions_pack, ensure_ascii=False, indent=2))

        # Optionally speak first question:
        # if questions_pack.get("questions"):
        #     speak("Here is your first interview question.")
        #     speak(questions_pack["questions"][0]["question"])
        responses = []
        Qs = []
        for questions in questions_pack["questions"]:
            speak(questions["question"])
            Qs.append(questions["question"])
            response = listen()
            responses.append(response)
        Result = EvaluateInterview(Qs,responses,cv,job_offer_raw)
        print(Result)
        Name = cv['Name']
        timestamp_str = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")  # safer for filenames
        with open(f"Interviews/Interview_{Name}_{timestamp_str}.json", "w") as of:
            json.dump(Result, of, indent=4)
    except Exception as e:
        print("Error while generating questions:", e)
