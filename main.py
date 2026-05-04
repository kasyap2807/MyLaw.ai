# from fastapi import FastAPI, HTTPException
# from fastapi.middleware.cors import CORSMiddleware
# from fastapi.staticfiles import StaticFiles
# from fastapi.responses import FileResponse
# from pydantic import BaseModel
# from mistralai import Mistral
# import os

# app = FastAPI(
#     title="LexAI - Legal Analysis API",
#     description="AI-powered legal scenario analysis using Mistral",
#     version="1.0.0"
# )

# app.add_middleware(
#     CORSMiddleware,
#     allow_origins=["*"],
#     allow_credentials=True,
#     allow_methods=["*"],
#     allow_headers=["*"],
# )

# # Mount static files
# app.mount("/static", StaticFiles(directory="static"), name="static")


# class ScenarioRequest(BaseModel):
#     scenario: str
#     jurisdiction: str = "India"  # Default jurisdiction
#     api_key: str  # User provides their Mistral API key


# class LegalAnalysis(BaseModel):
#     applicable_laws: list
#     consequences: list
#     recommendations: list
#     severity: str
#     summary: str
#     disclaimer: str


# SYSTEM_PROMPT = """You are LexAI, an expert legal analyst with deep knowledge of laws across multiple jurisdictions. 
# When given a scenario, you must analyze it thoroughly and respond ONLY with a valid JSON object in this exact format:

# {
#   "applicable_laws": [
#     {
#       "name": "Law/Act/Section Name",
#       "section": "Specific section or article number",
#       "description": "What this law covers and why it applies",
#       "jurisdiction": "Country/State this applies to"
#     }
#   ],
#   "consequences": [
#     {
#       "type": "Civil / Criminal / Administrative / Financial",
#       "description": "Detailed description of the consequence",
#       "severity": "Minor / Moderate / Severe / Critical",
#       "penalty": "Specific penalty if applicable (fine amount, jail term, etc.)"
#     }
#   ],
#   "recommendations": [
#     {
#       "action": "Recommended action",
#       "priority": "Immediate / Short-term / Long-term",
#       "description": "Why this is recommended"
#     }
#   ],
#   "severity": "Low / Medium / High / Critical",
#   "summary": "A comprehensive 3-5 sentence summary of the legal situation",
#   "disclaimer": "This analysis is for informational purposes only and does not constitute legal advice. Please consult a qualified attorney."
# }

# Be thorough, cite specific laws, sections, and provide realistic penalties. If the jurisdiction is India, cite Indian Penal Code, specific Acts, etc. Always include the disclaimer."""


# @app.get("/")
# async def serve_frontend():
#     return FileResponse("static/index.html")


# @app.post("/analyze", response_model=dict)
# async def analyze_scenario(request: ScenarioRequest):
#     if not request.scenario.strip():
#         raise HTTPException(status_code=400, detail="Scenario cannot be empty")
    
#     if len(request.scenario) < 20:
#         raise HTTPException(status_code=400, detail="Please provide a more detailed scenario (at least 20 characters)")

#     try:
#         client = Mistral(api_key=request.api_key)
        
#         user_message = f"""Jurisdiction: {request.jurisdiction}

# Legal Scenario:
# {request.scenario}

# Analyze this scenario and identify all applicable laws, consequences, and recommendations."""

#         response = client.chat.complete(
#             model="mistral-large-latest",
#             messages=[
#                 {"role": "system", "content": SYSTEM_PROMPT},
#                 {"role": "user", "content": user_message}
#             ],
#             temperature=0.2,
#             max_tokens=4000,
#         )

#         content = response.choices[0].message.content.strip()
        
#         # Clean up markdown code blocks if present
#         if content.startswith("```"):
#             content = content.split("```")[1]
#             if content.startswith("json"):
#                 content = content[4:]
#         if content.endswith("```"):
#             content = content[:-3]
        
#         import json
#         analysis = json.loads(content.strip())
#         return analysis

#     except json.JSONDecodeError:
#         raise HTTPException(status_code=500, detail="Failed to parse legal analysis. Please try again.")
#     except Exception as e:
#         error_msg = str(e)
#         if "401" in error_msg or "Unauthorized" in error_msg or "authentication" in error_msg.lower():
#             raise HTTPException(status_code=401, detail="Invalid Mistral API key. Please check your key.")
#         elif "429" in error_msg:
#             raise HTTPException(status_code=429, detail="Rate limit exceeded. Please wait before trying again.")
#         else:
#             raise HTTPException(status_code=500, detail=f"Analysis failed: {error_msg}")


# @app.get("/health")
# async def health_check():
#     return {"status": "healthy", "service": "LexAI Legal Analysis API", "version": "1.0.0"}


# @app.get("/jurisdictions")
# async def get_jurisdictions():
#     return {
#         "jurisdictions": [
#             "India", "United States", "United Kingdom", "Australia",
#             "Canada", "European Union", "Singapore", "UAE"
#         ]
#     }


# if __name__ == "__main__":
#     import uvicorn
#     uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)


import os
import json
import logging
import time
from contextlib import asynccontextmanager
from typing import Optional

from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse, JSONResponse
from pydantic import BaseModel, field_validator
from dotenv import load_dotenv

from fastapi.middleware.cors import CORSMiddleware

app = FastAPI()

origins = [
    "http://localhost:8080",  # React dev server
    "http://127.0.0.1:3000",
    "https://yourdomain.com",  # production frontend
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,        # allowed domains
    allow_credentials=True,
    allow_methods=["*"],          # GET, POST, PUT, DELETE
    allow_headers=["*"],          # all headers
)

# ── Load .env ─────────────────────────────────────────────────────────────────
load_dotenv()

# ── Config from environment ───────────────────────────────────────────────────
MISTRAL_API_KEY      = os.getenv("MISTRAL_API_KEY", "").strip()
MISTRAL_MODEL        = os.getenv("MISTRAL_MODEL", "mistral-large-latest")
MAX_TOKENS           = int(os.getenv("MAX_TOKENS", "4000"))
TEMPERATURE          = float(os.getenv("TEMPERATURE", "0.2"))
HOST                 = os.getenv("HOST", "0.0.0.0")
PORT                 = int(os.getenv("PORT", "8000"))
DEFAULT_JURISDICTION = os.getenv("DEFAULT_JURISDICTION", "India")
ALLOW_USER_API_KEY   = os.getenv("ALLOW_USER_API_KEY", "false").lower() == "true"
MAX_SCENARIO_LENGTH  = int(os.getenv("MAX_SCENARIO_LENGTH", "5000"))

# ── Logging ───────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("lexai")

# ── Mistral client import (handles v1 and v2 SDK layouts) ────────────────────
try:
    from mistralai import Mistral          # mistralai >= 1.x
    log.info("Using mistralai >= 1.x import path")
except ImportError:
    try:
        from mistralai.client import Mistral  # mistralai 2.x alternate path
        log.info("Using mistralai.client import path")
    except ImportError:
        from mistralai.client import MistralClient as Mistral  # mistralai 0.x
        log.info("Using legacy MistralClient import")


# ── Lifespan: startup / shutdown messages ────────────────────────────────────
@asynccontextmanager
async def lifespan(app: FastAPI):
    log.info("━" * 50)
    log.info("  LexAI Legal Intelligence API  v1.0.0")
    log.info("━" * 50)
    log.info(f"  Model      : {MISTRAL_MODEL}")
    log.info(f"  Server key : {'✓ configured' if MISTRAL_API_KEY else '✗ not set (user must supply)'}")
    log.info(f"  User key   : {'allowed' if ALLOW_USER_API_KEY else 'not allowed'}")
    log.info(f"  Jurisdiction: {DEFAULT_JURISDICTION}")
    log.info(f"  Docs       : http://{HOST}:{PORT}/docs")
    log.info("━" * 50)
    yield
    log.info("LexAI shutting down.")


# ── App ───────────────────────────────────────────────────────────────────────
app = FastAPI(
    title="LexAI — Legal Intelligence API",
    description=(
        "AI-powered legal scenario analysis using Mistral.\n\n"
        "Submit a scenario and receive applicable laws, consequences, "
        "severity level, and recommended actions."
    ),
    version="1.0.0",
    lifespan=lifespan,
    docs_url="/docs",
    redoc_url="/redoc",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.mount("/static", StaticFiles(directory="static"), name="static")


# ── Request / Response models ─────────────────────────────────────────────────
VALID_JURISDICTIONS = [
    "India", "United States", "United Kingdom",
    "Australia", "Canada", "European Union", "Singapore", "UAE",
]

class ScenarioRequest(BaseModel):
    scenario: str
    jurisdiction: str = DEFAULT_JURISDICTION
    api_key: str = ""          # only used when ALLOW_USER_API_KEY=true

    @field_validator("scenario")
    @classmethod
    def scenario_not_empty(cls, v: str) -> str:
        v = v.strip()
        if not v:
            raise ValueError("Scenario cannot be empty.")
        if len(v) < 20:
            raise ValueError("Scenario is too short — please provide at least 20 characters.")
        if len(v) > MAX_SCENARIO_LENGTH:
            raise ValueError(f"Scenario exceeds the {MAX_SCENARIO_LENGTH}-character limit.")
        return v

    @field_validator("jurisdiction")
    @classmethod
    def jurisdiction_valid(cls, v: str) -> str:
        if v not in VALID_JURISDICTIONS:
            raise ValueError(f"Unsupported jurisdiction '{v}'. Choose from: {', '.join(VALID_JURISDICTIONS)}")
        return v


# ── System prompt ─────────────────────────────────────────────────────────────
SYSTEM_PROMPT = """You are LexAI, an expert legal analyst with deep knowledge of laws across multiple jurisdictions.

When given a legal scenario, analyze it thoroughly and respond ONLY with a valid JSON object using this exact structure:

{
  "applicable_laws": [
    {
      "name": "Full name of the Act / Code / Regulation",
      "section": "Specific section, article, or clause number",
      "description": "What this law covers and exactly why it applies to this scenario",
      "jurisdiction": "Country or state where this law is in force"
    }
  ],
  "consequences": [
    {
      "type": "Criminal | Civil | Administrative | Financial",
      "description": "Detailed description of this consequence and how it arises",
      "severity": "Minor | Moderate | Severe | Critical",
      "penalty": "Specific penalty: fine amount, imprisonment term, disqualification, etc."
    }
  ],
  "recommendations": [
    {
      "action": "Concise name of the recommended action",
      "priority": "Immediate | Short-term | Long-term",
      "description": "Why this action is important and what outcome it achieves"
    }
  ],
  "severity": "Low | Medium | High | Critical",
  "summary": "3-5 sentence plain-language summary of the overall legal situation, key risks, and what the person should know first.",
  "disclaimer": "This analysis is for informational purposes only and does not constitute legal advice. Please consult a qualified attorney for advice specific to your situation."
}

Rules:
- Cite specific laws by their official name and section numbers (e.g. IPC Section 420, IT Act 2000 Section 66C).
- Include at least 2 applicable laws and at least 2 consequences when they exist.
- Severity levels for consequences: Minor = warning/small fine, Moderate = significant fine or civil liability, Severe = criminal charges or large penalty, Critical = imprisonment or major financial ruin.
- Overall severity: Low = informational/minor, Medium = civil risk, High = criminal risk, Critical = urgent/life-altering.
- Always write the disclaimer exactly as shown.
- Return ONLY the JSON — no markdown fences, no preamble, no explanation outside the JSON."""


# ── Helpers ───────────────────────────────────────────────────────────────────
def resolve_api_key(user_key: str) -> str:
    """Return the API key to use, or raise 400/401 if none available."""
    if MISTRAL_API_KEY:
        # Server key takes priority; user key allowed only if flag is set
        if ALLOW_USER_API_KEY and user_key.strip():
            log.info("Using user-supplied API key (ALLOW_USER_API_KEY=true)")
            return user_key.strip()
        return MISTRAL_API_KEY
    # No server key — require user to supply one
    if not user_key.strip():
        raise HTTPException(
            status_code=400,
            detail=(
                "No API key configured on the server. "
                "Set MISTRAL_API_KEY in .env, or pass api_key in your request."
            ),
        )
    return user_key.strip()


def clean_json(raw: str) -> str:
    """Remove markdown code fences that Mistral sometimes adds."""
    text = raw.strip()
    if text.startswith("```"):
        # Split on ``` and take the inner block
        parts = text.split("```")
        text = parts[1] if len(parts) >= 2 else text
        if text.lower().startswith("json"):
            text = text[4:]
    if text.endswith("```"):
        text = text[:-3]
    return text.strip()


def call_mistral(api_key: str, user_message: str) -> dict:
    """Call Mistral and return parsed JSON dict."""
    client = Mistral(api_key=api_key)
    response = client.chat.complete(
        model=MISTRAL_MODEL,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user",   "content": user_message},
        ],
        temperature=TEMPERATURE,
        max_tokens=MAX_TOKENS,
    )
    raw = response.choices[0].message.content or ""
    return json.loads(clean_json(raw))


# ── Global error handler ──────────────────────────────────────────────────────
@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    log.error(f"Unhandled error on {request.url}: {exc}")
    return JSONResponse(status_code=500, content={"detail": "Internal server error."})


# ── Routes ────────────────────────────────────────────────────────────────────
@app.get("/", include_in_schema=False)
async def serve_frontend():
    return FileResponse("static/index.html")


@app.post(
    "/analyze",
    response_model=dict,
    summary="Analyse a legal scenario",
    description=(
        "Submit a plain-language description of a legal situation. "
        "Returns applicable laws, consequences with severity levels, "
        "recommended actions, and an overall risk rating."
    ),
    tags=["Analysis"],
)
async def analyze_scenario(request: ScenarioRequest):
    api_key = resolve_api_key(request.api_key)

    user_message = (
        f"Jurisdiction: {request.jurisdiction}\n\n"
        f"Legal Scenario:\n{request.scenario}\n\n"
        "Analyse this scenario thoroughly. Identify all applicable laws, "
        "consequences, and recommended actions."
    )

    start = time.perf_counter()
    log.info(f"Analysing scenario | jurisdiction={request.jurisdiction} | length={len(request.scenario)}")

    try:
        result = call_mistral(api_key, user_message)
        elapsed = time.perf_counter() - start
        laws_n  = len(result.get("applicable_laws", []))
        cons_n  = len(result.get("consequences", []))
        sev     = result.get("severity", "?")
        log.info(f"Analysis complete  | {elapsed:.1f}s | laws={laws_n} | consequences={cons_n} | severity={sev}")
        return result

    except json.JSONDecodeError as e:
        log.warning(f"JSON parse error: {e}")
        raise HTTPException(
            status_code=502,
            detail="The AI returned an unexpected response format. Please try again.",
        )
    except HTTPException:
        raise
    except Exception as e:
        msg = str(e)
        log.error(f"Mistral API error: {msg}")
        if any(k in msg for k in ("401", "Unauthorized", "unauthorized", "authentication")):
            raise HTTPException(status_code=401, detail="Invalid Mistral API key. Please verify your key.")
        if "429" in msg or "rate" in msg.lower():
            raise HTTPException(status_code=429, detail="Mistral rate limit reached. Please wait a moment and try again.")
        if "timeout" in msg.lower() or "timed out" in msg.lower():
            raise HTTPException(status_code=504, detail="Request timed out. Please try again.")
        raise HTTPException(status_code=500, detail=f"Analysis failed: {msg}")


@app.get(
    "/health",
    summary="Health check",
    tags=["System"],
)
async def health_check():
    return {
        "status": "healthy",
        "service": "LexAI Legal Intelligence API",
        "version": "1.0.0",
        "model": MISTRAL_MODEL,
        "server_key_configured": bool(MISTRAL_API_KEY),
        "allow_user_key": ALLOW_USER_API_KEY,
        "max_scenario_length": MAX_SCENARIO_LENGTH,
    }


@app.get(
    "/config",
    summary="Frontend configuration",
    description="Public-safe config used by the frontend to decide which UI elements to show.",
    tags=["System"],
)
async def get_config():
    return {
        "default_jurisdiction": DEFAULT_JURISDICTION,
        "server_key_configured": bool(MISTRAL_API_KEY),
        "allow_user_api_key": ALLOW_USER_API_KEY,
        "max_scenario_length": MAX_SCENARIO_LENGTH,
    }


@app.get(
    "/jurisdictions",
    summary="List supported jurisdictions",
    tags=["System"],
)
async def get_jurisdictions():
    return {"jurisdictions": VALID_JURISDICTIONS}


# ── Entry point ───────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host=HOST, port=PORT, reload=True)
