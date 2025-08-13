from __future__ import annotations
import json

SUPPORTED = ("openai", "gemini", "anthropic")

SYS_PROMPT = (
    "You are a cryptography engineer. Provide precise, actionable remediation steps for improving an "
    "entropy source used in cryptographic contexts. Consider NIST SP 800-22 tests, SP 800-90B health tests, "
    "entropy/compressibility, and a short-history ML predictor.\n\n"
    "Constraints:\n"
    "- Be concise, practical, and technical; no marketing tone.\n"
    "- Prioritize steps that reduce predictability/local bias (e.g., conditioning, DRBG usage, pool mixing).\n"
    "- If all tests pass (incl. BH-FDR) and health tests pass, advise on continuous monitoring/health tests."
)


def _build_prompt(results: dict) -> str:
    return (
        f"{SYS_PROMPT}\n\n"
        f"Metrics JSON:\n```json\n{json.dumps(results, indent=2)}\n```\n"
        "Return bullet points only."
    )


def ai_advice(
    results: dict, *, provider: str, api_key: str, model: str, temperature: float = 0.2
) -> str:
    """
    provider: 'openai' | 'gemini' | 'anthropic'
    Sends ONLY summarized metrics/results (no raw bits).
    """
    if provider not in SUPPORTED:
        return f"Provider '{provider}' not supported."
    if not api_key:
        return "No API key provided."

    prompt = _build_prompt(results)

    try:
        if provider == "openai":
            from langchain_openai import ChatOpenAI

            llm = ChatOpenAI(
                model=model, temperature=temperature, api_key=api_key, timeout=20
            )
            resp = llm.invoke(prompt)
            return (getattr(resp, "content", None) or str(resp)).strip()

        if provider == "gemini":
            from langchain_google_genai import ChatGoogleGenerativeAI

            llm = ChatGoogleGenerativeAI(
                model=model, temperature=temperature, google_api_key=api_key
            )
            resp = llm.invoke(prompt)
            return (getattr(resp, "content", None) or str(resp)).strip()

        if provider == "anthropic":
            from langchain_anthropic import ChatAnthropic

            llm = ChatAnthropic(
                model=model,
                temperature=temperature,
                anthropic_api_key=api_key,
                timeout=20,
            )
            resp = llm.invoke(prompt)
            return (getattr(resp, "content", None) or str(resp)).strip()

    except Exception as e:
        return f"AI advisor error ({provider}): {e}"

    return "No AI advice returned."
