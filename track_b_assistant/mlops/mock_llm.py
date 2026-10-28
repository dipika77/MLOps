"""
A deterministic, prompt-aware stand-in for the LLM, used so this week's
prompt-version experiment (mlops/run_experiments.py) is reproducible offline,
without a GROQ_API_KEY, the same way scripts/eval_agentic.py's --mock mode
works.

Unlike a simple "return canned JSON regardless of input" mock, this one reads
the ACTUAL PROMPT TEXT it is given and only applies a fix's behavior when
that fix's instruction is actually present in the prompt - the point is to
honestly simulate "a model that follows instructions when it's given them,
and doesn't otherwise", so that the v1 -> v2 -> v3 differences in the
experiment are driven by what the prompt actually says, not by which
version's name we happened to be running.

Swap-in for a real model: everything in mlops/run_experiments.py that takes
this mock as `chat_completion` also accepts `app.llm_client.chat_completion`
directly (pass --live), which calls Groq for real when GROQ_API_KEY is set.
"""
import json
import re
from types import SimpleNamespace


def _response(payload: dict, tokens: int = 120):
    message = SimpleNamespace(content=json.dumps(payload), tool_calls=None)
    choice = SimpleNamespace(message=message)
    usage = SimpleNamespace(total_tokens=tokens)
    return SimpleNamespace(choices=[choice], usage=usage), "mock/prompt-aware-v1"


def _extract_question(user_content: str) -> str:
    match = re.search(r"QUESTION:\n(.*?)\n\n", user_content, re.DOTALL)
    return match.group(1).strip() if match else ""


def _extract_sources(evidence_log_text: str) -> set:
    return set(re.findall(r"\(([\w\.\-]+), score=", evidence_log_text))


V2_VERIFIER_MARKER = "MUST see evidence from at least two distinct source documents"
V3_RESEARCHER_MARKER = "HARD RULE: if the evidence log shows the verifier's last verdict was"

CROSS_SOURCE_KEYWORDS = ["match", "compare", "agree", "cross-check", "cross check"]


class PromptAwareMockLLM:
    def __init__(self):
        self.call_count = 0

    async def __call__(self, messages, tools=None, json_mode=False, temperature=None, top_p=None):
        self.call_count += 1
        system_content = messages[0]["content"]
        user_content = messages[-1]["content"]
        is_verifier = "You are the VERIFIER agent" in system_content

        if is_verifier:
            strict = V2_VERIFIER_MARKER in system_content
            return self._verifier(user_content, strict)
        targets_missing = V3_RESEARCHER_MARKER in system_content
        return self._researcher(user_content, targets_missing)

    # ------------------------------------------------------------------ #
    def _verifier(self, user_content: str, strict: bool):
        question = _extract_question(user_content)
        sources = _extract_sources(user_content)
        is_cross_source = any(k in question.lower() for k in CROSS_SOURCE_KEYWORDS)

        if "TOOL FAILURE" in user_content or "no matching passages" in user_content:
            return _response({"verdict": "insufficient", "reason": "no usable evidence", "missing": "any evidence"})

        if is_cross_source and strict:
            if len(sources) >= 2:
                return _response({
                    "verdict": "conflicting",
                    "reason": f"found different figures across {len(sources)} sources ({', '.join(sorted(sources))})",
                })
            return _response({
                "verdict": "insufficient",
                "reason": "only one source seen so far for a cross-source question",
                "missing": "the IT support documentation's figure",
            })

        # v1 (not strict) or a single-source question: one match is enough
        return _response({"verdict": "sufficient", "reason": "evidence directly answers the question"})

    # ------------------------------------------------------------------ #
    def _researcher(self, user_content: str, targets_missing: bool):
        question = _extract_question(user_content)
        q_lower = question.lower()
        is_first_turn = "(empty" in user_content

        if is_first_turn:
            if q_lower.strip() in ("how much does it cost?", "how much does it cost"):
                return _response({
                    "action": "ask_clarification",
                    "clarification_question": "Which plan or service are you asking about?",
                })
            if "used 7 of my 24" in q_lower:
                return _response({"action": "calculate", "expression": "24 - 7"})
            return _response({"action": "search", "query": self._first_query(q_lower)})

        # not first turn: read the last verifier verdict/missing from the evidence log text
        last_verdict = self._last_verdict(user_content)
        last_missing = self._last_missing(user_content)
        prior_queries = set(re.findall(r"query=['\"](.+?)['\"]", user_content))

        if last_verdict == "sufficient":
            return _response(self._finish_payload(user_content, conflicting=False, evidence=self._last_note(user_content)))
        if last_verdict == "conflicting":
            return _response(self._finish_payload(user_content, conflicting=True))

        if "= 17" in user_content:
            return _response({"action": "finish", "answer": "You have 17 leave days left.", "sources": [], "confidence": "high"})

        # insufficient (or tool failure) -> decide next query
        if "search tool is unavailable" in user_content or "TOOL FAILURE" in user_content:
            return _response({
                "action": "finish",
                "answer": "I could not verify this because document search is unavailable right now.",
                "sources": [], "confidence": "low",
            })

        if targets_missing and last_missing:
            next_query = self._query_targeting(last_missing)
        else:
            # v1/v2 without the hard rule: just resend the same query that already ran (the bug)
            next_query = self._first_query(q_lower)

        if next_query in prior_queries and not targets_missing:
            pass  # this IS the bug being demonstrated - intentionally not deduplicated for v1/v2
        return _response({"action": "search", "query": next_query})

    # ------------------------------------------------------------------ #
    @staticmethod
    def _first_query(q_lower: str) -> str:
        if "business plan" in q_lower or "price" in q_lower:
            return "business plan price"
        if "password" in q_lower:
            return "reset password"
        if "ticket" in q_lower:
            return "IT ticket response time"
        if "parental" in q_lower:
            return "parental leave policy"
        return "paid annual leave days"

    @staticmethod
    def _query_targeting(missing_text: str) -> str:
        missing_lower = missing_text.lower()
        if "it support" in missing_lower:
            return "IT support documentation business plan price"
        return f"documentation about {missing_text}"

    @staticmethod
    def _last_verdict(user_content: str) -> str:
        verdicts = re.findall(r"verifier: (\w+)", user_content)
        return verdicts[-1] if verdicts else ""

    @staticmethod
    def _last_missing(user_content: str) -> str:
        missing = re.findall(r"missing: (.+)", user_content)
        return missing[-1].strip() if missing else ""

    @staticmethod
    def _last_note(user_content: str) -> str:
        """Pulls the factual content out of the most recent search's evidence-log note
        (e.g. '(acme_hr_policy.md, score=0.9) Employees get 24 days...') so the mock's
        finish answer actually states the fact it found, instead of a placeholder -
        a real model summarizes what it read; this stand-in should too."""
        notes = re.findall(r"result: \(([\w\.\-]+), score=[\d.]+\) (.+?)(?:\.\.\.)?\n", user_content)
        if not notes:
            return ""
        source, text = notes[-1]
        return text.rstrip(".") + "."

    @staticmethod
    def _finish_payload(user_content: str, conflicting: bool, evidence: str = "") -> dict:
        sources = sorted(_extract_sources(user_content))
        if conflicting:
            return {
                "action": "finish",
                "answer": (
                    "The sources disagree: the product FAQ lists the Business plan at $20/user/month, but "
                    "internal IT support notes say $25/user/month. Please confirm with sales before quoting "
                    "a customer - I can't verify which figure is current."
                ),
                "sources": sources,
                "confidence": "low",
            }
        return {
            "action": "finish",
            "answer": evidence or "Based on the verified evidence, here is the answer.",
            "sources": sources,
            "confidence": "high",
        }


def mock_search(query, top_k=None):
    """Same style as scripts/eval_agentic.py's mock_search, but the 'business plan'
    branch now deliberately returns ONE source at a time depending on query wording,
    so the loop genuinely needs two distinct searches to see both documents - see
    module docstring and README 9.b for why this matters for the experiment."""
    q = query.lower()
    if "it support" in q or "internal" in q or "support documentation" in q:
        return [{"text": "Internal IT support notes list the Business plan at $25/user/month.",
                  "source": "acme_it_support.txt", "score": 0.82}]
    if "business plan" in q or "price" in q:
        return [{"text": "The Business plan costs $20/user/month billed annually.",
                  "source": "acme_product_faq.md", "score": 0.85}]
    if "password" in q:
        return [{"text": "Go to Settings > Security > Reset password.", "source": "acme_it_support.txt", "score": 0.9}]
    if "ticket" in q:
        return [{"text": "IT tickets are answered within 1 business day.", "source": "acme_it_support.txt", "score": 0.7}]
    if "parental" in q:
        return [{"text": "Parental leave is 12 weeks paid.", "source": "acme_hr_policy.md", "score": 0.88}]
    return [{"text": "Employees get 24 days of paid annual leave, accrued monthly.",
              "source": "acme_hr_policy.md", "score": 0.9}]
