"""Verifiable correctness checks for generated target answers.

Ported from Dr.Post-Training-Next (``SFT/data/target_verifiers.py``) and extended
for this repo's targets. Every domain verifier has the same two-stage shape::

    verifier = build_verifier("math_persona")
    context = verifier.prepare(row)      # None when the row cannot be verified
    result = verifier.verify(context, generated_text)

``prepare`` extracts the ground truth from the target row once; ``verify`` scores
one candidate against it. A row that cannot be verified is a first-class,
recorded outcome rather than a silent pass.

Domains
-------
``math`` (``math_persona``, ``math``, ``math_pool``, ``math_ref``, ...)
    Gold = the reference's last ``\\boxed{}``, else the Tulu-3 persona line
    ``Final Answer: The final answer is X. I hope it is correct.``, else a loose
    ``final answer ... X`` line. A candidate's answer is its ``\\boxed{}`` content
    (all boxes if several), else the same final-answer patterns. Scored with the
    pinned Math-Verify used by the MATH500 evaluator; when that fails on the
    free-form persona answers (``"20 plots and 60 miles"``) a numeric fallback
    requires every number of the gold answer to appear in the candidate answer,
    and a normalised string comparison covers number-free answers.
``if`` (``precise_if``)
    Dolci Precise-IF prompts embed IFEval's own instruction descriptions
    verbatim, so the constraints are recovered from the prompt text and checked
    with the vendored official verifier. Recovery is lenient: an unrecognised
    family is not checked (cannot invent a violation), and a prompt whose own
    reference fails the recovered constraints is treated as unverifiable.
``code`` (``mbpp``)
    The row's own ``test_list`` executed in a subprocess with wall-clock and
    address-space limits. This runs model-written code on the local machine.
"""

from __future__ import annotations

import ast
import os
import re
import subprocess
import sys
import tempfile
from dataclasses import dataclass, field
from typing import Any, List, Mapping, Optional, Sequence

from SFT.data.target_common import assistant_reference, base_target_name, infer_domain, user_prompt
from SFT.eval.tasks.common import clean_model_response

DEFAULT_EXECUTION_TIMEOUT_SEC = 10.0
DEFAULT_EXECUTION_MEMORY_MB = 4096


@dataclass(frozen=True)
class VerificationResult:
    """Outcome of scoring one candidate answer."""

    correct: bool
    status: str
    detail: Mapping[str, Any] = field(default_factory=dict)

    def to_json(self) -> dict:
        return {"correct": bool(self.correct), "status": self.status, **dict(self.detail)}


class TargetVerifier:
    """Interface shared by every domain verifier."""

    name = "base"

    def prepare(self, row: Mapping[str, Any]) -> Any:
        raise NotImplementedError

    def verify(self, context: Any, text: str) -> VerificationResult:
        raise NotImplementedError


# ---------------------------------------------------------------------------
# math
# ---------------------------------------------------------------------------


def extract_boxed_answer(text: str) -> Optional[str]:
    """Return the contents of the last ``\\boxed{...}``, brace-balanced."""
    boxes = extract_all_boxed(text)
    return boxes[-1] if boxes else None


def extract_all_boxed(text: str) -> List[str]:
    """Return the contents of every ``\\boxed{...}`` in order, brace-balanced."""
    marker = "\\boxed"
    out: List[str] = []
    cursor = 0
    while True:
        index = text.find(marker, cursor)
        if index == -1:
            return out
        cursor = index + len(marker)
        while cursor < len(text) and text[cursor].isspace():
            cursor += 1
        if cursor >= len(text) or text[cursor] != "{":
            continue
        depth = 0
        start = cursor + 1
        while cursor < len(text):
            if text[cursor] == "{":
                depth += 1
            elif text[cursor] == "}":
                depth -= 1
                if depth == 0:
                    out.append(text[start:cursor])
                    cursor += 1
                    break
            cursor += 1
        else:
            return out


# Tulu 3 persona format: "Final Answer: The final answer is $X$. I hope it is correct."
_TULU_FINAL_RE = re.compile(
    r"Final Answer:\s*The final answer is\s*(?P<ans>.+?)\.?\s*I hope it is correct\.?\s*$",
    re.S | re.I,
)
_LOOSE_FINAL_RE = re.compile(r"final answer(?:\s+is)?\s*[:=]?\s*(?P<ans>[^\n]+)", re.I)
_NUMBER_RE = re.compile(r"-?\d+(?:,\d{3})*(?:\.\d+)?")


def extract_final_answer(text: str, *, prefer: str = "boxed") -> Optional[str]:
    """Final answer of a solution: ``\\boxed{}`` content(s) or a final-answer line.

    ``prefer="boxed"`` (candidates written under a boxed instruction) looks at
    ``\\boxed{}`` first; ``prefer="final_line"`` (persona references) looks at the
    Tulu line first. Several boxes are joined with ``" and "``.
    """
    text = text.strip()
    boxes = [b.strip() for b in extract_all_boxed(text) if b.strip()]
    tulu = _TULU_FINAL_RE.search(text)
    boxed_answer = " and ".join(boxes) if boxes else None
    tulu_answer = tulu.group("ans").strip() if tulu else None
    order = (boxed_answer, tulu_answer) if prefer == "boxed" else (tulu_answer, boxed_answer)
    for answer in order:
        if answer:
            return answer
    hits = list(_LOOSE_FINAL_RE.finditer(text))
    if hits:
        return hits[-1].group("ans").strip().rstrip(".").strip() or None
    return None


def answer_numbers(answer: str) -> List[str]:
    """Normalised numbers in an answer string (``1,000`` -> ``1000``, ``2.50`` -> ``2.5``)."""
    out = []
    for raw in _NUMBER_RE.findall(answer):
        raw = raw.replace(",", "")
        try:
            out.append(f"{float(raw):.6g}")
        except ValueError:
            out.append(raw)
    return out


def normalize_answer_text(answer: str) -> str:
    """Strip LaTeX delimiters / ``\\text{}`` / spaces so ``$x=3$`` == ``\\(x = 3\\)``."""
    s = answer.strip()
    s = re.sub(r"\\text\{([^{}]*)\}", r"\1", s)
    s = re.sub(r"\\(?:left|right|,|;|!|displaystyle)", "", s)
    s = s.replace("\\(", "").replace("\\)", "").replace("\\[", "").replace("\\]", "").replace("$", "")
    s = re.sub(r"\s+", "", s).rstrip(".").lower()
    return s


class MathAnswerVerifier(TargetVerifier):
    """Math-Verify on the final answer, with numeric / string fallbacks for free-form golds."""

    name = "math_final_answer"

    def __init__(self) -> None:
        from SFT.eval.tasks.math500 import _load_scorer

        self._score, self.scorer_info = _load_scorer()

    def prepare(self, row: Mapping[str, Any]) -> Optional[dict]:
        gold = extract_final_answer(assistant_reference(row), prefer="final_line")
        if not gold:
            return None
        return {"gold_answer": gold, "gold_numbers": answer_numbers(gold)}

    def verify(self, context: Mapping[str, Any], text: str) -> VerificationResult:
        gold = context["gold_answer"]
        response = clean_model_response(text)
        boxes = [b.strip() for b in extract_all_boxed(response) if b.strip()]
        candidate = extract_final_answer(response, prefer="boxed")
        if not candidate:
            return VerificationResult(False, "no_final_answer", {"gold_answer": gold})
        detail = {"gold_answer": gold, "candidate_answer": candidate[:300]}

        # 1) Math-Verify: the gold against each box (multi-part answers) and against the joined answer.
        statuses = []
        for pred in (boxes or [candidate]) + ([candidate] if boxes and len(boxes) > 1 else []):
            scored = self._score(gold, f"\\boxed{{{pred}}}")
            statuses.append(scored["status"])
            if scored["correct"]:
                return VerificationResult(True, "correct", {**detail, "method": "math_verify"})

        # 2) Numeric fallback: every number of the gold answer appears in the candidate answer.
        gold_numbers = context["gold_numbers"]
        if gold_numbers:
            candidate_numbers = set(answer_numbers(candidate))
            if all(number in candidate_numbers for number in gold_numbers):
                return VerificationResult(True, "correct", {**detail, "method": "numeric_match"})
        # 3) Number-free answers: normalised string equality.
        elif normalize_answer_text(gold) == normalize_answer_text(candidate):
            return VerificationResult(True, "correct", {**detail, "method": "string_match"})

        status = "incorrect" if "incorrect" in statuses else statuses[0]
        return VerificationResult(False, status, detail)


# ---------------------------------------------------------------------------
# code (MBPP: execute the row's own asserts)
# ---------------------------------------------------------------------------

_HARNESS = """\
import resource, sys
resource.setrlimit(resource.RLIMIT_AS, ({memory_bytes}, {memory_bytes}))
resource.setrlimit(resource.RLIMIT_NPROC, (64, 64))
"""


class MbppVerifier(TargetVerifier):
    """Execute the row's own MBPP asserts against the candidate program."""

    name = "mbpp_tests"

    def __init__(
        self,
        *,
        timeout_sec: float = DEFAULT_EXECUTION_TIMEOUT_SEC,
        memory_mb: int = DEFAULT_EXECUTION_MEMORY_MB,
        python_executable: Optional[str] = None,
    ) -> None:
        self.timeout_sec = float(timeout_sec)
        self.memory_mb = int(memory_mb)
        self.python_executable = python_executable or sys.executable

    def prepare(self, row: Mapping[str, Any]) -> Optional[dict]:
        metadata = row.get("metadata") or {}
        tests = metadata.get("test_list") or []
        if not isinstance(tests, Sequence) or isinstance(tests, (str, bytes)) or not tests:
            return None
        return {"test_list": [str(t) for t in tests], "test_setup_code": str(metadata.get("test_setup_code") or "")}

    def _program(self, context: Mapping[str, Any], code: str) -> str:
        parts = [
            _HARNESS.format(memory_bytes=self.memory_mb * 1024 * 1024),
            code,
            "",
            context["test_setup_code"],
            "",
            *context["test_list"],
            "",
            "print('__DRPT_TESTS_PASSED__')",
        ]
        return "\n".join(parts)

    def verify(self, context: Mapping[str, Any], text: str) -> VerificationResult:
        from SFT.eval.tasks.mbpp_plus import extract_code

        code = extract_code(clean_model_response(text))
        if not code.strip():
            return VerificationResult(False, "empty_program")
        try:
            ast.parse(code)
        except SyntaxError as exc:
            return VerificationResult(False, "syntax_error", {"error": str(exc)})
        with tempfile.TemporaryDirectory() as workdir:
            script = os.path.join(workdir, "candidate_check.py")
            with open(script, "w", encoding="utf-8") as handle:
                handle.write(self._program(context, code))
            try:
                completed = subprocess.run(
                    [self.python_executable, "-I", "-S", script],
                    cwd=workdir, capture_output=True, text=True, timeout=self.timeout_sec,
                    env={"PATH": "/usr/bin:/bin", "HOME": workdir},
                )
            except subprocess.TimeoutExpired:
                return VerificationResult(False, "timeout")
        if "__DRPT_TESTS_PASSED__" in completed.stdout:
            return VerificationResult(True, "correct")
        return VerificationResult(
            False, "tests_failed" if completed.returncode else "no_success_marker",
            {"stderr": completed.stderr[-500:]},
        )


# ---------------------------------------------------------------------------
# precise_if (recover the IFEval constraints embedded in the prompt)
# ---------------------------------------------------------------------------

_KEYWORD = "keywords:"
_LANGUAGE = "language:"
_LENGTH = "length_constraints:"
_CONTENT = "detectable_content:"
_FORMAT = "detectable_format:"
_COMBINATION = "combination:"
_STARTEND = "startend:"
_CHANGE_CASES = "change_case:"
_PUNCTUATION = "punctuation:"

_RELATIONS = {"at least": "at least", "less than": "less than", "at most": "less than"}

# Every pattern below is the literal description IFEval's own ``build_description``
# emits, so a match recovers the exact constraint the prompt was generated from.
_LITERAL_CONSTRAINTS = (
    ("Entire output should be wrapped in JSON format", _FORMAT + "json_format"),
    ("Your answer must contain a title, wrapped in double angular brackets", _FORMAT + "title"),
    ("Wrap your entire response with double quotation marks", _STARTEND + "quotation"),
    ("refrain from the use of any commas", _PUNCTUATION + "no_comma"),
    ("Your entire response should be in English, and in all capital letters", _CHANGE_CASES + "english_capital"),
    ("Your entire response should be in English, and in all lowercase letters", _CHANGE_CASES + "english_lowercase"),
    ("separated by 6 asterisk symbols", _COMBINATION + "two_responses"),
)

_LIST_RE = r"\[(?P<items>[^\]]*)\]"
_NUM_RE = r"(?P<num>\d+)"
_REL_RE = r"(?P<relation>at least|less than|at most)"


def _parse_string_list(raw: str) -> List[str]:
    """Parse the ``['a', 'b']`` rendering IFEval descriptions embed."""
    try:
        parsed = ast.literal_eval("[" + raw + "]")
    except (SyntaxError, ValueError):
        return [item.strip().strip("'\"") for item in raw.split(",") if item.strip()]
    if isinstance(parsed, (list, tuple)):
        return [str(item) for item in parsed]
    return [str(parsed)]


def recover_if_constraints(prompt: str) -> List[tuple]:
    """Recover ``(instruction_id, kwargs)`` pairs from an IFEval-style prompt.

    Only families whose description template is matched verbatim are returned;
    unrecognised constraint text is skipped (lenient rather than wrong).
    """
    recovered: List[tuple] = []

    def add(instruction_id: str, **kwargs: Any) -> None:
        if not any(existing == instruction_id for existing, _ in recovered):
            recovered.append((instruction_id, kwargs))

    for literal, instruction_id in _LITERAL_CONSTRAINTS:
        if literal in prompt:
            add(instruction_id)

    match = re.search(rf"Include keywords {_LIST_RE} in the response", prompt)
    if match:
        add(_KEYWORD + "existence", keywords=_parse_string_list(match.group("items")))

    match = re.search(rf"Do not include keywords {_LIST_RE} in the response", prompt)
    if match:
        add(_KEYWORD + "forbidden_words", forbidden_words=_parse_string_list(match.group("items")))

    match = re.search(rf"the word (?P<keyword>\S+) should appear {_REL_RE} {_NUM_RE} times", prompt)
    if match:
        add(_KEYWORD + "frequency", keyword=match.group("keyword").strip("'\"."),
            relation=_RELATIONS[match.group("relation")], frequency=int(match.group("num")))

    match = re.search(rf"letter (?P<letter>[a-zA-Z]) should appear {_REL_RE} {_NUM_RE} times", prompt)
    if match:
        add(_KEYWORD + "letter_frequency", letter=match.group("letter"),
            let_relation=_RELATIONS[match.group("relation")], let_frequency=int(match.group("num")))

    match = re.search(rf"words with all capital letters should appear {_REL_RE} {_NUM_RE} times", prompt)
    if match:
        add(_CHANGE_CASES + "capital_word_frequency", capital_relation=_RELATIONS[match.group("relation")],
            capital_frequency=int(match.group("num")))

    match = re.search(rf"Your response should contain {_REL_RE} {_NUM_RE} sentences", prompt)
    if match:
        add(_LENGTH + "number_sentences", relation=_RELATIONS[match.group("relation")],
            num_sentences=int(match.group("num")))

    match = re.search(rf"Answer with {_REL_RE} {_NUM_RE} words", prompt)
    if match:
        add(_LENGTH + "number_words", relation=_RELATIONS[match.group("relation")], num_words=int(match.group("num")))

    # The nth-paragraph family reuses the "There should be N paragraphs" opener with a
    # different separator, so it must win over the plain divider form.
    match = re.search(
        r"There should be (?P<num>\d+) paragraphs\. Paragraphs and only paragraphs are "
        r"separated with each other by two new lines[^.]*\. "
        r"Paragraph (?P<nth>\d+) must start with word (?P<word>[^.\s]+)",
        prompt,
    )
    if match:
        add(_LENGTH + "nth_paragraph_first_word", num_paragraphs=int(match.group("num")),
            nth_paragraph=int(match.group("nth")), first_word=match.group("word"))
    else:
        match = re.search(
            r"There should be (?P<num>\d+) paragraphs\. Paragraphs are separated with the markdown divider: \*\*\*",
            prompt,
        )
        if match:
            add(_LENGTH + "number_paragraphs", num_paragraphs=int(match.group("num")))

    match = re.search(rf"must contain at least {_NUM_RE} placeholders represented by square brackets", prompt)
    if match:
        add(_CONTENT + "number_placeholders", num_placeholders=int(match.group("num")))

    match = re.search(r"add a postscript starting with (?P<marker>P\.?P\.?S|P\.?S\.?)", prompt)
    if match:
        add(_CONTENT + "postscript", postscript_marker=match.group("marker"))

    match = re.search(rf"must contain exactly {_NUM_RE} bullet points", prompt)
    if match:
        add(_FORMAT + "number_bullet_lists", num_bullets=int(match.group("num")))

    match = re.search(rf"Highlight at least {_NUM_RE} sections in your answer", prompt)
    if match:
        add(_FORMAT + "number_highlighted_sections", num_highlights=int(match.group("num")))

    match = re.search(
        r"Your response must have (?P<num>\d+) sections\. Mark the beginning of each section with (?P<spliter>\S+) X",
        prompt,
    )
    if match:
        add(_FORMAT + "multiple_sections", section_spliter=match.group("spliter"), num_sections=int(match.group("num")))

    match = re.search(
        r"Finish your response with this exact phrase (?P<phrase>.+?)\. No other words should follow this phrase",
        prompt, re.DOTALL,
    )
    if match:
        add(_STARTEND + "end_checker", end_phrase=match.group("phrase").strip())

    match = re.search(r"Your ENTIRE response should be in (?P<language>[A-Za-z]+) language", prompt)
    if match:
        code = _language_code(match.group("language"))
        if code:
            add(_LANGUAGE + "response_language", language=code)

    if "Answer with one of the following options:" in prompt:
        add(_FORMAT + "constrained_response")

    return recovered


def _language_code(language_name: str) -> Optional[str]:
    from SFT.eval.tasks.ifeval_lib.instructions import _LANGUAGES

    wanted = language_name.strip().lower()
    for code, name in _LANGUAGES.items():
        if str(name).strip().lower() == wanted:
            return code
    return None


class IfConstraintsVerifier(TargetVerifier):
    """Check recovered IFEval constraints with the vendored official verifier."""

    name = "if_constraints"

    def __init__(self, *, strict: bool = True) -> None:
        self.strict = bool(strict)

    def prepare(self, row: Mapping[str, Any]) -> Optional[dict]:
        prompt = user_prompt(row)
        recovered = recover_if_constraints(prompt)
        if not recovered:
            return None
        context = {
            "prompt": prompt,
            "instruction_id_list": [instruction_id for instruction_id, _ in recovered],
            "kwargs_list": [kwargs for _, kwargs in recovered],
        }
        # Recovery that the reference itself fails is recovery that went wrong
        # (e.g. the "repeat the request" prompts): treat the row as unverifiable.
        if not self.verify(context, assistant_reference(row)).correct:
            return None
        return context

    def verify(self, context: Mapping[str, Any], text: str) -> VerificationResult:
        from SFT.eval.tasks.ifeval_scoring import evaluate_instruction_following

        scored = evaluate_instruction_following(
            prompt=context["prompt"],
            response=clean_model_response(text),
            instruction_id_list=context["instruction_id_list"],
            kwargs_list=context["kwargs_list"],
            strict=self.strict,
        )
        correct = bool(scored["follow_all_instructions"])
        return VerificationResult(
            correct, "correct" if correct else "constraint_violation",
            {"instruction_id_list": list(scored["instruction_id_list"]),
             "follow_instruction_list": [bool(x) for x in scored["follow_instruction_list"]]},
        )


# ---------------------------------------------------------------------------


def build_verifier(target: str, **kwargs: Any) -> TargetVerifier:
    """Return the verifier for a target (variants ``<base>_gen*`` / ``<base>_rw*`` use the base's)."""
    domain = infer_domain(target)
    if domain == "if":
        return IfConstraintsVerifier(**kwargs)
    if domain == "code":
        return MbppVerifier(**kwargs)
    return MathAnswerVerifier()


__all__ = [
    "IfConstraintsVerifier", "MathAnswerVerifier", "MbppVerifier", "TargetVerifier", "VerificationResult",
    "answer_numbers", "base_target_name", "build_verifier", "extract_all_boxed", "extract_boxed_answer",
    "extract_final_answer", "recover_if_constraints",
]
