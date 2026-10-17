"""Augur request-local thinking policy and streaming channel separation."""
from __future__ import annotations
import copy
import time
import qwen35_chat as C

DEFAULT_LANGUAGE = 'en'
LANGUAGES = ('en', 'zh')
POLICY_VERSION = 'quasar-thinking-language-effort-v2'
INSTRUCTIONS = {
    'en': ('Use English for your reasoning by default. If the user explicitly requests another '
           'reasoning language, follow that request. Write the final answer in the language '
           'requested by the user; the reasoning language does not determine the answer language.'),
    'zh': ('Use Chinese for your reasoning. Write the final answer in the language requested '
           'by the user; the reasoning language does not determine the answer language.'),
}


ENGLISH_GATE_POLICY = 'english_thinking_gate045_v1'
GLOBAL_GATE_POLICY = 'global_gate045_v1'
ENGLISH_GATE_SHA256 = '0bcb694a4e84aa304d888829742fdef92941771ac5e97917ca5dc4388f6821c7'


def selector_experiment(environment, dflash):
    mode = environment.get('AUGUR_DFLASH_ENGLISH_GATE')
    if mode is None:
        return None
    C.require(mode in (ENGLISH_GATE_POLICY, GLOBAL_GATE_POLICY) and dflash
              and ((environment.get('AUGUR_DFLASH_SELECTOR_GATE_F32_SHA256') == ENGLISH_GATE_SHA256
                    and environment.get('AUGUR_DFLASH_SELECTOR_GATE_SOURCE') != 'embedded_gguf')
                   or (mode == GLOBAL_GATE_POLICY
                       and environment.get('AUGUR_DFLASH_SELECTOR_GATE_SOURCE') == 'embedded_gguf'
                       and not environment.get('AUGUR_DFLASH_SELECTOR_GATE_F32')
                       and not environment.get('AUGUR_DFLASH_SELECTOR_GATE_F32_SHA256')))
              and environment.get('AUGUR_DFLASH_SELECTOR_GATE_ACCUMULATION', 'fp32') == 'fp32'
              and environment.get('AUGUR_DFLASH_GATE') == 'fixed_b8',
              'English selector experiment requires pinned F32 gate and fixed B8')
    return mode


def selector_request(experiment, thinking, has_images):
    if experiment == GLOBAL_GATE_POLICY:
        return experiment if not has_images else None
    return experiment if experiment and thinking['enabled'] and thinking['language'] == 'en' and not has_images else None


def policy(enabled=False, language=DEFAULT_LANGUAGE, effort=None):
    C.require(type(enabled) is bool, 'enable_thinking must be a boolean')
    C.require(type(language) is str and language in LANGUAGES,
              'thinking_language must be en or zh')
    C.require(effort is None or type(effort) is str and effort in ('off', 'none', 'low', 'medium', 'high', 'xhigh'),
              'reasoning_effort must be off, none, low, medium, high or xhigh')
    C.require(not enabled or effort not in ('off', 'none'),
              'enable_thinking=true conflicts with reasoning_effort=off/none')
    C.require(enabled or effort in (None, 'off', 'none'),
              'enable_thinking=false conflicts with an enabled reasoning_effort')
    # The supplied model template supports low/medium/xhigh; DSH's high is
    # an alias for medium. Omitted effort preserves the existing xhigh prompt.
    resolved = ('medium' if effort == 'high' else effort or 'xhigh') if enabled else None
    return dict(enabled=enabled, language=language if enabled else None,
                effort=resolved, version=POLICY_VERSION)

def instruct(messages, language):
    C.require(language in LANGUAGES, 'unsupported thinking language')
    result = copy.deepcopy(messages)
    instruction = INSTRUCTIONS[language]
    if result and result[0]['role'] == 'system':
        result[0]['content'] += '\n\n' + instruction
    else:
        result.insert(0, C.ChatTemplate.message('system', instruction))
    return result


class RepetitionGuard:
    """Bounded check for long exact cycles in reasoning, independent of its language.

    At least six copies and 8192 repeated characters are required. The guard
    only rejects an unfinished generation; it never fabricates an answer or
    changes sampling, weights, tools, or the allowed length of useful reasoning.
    """
    # Retain six copies of a multi-paragraph cycle up to 16 Ki characters.
    # A 4 Ki period ceiling misses long exact planning loops.
    window = 131072
    interval = 256
    min_period = 128
    max_period = 16384
    min_copies = 6
    min_repeated = 8192

    def __init__(self):
        self.tail, self.total, self.since_check = '', 0, 0

    def push(self, text):
        while text:
            take = min(len(text), self.interval - self.since_check)
            part, text = text[:take], text[take:]
            self.tail = (self.tail + part)[-self.window:]
            self.total += take
            self.since_check += take
            if self.since_check == self.interval:
                self.since_check = 0
                self.check()

    def check(self):
        tail = self.tail
        if len(tail) < self.min_repeated:
            return
        # Search a fixed-size suffix; positions stay comparable even when a
        # network delta ends in the middle of a repeated paragraph.
        anchor = tail[-self.min_period:]
        end = len(tail) - self.min_period
        start = max(0, len(tail) - self.min_period - self.max_period)
        for _ in range(8):
            previous = tail.rfind(anchor, start, end)
            if previous < 0:
                break
            period = len(tail) - self.min_period - previous
            copies = max(self.min_copies, (self.min_repeated + period - 1) // period)
            repeated = copies * period
            if repeated <= len(tail) and tail[-repeated:] == tail[-period:] * copies:
                raise C.GenerationFailure('reasoning_repetition',
                    reasoning_characters=self.total, period_characters=period,
                    repeated_characters=repeated, copies=copies)
            end = previous + self.min_period - 1


class Output:
    """The input template has already opened <think>; only final text reaches tools."""
    close = '</think>'

    def __init__(self, content_emit, reasoning_emit, loop_guard=True):
        self.content_emit, self.reasoning_emit = content_emit, reasoning_emit
        self.pending, self.reasoning, self.content = '', '', ''
        self.closed = False
        self.loop_guard = RepetitionGuard() if loop_guard else None
        self.started = time.perf_counter()
        self.first_reasoning_s = self.first_content_s = None

    def _reasoning(self, text):
        if text:
            if self.loop_guard is not None:
                self.loop_guard.push(text)
            if self.first_reasoning_s is None:
                self.first_reasoning_s = time.perf_counter() - self.started
            self.reasoning += text
            self.reasoning_emit(dict(reasoning_content=text))

    def _content(self, text):
        if text:
            if self.first_content_s is None:
                self.first_content_s = time.perf_counter() - self.started
            self.content += text
            self.content_emit(text)

    def push(self, text):
        C.require(type(text) is str, 'thinking stream must contain text')
        if self.closed:
            self._content(text)
            return
        self.pending += text
        offset = self.pending.find(self.close)
        if offset >= 0:
            self._reasoning(self.pending[:offset])
            suffix = self.pending[offset + len(self.close):]
            self.pending = ''
            self.closed = True
            self._content(suffix)
            return
        # Hold only a possible split delimiter; never withhold an arbitrary
        # trailing paragraph or mistake tool XML in reasoning for a real call.
        keep = next((n for n in range(min(len(self.close)-1, len(self.pending)), 0, -1)
                     if self.pending.endswith(self.close[:n])), 0)
        ready = self.pending[:-keep] if keep else self.pending
        self.pending = self.pending[-keep:] if keep else ''
        self._reasoning(ready)

    def finish(self, stop):
        C.require(stop in ('eos', 'length'), 'unsupported thinking stop')
        if not self.closed:
            self._reasoning(self.pending)
            self.pending = ''
            C.require(stop == 'length', 'model ended before closing its thinking channel')
        return self.content

    def token_counts(self, tokenizer, ids):
        # Count actual generated IDs, never retokenize text or include a seed.
        delimiter = tokenizer.token_to_id(self.close)
        C.require(type(delimiter) is int, 'thinking close token is absent from tokenizer')
        if self.closed:
            C.require(delimiter in ids, 'thinking text/token boundary mismatch')
            boundary = ids.index(delimiter)
            return dict(reasoning_tokens=boundary, reasoning_boundary_tokens=1,
                        answer_tokens=len(ids)-boundary-1)
        C.require(delimiter not in ids, 'unclosed thinking text/token boundary mismatch')
        return dict(reasoning_tokens=len(ids), reasoning_boundary_tokens=0, answer_tokens=0)

    def timings(self):
        return dict(thinking_complete=self.closed,
                    bridge_first_reasoning_s=self.first_reasoning_s,
                    bridge_first_answer_content_s=self.first_content_s)
