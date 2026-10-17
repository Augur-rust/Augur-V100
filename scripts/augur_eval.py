"""Accuracy evaluation through a running Augur endpoint: GPQA-Diamond, AIME 2025, MATH-500.

Questions come from local copies of the public datasets (paths given on the command line); nothing
is downloaded here. Requests use the product sampling contract with thinking on, one sample per
question. GPQA choices are shuffled with a fixed per-question seed; answers are graded by the
final "Answer: X" letter (GPQA) or the last \\boxed{} value (AIME, MATH-500). Results keep the
question id, the extracted answer, correctness and token counts; generated text is not saved.

  python scripts/augur_eval.py --endpoint http://127.0.0.1:8294 --api-key-file KEY --gpqa gpqa_diamond.jsonl --out eval/gpqa

The default budget is the full product output ceiling. Rerunning with a larger budget retries only
the questions that were truncated.
"""
import argparse
import hashlib
import json
from pathlib import Path
import random
import re
import time
import urllib.error
import urllib.request

GPQA_PROMPT = ('Answer the following multiple choice question. Think step by step, then end your reply with a '
               'line of the form "Answer: X", where X is one of A, B, C or D.\n\n{question}\n\n'
               'A) {A}\nB) {B}\nC) {C}\nD) {D}')
MATH_PROMPT = '{question}\n\nPlease reason step by step, and put your final answer within \\boxed{{}}.'


def gpqa(path):
    for row in map(json.loads, Path(path).read_text(encoding='utf-8').splitlines()):
        choices = [row['Correct Answer'], row['Incorrect Answer 1'], row['Incorrect Answer 2'], row['Incorrect Answer 3']]
        choices = [c.strip() for c in choices]
        order = list(range(4))
        random.Random(row['Record ID']).shuffle(order)
        letters = dict(zip('ABCD', (choices[i] for i in order)))
        yield dict(id=row['Record ID'], prompt=GPQA_PROMPT.format(question=row['Question'].strip(), **letters),
                   answer='ABCD'[order.index(0)], kind='letter')


def aime(paths):
    for path in paths:
        for i, row in enumerate(map(json.loads, Path(path).read_text(encoding='utf-8').splitlines())):
            yield dict(id=f'{Path(path).stem}-{i + 1}', prompt=MATH_PROMPT.format(question=row['question']),
                       answer=str(row['answer']).strip(), kind='integer')


def math500(path):
    for row in map(json.loads, Path(path).read_text(encoding='utf-8').splitlines()):
        yield dict(id=row['unique_id'], prompt=MATH_PROMPT.format(question=row['problem']), answer=row['answer'], kind='math')


def boxed(text):
    """Contents of the last \\boxed{...}, with balanced braces."""
    start = text.rfind('\\boxed{')
    if start < 0:
        return None
    depth, i = 0, start + len('\\boxed{') - 1
    for j in range(i, len(text)):
        depth += {'{': 1, '}': -1}.get(text[j], 0)
        if depth == 0:
            return text[i + 1:j].strip()
    return None


def normalize(value):
    value = value.strip().strip('$').replace('\\!', '').replace('\\,', '').replace('\\ ', '').replace(' ', '')
    value = re.sub(r'\\(?:text|mathrm|textbf)\{([^}]*)\}', r'\1', value)
    value = value.replace('\\dfrac', '\\frac').replace('\\tfrac', '\\frac').replace('\\left', '').replace('\\right', '')
    value = re.sub(r'\^\{?\\circ\}?|°', '', value)
    value = value.rstrip('.')
    if re.fullmatch(r'-?\d+(?:\.0+)?', value):
        value = str(int(float(value)))
    return value


def grade(item, content):
    if item['kind'] == 'letter':
        found = re.findall(r'Answer\s*[:：]\s*\(?([ABCD])\)?', content)
        got = found[-1] if found else None
        return got, got == item['answer']
    got = boxed(content)
    if got is None:
        return None, False
    if item['kind'] == 'integer':
        digits = re.sub(r'[^\d-]', '', normalize(got))
        return got, digits.lstrip('0') == item['answer'].lstrip('0') or digits == item['answer']
    return got, normalize(got) == normalize(item['answer'])


def ask(endpoint, key, prompt, max_tokens):
    payload = dict(model='quasar-27b', messages=[dict(role='user', content=prompt)], stream=False,
                   enable_thinking=True, reasoning_effort='xhigh', max_completion_tokens=max_tokens,
                   temperature=1.0, top_p=0.95, top_k=20, min_p=0, presence_penalty=0, frequency_penalty=0, seed=0)
    request = urllib.request.Request(endpoint + '/v1/chat/completions', data=json.dumps(payload).encode(),
        headers={'Authorization': 'Bearer ' + key, 'Content-Type': 'application/json'})
    with urllib.request.urlopen(request, timeout=7200) as response:
        return json.load(response)


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--endpoint', default='http://127.0.0.1:8294')
    p.add_argument('--api-key-file', type=Path, required=True)
    p.add_argument('--gpqa', type=Path)
    p.add_argument('--aime', type=Path, nargs='*')
    p.add_argument('--math500', type=Path)
    # The full product output ceiling (262,144-token context minus the prompt reserve). This model is
    # small and makes up for it by thinking long; a benchmark must not cut that short.
    p.add_argument('--max-tokens', type=int, default=249036)
    p.add_argument('--limit', type=int)
    p.add_argument('--out', type=Path, required=True)
    a = p.parse_args()
    items = list(gpqa(a.gpqa)) if a.gpqa else list(aime(a.aime)) if a.aime else list(math500(a.math500)) if a.math500 else []
    if not items:
        p.error('choose one of --gpqa, --aime or --math500')
    items = items[:a.limit] if a.limit else items
    key = a.api_key_file.read_text().strip()
    a.out.mkdir(parents=True, exist_ok=True)
    results = a.out / 'results.jsonl'
    def latest():
        rows = [json.loads(line) for line in results.read_text(encoding='utf-8').splitlines()] if results.exists() else []
        return {r['id']: r for r in rows}  # a rerun appends; the last row per question wins
    done = latest()
    for item in items:
        prior = done.get(item['id'])
        # Resume skips answered questions; a truncated one is rerun when the budget has grown.
        if prior and not (prior['finish'] == 'length' and prior.get('max_tokens', 32768) < a.max_tokens):
            continue
        started = time.perf_counter()
        try:
            response = ask(a.endpoint, key, item['prompt'], a.max_tokens)
        except urllib.error.HTTPError as error:
            # A refused or stopped generation (for example reasoning_repetition) is graded wrong.
            code = json.loads(error.read() or b'{}').get('error', {}).get('code', str(error.code))
            response = dict(choices=[dict(message={}, finish_reason='error:' + code)],
                            usage=dict(completion_tokens=0, completion_tokens_details=dict(reasoning_tokens=0)))
        message, usage = response['choices'][0]['message'], response['usage']
        got, correct = grade(item, message.get('content') or '')
        row = dict(id=item['id'], answer=item['answer'], extracted=got, correct=correct,
                   finish=response['choices'][0]['finish_reason'], completion_tokens=usage['completion_tokens'],
                   reasoning_tokens=usage['completion_tokens_details']['reasoning_tokens'],
                   wall_s=time.perf_counter() - started, max_tokens=a.max_tokens,
                   prompt_sha256=hashlib.sha256(item['prompt'].encode()).hexdigest())
        with results.open('a', encoding='utf-8') as stream:
            stream.write(json.dumps(row) + '\n')
        print(f"{item['id']:28s} {str(got)[:24]:24s} {'ok' if correct else '--'} {row['completion_tokens']:6d} tok "
              f"{row['wall_s']:6.1f} s", flush=True)
    rows = [r for r in latest().values() if r['id'] in {i['id'] for i in items}]
    summary = dict(questions=len(rows), correct=sum(r['correct'] for r in rows),
                   accuracy=sum(r['correct'] for r in rows) / len(rows),
                   truncated=sum(r['finish'] == 'length' for r in rows),
                   errors=sum(r['finish'].startswith('error') for r in rows),
                   mean_completion_tokens=sum(r['completion_tokens'] for r in rows) / len(rows),
                   max_tokens=a.max_tokens, max_completion_tokens_used=max(r['completion_tokens'] for r in rows))
    (a.out / 'summary.json').write_text(json.dumps(summary, indent=1) + '\n', encoding='utf-8')
    print(json.dumps(summary))


if __name__ == '__main__':
    main()
