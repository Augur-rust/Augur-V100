"""Long-context decode suite in the NInfer V100 benchmark format.

Prompt construction is adapted from NInfer bench/v100/ctx_prompt.py (Apache License 2.0), itself
adapted from huangserva/ninfer-v100-tpx tools/bench/v100. A prompt is N numbered filler blocks
followed by a fixed task:

  code    the NInfer Python module fragments and task, unchanged, so token counts match
          NInfer's published rows (29 blocks -> 7,353 tokens, 722 -> 186,420, 1007 -> 260,283)
  en-doc  NInfer's Chinese technical document translated to English, block for block

Levels use NInfer's block-offset idea so no two prompts share a cacheable prefix: the 8K level keeps
offset 0 (NInfer's exact 7,353-token code prompt), longer levels use offsets 10000, 20000, 30000,
which keep every block number at five digits and the counts recalibrated per level.

Requests decode 1,024 tokens with thinking off, as NInfer does, but keep Augur's product
sampling contract (temperature 1, top-p 0.95, top-k 20, exact sampling); it is not relaxed here.

  python scripts/augur_bench_longctx.py --tokenizer MODEL/tokenizer.json --out bench/long-context.json
"""
import argparse
import json
from pathlib import Path

CODE_SENT = (
    "# ---- module part {n} ----\n"
    "class TaskQueue{n}:\n"
    "    def __init__(self, max_concurrency: int = {n}) -> None:\n"
    "        self._pq: list[tuple[int, int, Task]] = []\n"
    "        self._running: set[int] = set()\n"
    "        self._max = max_concurrency\n"
    "        self._seq = 0\n\n"
    "    def submit(self, task: 'Task', priority: int = 0) -> int:\n"
    "        self._seq += 1\n"
    "        heapq.heappush(self._pq, (-priority, self._seq, task))\n"
    "        return self._seq\n\n"
    "    async def _drain{n}(self) -> None:\n"
    "        while self._pq and len(self._running) < self._max:\n"
    "            _, _, task = heapq.heappop(self._pq)\n"
    "            self._running.add(task.id)\n"
    "            asyncio.create_task(self._guard{n}(task))\n\n"
    "    async def _guard{n}(self, task: 'Task') -> None:\n"
    "        try:\n"
    "            await asyncio.wait_for(task.run(), timeout=task.timeout)\n"
    "        finally:\n"
    "            self._running.discard(task.id)\n"
)
CODE_ASK = ("\n\n# === 以上是模块全文 ===\n"
            "# 请参照上面的代码风格，补全一个带优先级队列、并发上限、超时重试和优雅关闭的\n"
            "# 异步任务调度器主循环函数（含类型注解和中文注释），约 120 行。\n")
EN_SENT = (
    "Section {n}. Memory bandwidth and autoregressive decoding. During inference of a large language "
    "model, each decode step produces only one token, yet it must read every weight from GPU memory into "
    "the compute units once. For a model with 27 billion parameters, even with 4-bit quantization the "
    "weights still occupy about 15 GB; at a memory bandwidth of 900 GB per second the theoretical decode "
    "ceiling is only about 60 tokens per second, which is the so-called memory wall. Batched inference can "
    "raise arithmetic intensity, because one copy of the weights can serve several requests; but in "
    "single-stream serving the arithmetic intensity stays close to one, and the bottleneck always falls on "
    "bandwidth. The conclusion of section {n} is: decode speed can only be raised in three directions, by "
    "shrinking the weights, by improving bandwidth utilization, or by reducing the number of reads per step.\n"
)
EN_ASK = ("\n\n=== End of the document ===\n"
          "Read the entire document above and write an English synthesis report of about 600 words in three "
          "parts: (1) the core argument these sections keep emphasizing; (2) the quantitative premises they all "
          "rely on, with concrete numbers; (3) to double decode speed on this kind of hardware, which part should "
          "be changed first, ranked by benefit. Do not summarize section by section; give your own overall judgment.")
KINDS = {'code': (CODE_SENT, CODE_ASK), 'en-doc': (EN_SENT, EN_ASK)}
# NInfer's published code block counts; en-doc counts are calibrated below to the matching NInfer
# Chinese prompt lengths (7,792 / 31,924 / 193,104 tokens and the ~260K level).
CODE_LEVELS = {'8K': 29, '32K': 124, '186K': 722, '256K': 1007}
EN_TARGETS = {'8K': 7792, '32K': 31924, '193K': 193104, '256K': 260283}


def build(kind, blocks, offset=0):
    sentence, ask = KINDS[kind]
    return ''.join(sentence.format(n=i + offset) for i in range(1, blocks + 1)) + ask


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--tokenizer', type=Path, required=True)
    p.add_argument('--out', type=Path, required=True)
    p.add_argument('--template-tokens', type=int, default=11, help='chat-template tokens around one user turn (NInfer prompt_n = text + 11)')
    a = p.parse_args()
    from tokenizers import Tokenizer
    tok = Tokenizer.from_file(str(a.tokenizer))
    count = lambda text: len(tok.encode(text, add_special_tokens=False).ids) + a.template_tokens
    code_targets = {level: count(build('code', blocks)) for level, blocks in CODE_LEVELS.items()}
    cases = []
    for kind, targets in (('code', code_targets), ('en-doc', EN_TARGETS)):
        for index, (level, target) in enumerate(targets.items()):
            offset = 10000 * index
            per_block = (count(build(kind, 101, offset)) - count(build(kind, 1, offset))) / 100
            blocks = max(1, round((target - count(build(kind, 0, offset))) / per_block))
            while count(build(kind, blocks, offset)) > target:
                blocks -= 1
            cases.append((kind, level, blocks, offset))
    suite = dict(schema='augur-bench-suite-v1', name='long-context-ninfer-format',
        description='NInfer V100 long-context prompts (code unchanged, Chinese document translated to English), '
                    'thinking off, 1,024 output tokens, product sampling contract.',
        request=dict(model='quasar-27b', stream=False, enable_thinking=False, temperature=1.0, top_p=0.95, top_k=20,
                     min_p=0, presence_penalty=0, frequency_penalty=0, seed=0),
        cases=[])
    for kind, level, blocks, offset in cases:
        prompt = build(kind, blocks, offset)
        suite['cases'].append(dict(name=f'{kind}_{level.lower()}', kind=kind, blocks=blocks, offset=offset,
            prompt_tokens_estimate=count(prompt), max_completion_tokens=1024,
            messages=[dict(role='user', content=prompt)]))
        print(f'{kind:7s} {level:5s} blocks {blocks:5d} offset {offset:5d} ~{count(prompt):7d} tokens', flush=True)
    a.out.write_text(json.dumps(suite, ensure_ascii=False) + '\n', encoding='utf-8')


if __name__ == '__main__':
    main()
