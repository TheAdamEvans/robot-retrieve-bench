"""text → QueryProgram generation. dev = demo5 canonicals + regressions (tune on these); test = compose_test +
demo5_para (report only, never tune). Cases are promoted from the benchmark by `alloy-evals add --from-query`.

Hill-climb knobs arrive in ctx.params: prompt (prompts/<v>.md), model, reasoning. Programs are cached per
(utterance, prompt, model, reasoning), so re-running a configuration costs nothing."""
import json
import time

from alloy_server.catalog.registry import REGISTRY
from alloy_server.evalkit import REPO, Dataset, Gate, exact, excludes, includes, measured
from alloy_server.io import cost
from alloy_server.programs import openai_generator as G

REQUIRES = ["bundle", "llm"]
DATASETS = [Dataset("dev", "cases_dev.jsonl", split="dev"), Dataset("test", "cases_test.jsonl", split="test")]
SCORERS = [exact("valid"), includes("features"), excludes("features"), exact("partial"), exact("events_empty"),
           exact("leak_free"), measured("attempts"), measured("tokens"), measured("latency_ms")]
GATES = [Gate("valid.accuracy", ">=", 0.9), Gate("features.recall", ">=", 0.8), Gate("features_none.clean", ">=", 0.9),
         Gate("leak_free.accuracy", ">=", 0.9)]

LEAK_WORDS = ("feature", "registry", "index", "detector", "this system", "not recorded", "sensor")
_gens: dict = {}


def generator(ctx):
    p = ctx.params
    key = (p.get("prompt", G.PROMPT), p.get("model", G.MODEL), p.get("reasoning", G.REASONING))
    if key not in _gens:
        shots = json.loads((ctx.bundle.root / "prompts" / "fewshots.json").read_text())
        _gens[key] = G.OpenAIProgramGenerator(ctx.bundle, [(s["utterance"], s["program"]) for s in shots],
                                              cache_dir=REPO / "results" / "evals" / "program_cache",
                                              prompt=key[0], model=key[1], reasoning=key[2])
    return _gens[key]


def run_case(case, ctx):
    gen, utt = generator(ctx), case["input"]["utterance"]
    t = time.perf_counter()
    with cost.scope("program_generation") as s:
        prog, diag = gen(utt, ctx.bundle)
    ms = (time.perf_counter() - t) * 1e3
    hit = json.loads((gen.cache_dir / f"{gen._key(utt)}.json").read_text()) if diag.cache_hit else {}
    usage = hit.get("usage") or {"prompt_tokens": s.prompt_tokens, "completion_tokens": s.completion_tokens}
    obs = {"valid": prog is not None, "attempts": diag.attempts if not diag.cache_hit else hit.get("attempts", 1),
           "tokens": usage.get("prompt_tokens", 0) + usage.get("completion_tokens", 0),
           "latency_ms": ms, "cache_hit": diag.cache_hit, "errors": list(diag.errors)}
    if prog is None:
        return obs
    texts = [ev.doc for ev in prog.events] + [r.doc for r in prog.relations] + list(prog.unexpressible)
    obs.update(features=sorted({ev.feature for ev in prog.events}), partial=bool(prog.unexpressible),
               events_empty=not prog.events, unexpressible=list(prog.unexpressible),
               leak_free=not any(w in x.lower() and w not in utt.lower()  # copying the question's own words is fine
                                 for x in texts for w in LEAK_WORDS + tuple(REGISTRY)))
    return obs
