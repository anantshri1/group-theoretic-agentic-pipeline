# Group-Theoretic Agentic Pipeline

A from-scratch end-to-end multi-agent LLM pipeline that solves and symbolically verifies `SU(N)` Lie algebra identities, built with `LangGraph` and `MCP` - evaluated independently by a domain expert. The project was structured as follows:
- Stage 0: SymPy foundations — hand-rolled `su(N)` generators, structure constants, exhaustive verification
- Stage 1: Solver-only baseline — unverified LLM attempts, failure mode documentation
- Stage 2: Single-pass solver + symbolic verifier + LLM judge
- Stage 3: Cyclic critic-retry loop with `LangGraph` conditional edges
- Stage 4: MCP-ify the verifier as a standalone server (`stdio` transport)
- Stage 5: `SSE` transport + `Gradio` + HF Spaces deployment


**What it does**

You submit a group theory problem — a commutator identity or structure constant claim in `su(2)`, `su(3)`, or `su(4)`. A solver agent (`Qwen2.5-72B`) attempts a solution and emits a structured `CLAIM` line. An independent symbolic verifier checks the claim using `SymPy` (exact matrix arithmetic, no LLM involved). If the claim fails verification or the solver's prose conclusion contradicts the symbolic result, a critic agent generates structured feedback and the solver retries — up to 3 times.

**Pipeline**
```
User problem
       │
       ▼
┌─────────────┐
│  Solver     │  Qwen2.5-72B via HF Inference Router
│  Agent      │  Emits structured CLAIM line
└──────┬──────┘
       │
       ▼
┌─────────────┐        ┌─────────────────────────┐
│  Verifier   │◄──────►│  MCP Server (SSE)       │
│  Node       │        │  SymPy: su_n_generators │
└──────┬──────┘        │  commutator, f_abc      │
       │               └─────────────────────────┘
       │ fail
       ▼
┌─────────────┐
│  Critic     │  Qwen2.5-72B, structured JSON feedback
│  Agent      │
└──────┬──────┘
       │ retry
       └──────► Solver (up to 3 retries)
```

The verifier runs as a standalone `MCP` server inside the same container, communicating with the `LangGraph` graph over `SSE` transport on `localhost:8000`. The agent side has no direct SymPy dependency — it calls a tool over the wire and receives a JSON result.


**Conventions**

Generators follow the physics convention $T_a = \frac{\lambda_a}{2}$, where $\lambda_a$ are the generalized Gell-Mann matrices. Structure constants satisfy $[T_a, T_b] = if_{abc}T_c$, with $Tr(T_a T_b) = \frac{1}{2} \delta_{ab}$.

------

## Scoping `SymPy` Capabilities

`SymPy` is a Python library for symbolic mathematics; as such, Stage 0 was done to assess the capabilities of the package. Some examples of the assessments include:
* Can we instantiate `su(2)` and `su(3)` via `sympy.liealgebras` (it's usually accessed via the Cartan matrix / algebra type strings like `"A1"`, `"A2"` — `su(N`) corresponds to Cartan type $A_{N-1}$)
* Can we get structure constants out of it (this is what the `verifier` needs for commutator checks)
* Can we get the Casimir eigenvalue computation for a given representation (this is the other thing we need)

> `"A1"` is a rank-1 edge. This is a genuine bug/gap in `SymPy`'s `TypeA.cartan_matrix()` for the rank-1 case. `su(2)` support is broken in this `SymPy` version. Therefore, `su(2)` needs a manual fallback (we hand-code its Cartan matrix / structure constants ourselves, since they're trivial — `su(2)` is just the Pauli matrix algebra) rather than relying on `sympy.liealgebras` for it.

Due to the limited capabilities of `SymPy`, we tried to incorporate `liesym` in early stages of this project:
* `liesym` exists specifically to extend/replace `SymPy`'s `liealgebras` because of the gaps we just hit ourselves — `SymPy`'s `liealgebras` module made tradeoffs that locked the basis for classic Lie algebras in favor of speed, which would require anyone using a different basis to hand-calculate representations themselves. That's our exact problem, described by someone who clearly hit the same wall.
* It's not abandoned-on-PyPI-only — there's real documentation at a dedicated docs site, and it's explicitly described as an extension module on SymPy that reimplements the liealgebra module using a compiled backend for speedups. [[GitHub Documentation]](https://github.com/npapapietro/liesym)
> It's at version 0.8.1 per the docs site (vs. 0.7.0 listed on PyPI for some wheels), so there's been at least one release cycle of iteration. 
* The actual usage examples from the docs show exactly the primitives we need and more: `from liesym import SO, SU` then `so10.product(...) `returns tensor product decomposition results as a list of representation matrices — that's tensor product decomposition working out of the box.
* We also see fundamental weight and dimension-name computations for `A3` and simple roots, positive roots, and Cartan matrix access for `A3`, with `A1` `(su(2))` explicitly demonstrated working via `ls.A(1)` and .`simple_roots()` — which directly answers our `su(2)-was-broken-in-sympy` concern.

> Unfortunately, there was a staleness issue with `liesym` and we were forced to continue with `SymPy`.

Remaining within the scope of `SymPy` capabilities, we identified *verification of commutation relations*, and  *calculation of structure constants* as realistic goals of this project. As such, representation theoretic aspects of Lie algebra, and extensions to B, C, D groups are deferred to a future extension of this project.

The remainder of this stage focused on constructing generators of `su(2)` and `su(3)`, and verifying their commutation relations. Based on these results, we iteratively construct generators of `su(N)`, affording the agentic pipeline full generality for arbitrary `N`.

----

## Solver-only Baseline

The scope for this section was as follows:

1. **A `SolverState TypedDict`** — the state schema `LangGraph` will pass around. Minimal for now: something like `{"problem": str, "solution": str}` — input problem statement in, free-text answer out.
2. **A single-node `LangGraph` graph** — one node (`solver_node`) wrapping a call to `DeepSeek-V3.2` via the HF router, pinned to `featherless-ai`, using `langchain_openai.ChatOpenAI` with a custom `base_url`.
3. **A system/user prompt that frames the task** — e.g., `"you are solving a group-theory identity verification problem, here's the problem statement, give your reasoning and answer."` This is the one place actual prompt design happens.
4. **A handful of test problem statements** (3-5 sample prompts spanning `su(2)`/`su(3)` commutator-type questions) to run through the graph and sanity-check that we get sensible-looking free text back.
5. **Explicitly NOT here**: no `SymPy` involvement yet (the solver is "vibing" off the LLM's own math, unverified), no `Verifier`, no loop, no correctness checking.

> Since the `Solver` doesn't call `SymPy` and isn't checked, there's a real chance `DeepSeek` just confidently produces wrong group theory (these are exactly the kind of index-gymnastics calculations LLMs flub).


**Architecture of state schema and solver at this stage**:
* `SolverState` is deliberately minimal (just 2 fields) — this is the contract Stage 2's `Verifier` will need to extend (it'll likely add a `verified: bool or verifier_notes: str` field later). Keeping it lean now means Stage 2's diff is additive, not a rewrite.
* The node function takes the whole state in and returns the whole state out — that's the `LangGraph` convention (nodes are state transformers), even though right now it feels like overkill for a single field update.


The following test problems were used to assess the performance of the LLM:
```
...
test_problems = [
    "Verify that for su(2), [T_1, T_2] = i*T_3, where T_a = sigma_a / 2 "
    "and sigma_a are the Pauli matrices.",

    "Compute the structure constant f_123 for su(3) using the convention "
    "T_a = lambda_a / 2, where lambda_a are the Gell-Mann matrices.",

    "Is the following a valid su(3) commutation relation? "
    "[T_4, T_5] = (i/2)*T_3 + (i*sqrt(3)/2)*T_8",
]
...
```
These were chosen for the following reasons:
* Problem 1 is the simplest possible `su(2)` case — straight off the Stage 0 verified facts. Good baseline: if `DeepSeek` can't get this right, that's a strong early signal about how much we should trust unverified output later.
* Problem 2 asks for a single structure constant in `su(3)` — again, something Stage 0 already exhaustively verified the true value of (you'll know if it's right or wrong immediately, no need to compute it fresh).
* Problem 3 is deliberately a true relation (this one's actually a correct `su(3)` commutator) phrased as a yes/no verification question — testing whether `DeepSeek` can confirm a correct claim, not just generate one from scratch.

The results of the diagnostic were the following:
* **Problem 1 (`su(2)`)**: Clean, correct, no issues. The `σ₁σ₂` computation is right, the conclusion is right. Nothing to flag here — `DeepSeek` handled this one solidly.
* **Problem 2 (`f₁₂₃`)**: Final answer `(1)` is correct, but **look at element (1,1) of $T_1 T_2$​ in step 4**: it computes $0⋅0+1⋅i+0⋅0=i0\cdot0 + 1\cdot i + 0\cdot 0 = i
0⋅0+1⋅i+0⋅0=i$.
That's right by luck of clean numbers, but the method shown — "first row times first column" written as a dot product of mismatched-looking vectors — is sloppy notation that happened not to produce an error here. Not a correctness problem in this instance, but a flag that its arithmetic narration isn't rigorous; on a messier matrix it could easily drop a term.
* **Problem 3**: The boxed answer is correct, but look at the path it took to get there. The model said *"Hmm, that's suspicious"* mid-derivation, second-guessed itself, redid the same calculation a different way, and arrived at the same answer.
That self-doubt-then-recovery pattern is the real signal worth flagging: it got lucky that both paths landed on the right answer. This is exactly the kind of *"looks rigorous, actually wobbling"* behavior an automated Verifier (Stage 2) needs to catch, but it's there in the reasoning trace.

Subsequently, we gave it two additional harder questions:
* A higher-index `su(3)` constant the model is less likely to have "seen" cleanly — e.g. $f_{246}$
f246​ (it's in the standard table at $\frac{1}{2}$​, but less commonly quoted than $f_{123}$ or $f_{458}$, so there's less "memorized answer" risk and more "actually has to do index gymnastics" risk).
* A deliberately `FALSE` claim — testing whether the model can say `"no, that's wrong"` rather than just pattern-matching toward agreement. This is important: an LLM that says `"yes, valid"` to everything you ask is not actually verifying, it's just being agreeable. Stage 1 should expose this risk now, even if Stage 1 isn't required to detect it.

```
...
more_problems = [
    "Compute the structure constant f_246 for su(3) using the convention "
    "T_a = lambda_a / 2, where lambda_a are the Gell-Mann matrices.",

    "Is the following a valid su(3) commutation relation? "
    "[T_1, T_2] = i*T_3 + i*T_8",  # deliberately false: extra spurious T_8 term
]
...
```
* **Problem 4 ($f_{246}$)**: Clean and correct throughout. The matrix arithmetic is careful, properly checked step by step, and it landed on $\frac{1}{2}$, matching the verified table. No wobble this time — this is the model performing well on a genuine (if textbook-standard) computation.
* **Problem 5 (the trap question)** — this is the one to really look at. The model correctly concluded "No" in the end, and explicitly stated the correct relation ($[T_1,T_2]=iT_3$​). So at the surface level: the trap was caught. That's a meaningful positive signal. It correctly identified in Step 3 that $f_{128}=0$ in the standard basis — that's the whole answer, right there, immediately. Then it didn't stop. It spent Steps 4 through 8 constructing an elaborate argument about whether some rotated basis of `su(3)` could make the relation true, including inventing a basis transformation, checking it, finding it almost works (gets $\frac{1}{\sqrt2}$ coefficients instead of $1$), then reasoning about rescaling generators breaking orthonormality, and only then circling back to "but in the standard basis, no." It even hedges the final framing: `"if the question means 'does there exist a basis...' then yes... But... given typical textbook contexts, they expect..."` — it's answering based on guessing what the expected answer is, not because it converged on a unique mathematical truth.
This is a more sophisticated and more concerning failure mode than Problem 3's `"hmm, that's suspicious"` wobble. There, the model second-guessed a correct calculation and recovered. Here, the model manufactured genuine ambiguity where none should exist — "is $[T_1,T_2] = iT_3$​" is a question with one correct answer, given a fixed convention (which the problem statement explicitly fixed via $T_a=\frac{\lambda_a}{2}$). The detour into `"well, in some other basis..."` is not wrong physics, but it's the model hedging toward agreeableness rather than just stating the clean falsehood-detection it had already correctly derived in step 3. It talked itself almost back into validating the false claim before recovering.

> **Why this matters architecturally, not just as a curiosity**: this is exactly the failure mode that makes an LLM a bad verifier of its own work, and exactly why independent SymPy verification, not `"ask the LLM to double check itself"` is the right call. An LLM solver that can be talked into self-doubt on a clean `true/false` question, even when it already has the right answer in hand, is not a reliable arbiter — it's a reasoning generator that needs an external, deterministic check.
For my records — **solver baseline correctly identifies false claims but exhibits agreeableness-hedging: it can construct elaborate (correct, but irrelevant) justifications for why a false claim "might" hold in some alternate framing, before correctly rejecting it. This motivates an independent symbolic verifier over self-verification.**

-----

## Single-Pass Solver & Verifier

