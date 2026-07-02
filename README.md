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
│  Verifier   │◄──────►│  MCP Server (SSE)        │
│  Node       │        │  SymPy: su_n_generators  │
└──────┬──────┘        │  commutator, f_abc       │
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

The remainder of this stage focused on constructing generators of `su(2)` and `su(3)`, and verifying their commutation relations.



