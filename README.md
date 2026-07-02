# Group-Theoretic Agentic Pipeline

A multi-agent LLM system that solves and symbolically verifies `SU(N)` Lie algebra identities, built with `LangGraph` and `MCP`, deployed via Hugging Face Spaces.

## What it does

You submit a group theory problem — a commutator identity or structure constant claim in su(2), su(3), or su(4). A solver agent (Qwen2.5-72B) attempts a solution and emits a structured CLAIM line. An independent symbolic verifier checks the claim using SymPy (exact matrix arithmetic, no LLM involved). If the claim fails verification or the solver's prose conclusion contradicts the symbolic result, a critic agent generates structured feedback and the solver retries — up to 3 times.

## Architecture
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

The verifier runs as a standalone MCP server inside the same container, communicating with the LangGraph graph over SSE transport on localhost:8000. The agent side has no direct SymPy dependency — it calls a tool over the wire and receives a JSON result.

## Five-stage roadmap

- Stage 0: SymPy foundations — hand-rolled su(N) generators, structure constants, exhaustive verification
- Stage 1: Solver-only baseline — unverified LLM attempts, failure mode documentation
- Stage 2: Single-pass solver + symbolic verifier + LLM judge
- Stage 3: Cyclic critic-retry loop with LangGraph conditional edges
- Stage 4: MCP-ify the verifier as a standalone server (stdio transport)
- Stage 5: SSE transport + Gradio + HF Spaces deployment (this)

## Key technical decisions

**Why SymPy over LLM self-verification**: Stage 1 documented three solver failure modes — reasoning wobble, agreeableness-hedging on false claims, and format compliance without content. An LLM that can be talked into self-doubt on a clean true/false question is not a reliable verifier of its own work.

**Why MCP**: the verifier is a separately deployable artifact with a defined tool interface. The agent graph has no import-time dependency on SymPy — it calls verify_lie_algebra_identity(claim_text) over a wire and receives a structured result. This is the architecture that would scale to a remote verifier service.

**Why SSE over stdio**: stdio transport requires subprocess spawning, which is incompatible with single-container deployment. SSE runs the server as a background thread on localhost, preserving the over-the-wire MCP story without needing a second container.

## How to use

Enter a problem in one of these forms:
- "Verify that for su(2), [T_1, T_2] = i*T_3"
- "Compute the structure constant f_123 for su(3)"
- "Is the following a valid su(3) commutation relation? [T_4, T_5] = (i/2)*T_3 + (i*sqrt(3)/2)*T_8"

Use the example buttons to load pre-tested problems. The right panel shows the final verdict, verifier detail, and the full reasoning trace with per-attempt critic feedback if retries occurred.

## Conventions

Generators follow the physics convention T_a = lambda_a / 2, where lambda_a are the generalized Gell-Mann matrices. Structure constants satisfy [T_a, T_b] = i * f_abc * T_c with Tr(T_a T_b) = (1/2) delta_ab.
