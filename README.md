# P2I Data Construction Pipeline

Anonymous repository for NeurIPS 2026 review.

This repository provides the data construction pipeline for **Planning2Interaction (P2I)**, a recovery-oriented dataset and benchmark for stateful multi-turn function calling.

P2I constructs multi-turn function-calling trajectories from structured plans. It supports three trajectory types:

- **All-correct trajectories**: successful multi-turn tool-use interactions.
- **ERS trajectories**: same-turn recovery from explicit tool feedback.
- **MTR trajectories**: delayed cross-turn realignment after omitted prerequisite actions.
