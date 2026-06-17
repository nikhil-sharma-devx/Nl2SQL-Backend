"""RAG (Retrieval-Augmented Generation) package.

Split into two sub-packages with separate lifecycles:
  - ingestion/  — Offline batch pipeline (runs when schema changes).
  - retrieval/  — Online per-query pipeline (runs on every user message).
"""
