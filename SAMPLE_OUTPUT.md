# Sample Output

A real end-to-end run of the assistant against the bundled sample corpus
(`data/documents/` — 9 Northwind Dynamics policy files, `.md` + `.txt`).

- **Command:** `python ingest.py` then `python eval.py`
- **Models:** `gpt-4o-mini` (LLM + LLM-reranker), `text-embedding-3-small`
- **Index:** FAISS `IndexFlatIP`, 9 sources → 19 chunks
- **Config:** vector top-k 8 + BM25 top-k 8 → RRF fuse 10 → rerank → final 4; grounding threshold 0.20

The six questions were chosen to exercise every functional requirement:

| Q | Demonstrates |
|---|---|
| 1 | Basic retrieval + multi-document source citation (`[1][2]`) |
| 2 | **Conversational memory** — bare follow-up rewritten to a standalone query |
| 3 | Conversational memory again (`"What about the rules for sick leave?"`) |
| 4 | Precise fact retrieval + single citation |
| 5 | Precise fact retrieval + single citation |
| 6 | **Hallucination handling** — topic absent from the corpus → fixed "not found" reply, no sources |

---

```text
========================================================================================
Q: What is the annual leave entitlement?

A: - Full-time employees receive **20 working days** of annual leave per calendar year.
- Employees with **5 or more years** of continuous service receive **25 working days** per year [1][2].

   sources        : Leave_Policy.md, Company_FAQ.txt
   grounded       : True  best_sim=0.555
   retrieved      : Leave_Policy.md[10.0], Company_FAQ.txt[10.0], Leave_Policy.md[0.0], Benefits_Guide.md[0.0]
   latency        : 9.47s

========================================================================================
Q: How does carry-forward work and by when must it be used?
   (rewritten -> How does the carry-forward of annual leave work, and by when must it be used?)

A: - Up to **5 unused annual leave days** may be carried forward into the next calendar year.
- Carried-forward days **must be used by 31 March**; any balance above 5 days, or not used by 31 March, is forfeited without payment.
- Carry-forward of more than 5 days requires **written Director approval** and is granted only where business needs prevented leave from being taken [1].

   sources        : Leave_Policy.md
   grounded       : True  best_sim=0.552
   retrieved      : Leave_Policy.md[10.0], Company_FAQ.txt[10.0], Company_FAQ.txt[0.0], Leave_Policy.md[0.0]
   latency        : 8.74s

========================================================================================
Q: What about the rules for sick leave?
   (rewritten -> What are the rules for sick leave?)

A: - Employees are entitled to **10 paid sick days** per calendar year.
- A **medical certificate** is required for any absence of **3 or more consecutive days**.
- Unused sick leave does **not** carry forward and is **not** paid out [1].

   sources        : Leave_Policy.md
   grounded       : True  best_sim=0.676
   retrieved      : Leave_Policy.md[10.0], Company_FAQ.txt[0.0], Benefits_Guide.md[0.0], Company_FAQ.txt[0.0]
   latency        : 5.09s

========================================================================================
Q: What is the per diem for international business travel?

A: - The per diem for international business travel is **USD 100 per day** [1].

   sources        : Travel_Policy.md
   grounded       : True  best_sim=0.587
   retrieved      : Travel_Policy.md[10.0], Travel_Policy.md[0.0], Company_FAQ.txt[0.0], Remote_Work_Policy.md[0.0]
   latency        : 7.67s

========================================================================================
Q: How much parental leave does a primary caregiver get?

A: - A primary caregiver is entitled to **16 weeks** of fully paid parental leave [1].

   sources        : Benefits_Guide.md
   grounded       : True  best_sim=0.570
   retrieved      : Benefits_Guide.md[10.0], Company_FAQ.txt[0.0], Leave_Policy.md[0.0], Leave_Policy.md[0.0]
   latency        : 4.87s

========================================================================================
Q: What is the company's policy on cryptocurrency trading bonuses?

A: I could not find this information in the available documents.

   sources        : (none)
   grounded       : False  best_sim=0.352
   retrieved      : Remote_Work_Policy.md[0.0], Code_of_Conduct.md[0.0], Benefits_Guide.md[0.0], Company_FAQ.txt[0.0]
   latency        : 5.90s
```

---

## Notes on the run

- **Every fact is correct** against the source files (20/25 leave days, 5-day carry-forward by 31 March, 10 sick days, USD 100 international per diem, 16 weeks primary parental leave).
- **Memory:** questions 2 and 3 were phrased as follow-ups; the `rewritten ->` line shows the standalone query that was actually sent to retrieval.
- **Citations:** `[n]` markers in each answer are resolved back to the file names shown on the `sources` line. Question 1 legitimately drew on two documents.
- **Hallucination handling, both layers:** for question 6 the best similarity (0.352) was above the 0.20 gate, so the request reached the LLM — and the LLM still returned the exact "not found" sentence because the retrieved context did not contain the answer. The similarity gate catches the clearly-irrelevant cases earlier (see `tests/run_offline_tests.py::test_pipeline_not_found_for_unrelated_question`).
- `retrieved` shows the final 4 chunks with their LLM-reranker score in brackets; `gpt-4o-mini` tends to score decisively (10 vs 0).

## Offline test suite

`python tests/run_offline_tests.py` — 18/18 passing, no network calls
(loaders, chunking, FAISS + NumPy vector stores, BM25, RRF fusion, hybrid
retriever, both rerankers, memory windowing + condensation, SQLite history +
owner isolation + schema migration, corpus add/remove, and the full pipeline
including the grounding gate and citation resolution).
