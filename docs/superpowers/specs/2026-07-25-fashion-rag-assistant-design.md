# Fashion RAG Assistant — Design Specification

**Date:** 2026-07-25  
**Status:** Approved for implementation  
**Repository target:** `eeeewyz/Project` (planned rename: `fashion-rag-assistant`)

## 1. Objective

Transform the completed Coursera RAG notebook into an independently implemented, interview-ready portfolio project with:

- a public GitHub repository;
- a live Gradio demo hosted on Hugging Face Spaces;
- modular Python code separated from the exploratory notebook;
- measurable retrieval and answer-quality improvements;
- transparent retrieval details suitable for an Applied AI / LLM Engineer interview.

The public project will acknowledge that its concepts were inspired by a RAG course. It will not publish Coursera-provided tests, proxy code, server helpers, or the raw assignment notebook.

## 2. Audience and Language

The primary audience is Applied AI, LLM, and ML engineering interviewers in Japan.

- Main README: English.
- Opening project summary: short Japanese introduction.
- Source code, comments, tests, and UI labels: English.
- Private learning notes and course instructions: removed from the public notebook.

## 3. Scope

### In scope

- FAQ and product-query routing.
- Structured metadata extraction for product queries.
- FAQ semantic retrieval.
- Product metadata filtering plus vector similarity ranking.
- Controlled filter relaxation when the initial result set is too small.
- Grounded answer generation with product IDs and retrieved-source details.
- Short conversational context for follow-up questions.
- Gradio web interface.
- Offline unit tests, a small labeled evaluation set, and GitHub Actions.
- Hugging Face Spaces deployment using a Together AI secret.

### Out of scope for version 1

- User accounts, payments, carts, and production authentication.
- Distributed vector databases or paid hosted databases.
- Full e-commerce catalog synchronization.
- Fine-tuning.
- Agent tool execution.
- Production monitoring infrastructure.
- Publishing Coursera assets or data with unclear redistribution rights.

## 4. Technology Choices

- **UI and deployment:** Gradio on Hugging Face Spaces.
- **LLM provider:** Together AI.
- **LLM:** `Qwen/Qwen3.5-9B`.
- **Embeddings:** a compact Sentence Transformers model suitable for CPU inference.
- **Vector retrieval:** FAISS with a reproducible index-building script.
- **Structured outputs:** Pydantic models and validation.
- **Tests:** pytest.
- **CI:** GitHub Actions.
- **Configuration:** environment variables, with `TOGETHER_API_KEY` stored only as a Hugging Face Space secret.

## 5. Repository Structure

```text
fashion-rag-assistant/
├── app.py
├── src/
│   └── fashion_rag/
│       ├── __init__.py
│       ├── config.py
│       ├── schemas.py
│       ├── llm.py
│       ├── router.py
│       ├── query_parser.py
│       ├── retriever.py
│       ├── prompts.py
│       └── pipeline.py
├── data/
│   ├── sample_products.csv
│   └── faq.json
├── notebooks/
│   └── rag_pipeline_exploration.ipynb
├── evaluation/
│   ├── eval_dataset.json
│   └── run_evaluation.py
├── tests/
│   ├── test_router.py
│   ├── test_filters.py
│   └── test_retriever.py
├── scripts/
│   └── build_index.py
├── assets/
│   └── demo_screenshot.png
├── .github/workflows/tests.yml
├── .env.example
├── .gitignore
├── requirements.txt
├── LICENSE
└── README.md
```

The raw Coursera notebook and its support files are development references only and will not be committed.

## 6. System Architecture

```text
User query + recent conversation
        |
        v
FAQ / Product router
        |
        +--> FAQ semantic search --------+
        |                                |
        +--> Product metadata parser     |
                    |                    |
                    v                    |
          metadata filter + FAISS        |
                    |                    |
                    v                    |
          controlled filter fallback    |
                    |                    |
                    +--------------------+
                              |
                              v
                    grounded generation
                              |
                              v
             answer + sources + retrieval details
```

### 6.1 Router

The router returns one validated label: `faq`, `product`, or `unsupported`.

- It receives the current query and a bounded recent conversation history.
- Invalid model output is rejected rather than silently accepted.
- An unsupported query receives a concise scope message and does not trigger ungrounded product generation.

### 6.2 FAQ path

- Embed FAQ questions.
- Retrieve the top three most similar entries.
- Generate an answer using only retrieved FAQ entries.
- Display retrieved FAQ questions and similarity scores in the details panel.

This replaces the course notebook's approach of placing the complete FAQ collection into every prompt.

### 6.3 Product path

The parser produces a validated structure containing:

- task nature: `technical` or `creative`;
- requested product count;
- gender;
- master category;
- article type;
- color;
- minimum and maximum price;
- usage;
- season.

Unspecified categorical fields are represented as null or empty values, not the literal filter value `Any`.

Retrieval proceeds as follows:

1. Apply valid hard metadata constraints.
2. Rank eligible products by vector similarity.
3. Return the configured top-k products.
4. If too few products remain, relax soft filters in this order:
   `usage → season → color → gender`.
5. Do not relax price constraints or the core article type.
6. If no valid products remain, return an explicit no-match result.

### 6.4 Generation

The generation prompt receives only:

- the resolved query;
- bounded recent conversation context;
- retrieved FAQ or product records;
- the applied filters;
- response-format instructions.

Product answers must include product IDs. The system verifies that referenced IDs exist in the retrieved set before presenting retrieval details.

## 7. Conversation Behavior

The application keeps a bounded number of recent turns. Recent context is supplied to routing and metadata analysis so follow-ups such as “What about the blue ones?” can be resolved without an additional query-rewrite call.

Conversation state is session-local and is not persisted.

## 8. Error Handling and Security

- Missing `TOGETHER_API_KEY`: show a configuration message without exposing secret values.
- LLM timeout or transient provider failure: retry once, then return a user-facing error.
- Invalid structured output: validate with Pydantic and retry once with a correction instruction.
- Empty retrieval result: relax only approved soft filters; otherwise report no match.
- Index or data loading failure: fail startup with a concise actionable error.
- No bare `except:` clauses.
- No disabled TLS verification.
- No Coursera proxy endpoints or absolute Coursera filesystem paths.
- Secrets, local indexes, caches, and `.env` are excluded through `.gitignore`.
- Logs must not include API keys, authorization headers, or complete user histories.

## 9. Data and Provenance

The supplied product schema appears related to the Kaggle Fashion Product Images dataset, but redistribution permission is not sufficiently clear for this public portfolio repository.

Version 1 therefore uses:

- a small synthetic product catalog generated specifically for the demo;
- a small independently written FAQ set;
- source and generation notes in the README.

The synthetic catalog retains the useful schema—product ID, name, category, type, color, season, usage, gender, and price—without copying brand descriptions or Coursera files.

## 10. Evaluation

The labeled evaluation set contains approximately 30–50 queries covering both normal and edge cases.

| Component | Metric | Purpose |
|---|---|---|
| Router | Accuracy | FAQ/Product classification |
| Metadata parser | Field accuracy | Correct extraction of constraints |
| Retriever | Hit@5 | Relevant products in top five |
| Filters | Constraint compliance | Returned items satisfy hard constraints |
| Generation | Groundedness | Referenced items come from retrieval |
| System | End-to-end latency | Practical demo responsiveness |

The report compares:

- **Baseline:** vector search only.
- **Improved:** metadata filtering + vector search + controlled fallback.

No evaluation result will be placed in the README until it has been measured. GitHub Actions runs deterministic tests that do not require an API key. LLM-dependent evaluation is launched manually.

## 11. Gradio Interface

The page contains:

- title and one-sentence description;
- three example-query buttons;
- chat input and conversation window;
- an expandable `Retrieval Details` area;
- clear configuration, timeout, and no-result messages.

Retrieval details include:

- selected route;
- technical or creative task nature;
- applied filters;
- top-k retrieved items;
- similarity scores;
- product IDs.

Example prompts:

1. “What is your return policy?”
2. “Show me three blue T-shirts under $100.”
3. “Create a summer wedding look for a man.”

## 12. README Narrative

The README is answer-first and follows this order:

1. Japanese summary.
2. English overview.
3. Live demo link and screenshot.
4. Problem statement.
5. Architecture.
6. Key features.
7. Baseline and improvements.
8. Measured evaluation results.
9. Local setup.
10. Hugging Face deployment.
11. Limitations and future work.
12. Data, model, and course-inspiration attribution.

Required provenance statement:

> The project was inspired by concepts learned in a RAG course, while the application architecture, evaluation pipeline, deployment, and implementation were independently redesigned.

## 13. Acceptance Criteria

The design is implemented when all of the following are true:

- A new user can run the app from documented commands.
- The public repository contains no Coursera test, proxy, server-helper, or raw-assignment files.
- The app answers FAQ and product questions through distinct retrieval paths.
- Product answers use only retrieved product IDs.
- Hard price and article-type constraints are never relaxed.
- Retrieval details are visible in Gradio.
- Deterministic unit tests pass locally and in GitHub Actions.
- Baseline and improved evaluation results are generated from the committed labeled set.
- The Hugging Face Space starts successfully using a secret API key.
- README links to the live demo and reports only verified results.
