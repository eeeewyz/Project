"""Data loading and presentation helpers for the Gradio application."""

from __future__ import annotations

import csv
import inspect
import json
from pathlib import Path
from typing import TYPE_CHECKING

from fashion_rag.config import MissingAPIKeyError, Settings
from fashion_rag.llm import LLMServiceError
from fashion_rag.pipeline import ApplicationError, GroundingError
from fashion_rag.schemas import ChatTurn, FAQEntry, PipelineResponse, Product

if TYPE_CHECKING:
    import gradio as gr

    from fashion_rag.pipeline import FashionRAGPipeline


PROJECT_ROOT = Path(__file__).resolve().parents[2]
PRODUCTS_PATH = PROJECT_ROOT / "data" / "sample_products.csv"
FAQS_PATH = PROJECT_ROOT / "data" / "faq.json"
TITLE = "Fashion RAG Assistant"
SUBTITLE = (
    "An explainable RAG demo for store FAQs, catalog search, "
    "and outfit recommendations."
)
EXAMPLE_PROMPTS = (
    "What is your return policy?",
    "Show me three blue T-shirts under $100.",
    "Create a summer wedding look for a man.",
)


def load_products(path: str | Path) -> list[Product]:
    """Load and strictly validate the public product catalogue."""

    with Path(path).open(encoding="utf-8", newline="") as handle:
        rows = []
        for row in csv.DictReader(handle):
            values: dict[str, object] = dict(row)
            values["price"] = float(row["price"])
            rows.append(Product.model_validate(values))
    return rows


def load_faqs(path: str | Path) -> list[FAQEntry]:
    """Load and strictly validate the public FAQ collection."""

    records = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(records, list):
        raise ValueError("FAQ data must be a JSON array")
    return [FAQEntry.model_validate(record) for record in records]


def _table_text(value: str) -> str:
    return value.replace("|", r"\|").replace("\r", " ").replace("\n", " ")


def format_retrieval_details(response: PipelineResponse) -> str:
    """Render a stable, interview-friendly view of retrieval evidence."""

    nature = response.nature.value if response.nature is not None else "—"
    if response.applied_filters is None:
        filters = "—"
    else:
        filters = (
            "`"
            + json.dumps(
                response.applied_filters.model_dump(exclude_none=True),
                ensure_ascii=False,
                separators=(",", ":"),
            )
            + "`"
        )

    lines = [
        f"**Route:** {response.route.value}  ",
        f"**Nature:** {nature}  ",
        f"**Latency:** {response.latency_ms:.0f} ms  ",
        f"**Applied filters:** {filters}",
        "",
        "| ID | Evidence | Score | Relaxed filters |",
        "|---|---|---:|---|",
    ]
    if response.hits:
        for hit in response.hits:
            relaxed = ", ".join(hit.relaxed_filters) or "—"
            lines.append(
                f"| {_table_text(hit.record_id)} "
                f"| {_table_text(hit.text)} "
                f"| {hit.score:.3f} "
                f"| {_table_text(relaxed)} |"
            )
    else:
        lines.append("| — | — | — | — |")
    return "\n".join(lines)


def create_pipeline(settings: Settings) -> FashionRAGPipeline:
    """Construct every application service exactly once."""

    # Fail before model initialization so importing the Space entry point is
    # quick and deterministic when its secret has not yet been configured.
    settings.require_api_key()

    from fashion_rag.llm import StructuredGenerator, TogetherLLM
    from fashion_rag.pipeline import FashionRAGPipeline
    from fashion_rag.query_parser import ProductQueryParser
    from fashion_rag.retriever import (
        FAQRetriever,
        ProductRetriever,
        SentenceTransformerEmbedder,
    )
    from fashion_rag.router import QueryRouter

    products = load_products(PRODUCTS_PATH)
    faqs = load_faqs(FAQS_PATH)
    embedder = SentenceTransformerEmbedder(settings.embedding_model)
    llm = TogetherLLM(settings)
    structured = StructuredGenerator(llm)
    return FashionRAGPipeline(
        router=QueryRouter(
            structured,
            max_history_turns=settings.max_history_turns,
        ),
        product_parser=ProductQueryParser(
            structured,
            max_history_turns=settings.max_history_turns,
        ),
        product_retriever=ProductRetriever(products, embedder),
        faq_retriever=FAQRetriever(faqs, embedder),
        llm=llm,
        top_k=settings.top_k,
    )


def _safe_request_error(error: Exception) -> str:
    if isinstance(error, MissingAPIKeyError):
        return (
            "This demo is not configured yet. "
            "Please ask the project owner to add the API secret."
        )
    if isinstance(error, LLMServiceError):
        return (
            "The language-model service is temporarily unavailable. "
            "Please try again."
        )
    if isinstance(error, GroundingError):
        return "I could not verify a grounded answer. Please try again."
    return "The assistant could not process this request. Please try again."


def _startup_message(error: Exception) -> str:
    if isinstance(error, MissingAPIKeyError):
        return (
            "The demo is not configured yet. "
            "The project owner needs to add the API secret."
        )
    if isinstance(error, LLMServiceError):
        return (
            "The demo could not connect to its language-model service. "
            "Please try again later."
        )
    return "The demo could not start. Please try again later."


def build_demo(
    pipeline_or_error: FashionRAGPipeline | Exception,
) -> gr.Blocks:
    """Build the chat UI or a sanitized startup-error page."""

    import gradio as gr

    with gr.Blocks(title=TITLE) as demo:
        gr.Markdown(f"# {TITLE}")
        gr.Markdown(SUBTITLE)

        if isinstance(pipeline_or_error, Exception):
            gr.Markdown(f"**Startup status:** {_startup_message(pipeline_or_error)}")
            return demo

        pipeline = pipeline_or_error
        chatbot_parameters = inspect.signature(gr.Chatbot).parameters
        chatbot_options: dict[str, str] = {}
        # Gradio 6 uses message dictionaries exclusively and removed this
        # switch; retain it for compatibility with earlier supported releases.
        if "type" in chatbot_parameters:
            chatbot_options["type"] = "messages"

        chatbot = gr.Chatbot(
            label="Conversation",
            height=420,
            **chatbot_options,
        )
        message_box = gr.Textbox(
            label="Your question",
            placeholder="Ask about a store policy or fashion product…",
        )
        history_state = gr.State([])
        with gr.Accordion("Retrieval Details", open=False):
            details = gr.Markdown(
                "Submit a question to inspect the route and retrieved evidence."
            )

        gr.Markdown("### Try an example")
        with gr.Row():
            example_buttons = [
                gr.Button(prompt, variant="secondary")
                for prompt in EXAMPLE_PROMPTS
            ]

        def respond(
            message: str,
            messages: list[dict[str, str]],
            history_state: list[dict[str, str]],
        ) -> tuple[
            str,
            list[dict[str, str]],
            list[dict[str, str]],
            str,
        ]:
            display_messages = list(messages or [])
            stored_messages = list(history_state or [])
            user_message = {"role": "user", "content": message}
            try:
                bounded_history = [
                    ChatTurn.model_validate(turn)
                    for turn in stored_messages[-4:]
                ]
                response = pipeline.answer(message, bounded_history)
                assistant_text = response.answer
                retrieval_details = format_retrieval_details(response)
            except MissingAPIKeyError as error:
                assistant_text = _safe_request_error(error)
                retrieval_details = (
                    "**Retrieval details:** unavailable for this request."
                )
            except LLMServiceError as error:
                assistant_text = _safe_request_error(error)
                retrieval_details = (
                    "**Retrieval details:** unavailable for this request."
                )
            except GroundingError as error:
                assistant_text = _safe_request_error(error)
                retrieval_details = (
                    "**Retrieval details:** unavailable for this request."
                )
            except ApplicationError as error:
                assistant_text = _safe_request_error(error)
                retrieval_details = (
                    "**Retrieval details:** unavailable for this request."
                )

            assistant_message = {
                "role": "assistant",
                "content": assistant_text,
            }
            display_messages.extend((user_message, assistant_message))
            stored_messages.extend((user_message, assistant_message))
            return (
                "",
                display_messages,
                stored_messages,
                retrieval_details,
            )

        message_box.submit(
            respond,
            inputs=[message_box, chatbot, history_state],
            outputs=[message_box, chatbot, history_state, details],
            api_name="respond",
        )

        def example_callback(prompt: str):
            def run_example(
                messages: list[dict[str, str]],
                stored: list[dict[str, str]],
            ) -> tuple[
                str,
                list[dict[str, str]],
                list[dict[str, str]],
                str,
            ]:
                return respond(prompt, messages, stored)

            return run_example

        for button, prompt in zip(example_buttons, EXAMPLE_PROMPTS, strict=True):
            button.click(
                example_callback(prompt),
                inputs=[chatbot, history_state],
                outputs=[message_box, chatbot, history_state, details],
                api_visibility="private",
            )

    return demo


def build_application(settings: Settings) -> gr.Blocks:
    """Build the deployable app while sanitizing all startup failures."""

    try:
        pipeline: FashionRAGPipeline | Exception = create_pipeline(settings)
    except Exception as error:
        pipeline = error
    return build_demo(pipeline)
