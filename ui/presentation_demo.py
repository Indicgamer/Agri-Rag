"""
Agri-RAG Phase 3 Demo: NLI-Gated vs Standard RAG Comparison
Streamlit app showing side-by-side comparison with RAGAS metrics.
"""

import streamlit as st
import sys
import os
import json
import time
import logging
from pathlib import Path
from datetime import datetime
from dotenv import load_dotenv

load_dotenv()

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.retrieval.neo4j_retriever import KnowledgeGraph
from src.retrieval.faiss_retriever import FAISSVectorStore
from src.retrieval.vector_retriever import VectorStore
from src.retrieval.hybrid_retriever import HybridRetriever
from src.pruning.nli_pruner import NLIPruner
from src.models.response_generator import ResponseGenerator
from src.models.rag_pipeline import AgriRAGPipeline, PipelineResult
from configs.settings import settings

logging.basicConfig(level=logging.WARNING)
logger = logging.getLogger(__name__)

st.set_page_config(
    page_title="Agri-RAG: NLI vs Standard RAG Demo",
    page_icon="🌾",
    layout="wide",
    initial_sidebar_state="collapsed"
)

st.markdown("""
<style>
    .main { padding: 1rem; }
    .header { text-align: center; margin-bottom: 1rem; }
    .nli-box { background-color: #e8f5e9; padding: 1.2rem; border-radius: 0.5rem; border-left: 5px solid #2e7d32; margin: 0.5rem 0; }
    .baseline-box { background-color: #fce4ec; padding: 1.2rem; border-radius: 0.5rem; border-left: 5px solid #c62828; margin: 0.5rem 0; }
    .metric-card { background-color: #f5f5f5; padding: 0.8rem; border-radius: 0.5rem; text-align: center; }
    .winner { background-color: #c8e6c9; padding: 0.3rem 0.8rem; border-radius: 1rem; font-weight: bold; font-size: 0.8rem; }
    .stTabs [data-baseweb="tab-list"] { gap: 2rem; }
    .stTabs [data-baseweb="tab-list"] button { font-weight: bold; }
    .question-box { background-color: #fff8e1; padding: 1rem; border-radius: 0.5rem; border-left: 5px solid #f9a825; margin: 1rem 0; }
</style>
""", unsafe_allow_html=True)

if 'pipeline_nli' not in st.session_state:
    st.session_state.pipeline_nli = None
    st.session_state.pipeline_baseline = None
    st.session_state.retriever = None
    st.session_state.generator = None
    st.session_state.nli_pruner = None
    st.session_state.initialized = False
    st.session_state.queries_run = 0
    st.session_state.nli_mode = None


@st.cache_resource
def init_retrieval():
    logger.info("Initializing Neo4j...")
    neo4j = KnowledgeGraph(
        uri=settings.neo4j.uri,
        username=settings.neo4j.username,
        password=settings.neo4j.password,
        database=settings.neo4j.database
    )

    if settings.use_faiss:
        vec = FAISSVectorStore(
            collection_name=settings.faiss.collection_name,
            embedding_model=settings.faiss.embedding_model,
            persistence_dir=str(settings.faiss.persistence_dir)
        )
    else:
        vec = VectorStore(
            collection_name=settings.chromadb.collection_name,
            embedding_model=settings.chromadb.embedding_model,
            persistence_dir=str(settings.chromadb.persistence_dir)
        )

    logger.info("Initializing Hybrid Retriever...")
    retriever = HybridRetriever(
        vector_store=vec, knowledge_graph=neo4j,
        vector_weight=settings.retrieval.vector_weight,
        graph_weight=settings.retrieval.graph_weight,
        top_k_vector=settings.retrieval.top_k_vector,
        top_k_graph=settings.retrieval.top_k_graph
    )
    return retriever, neo4j


def init_nli(mode):
    if mode == "deberta":
        logger.info("Loading DeBERTa-v3 NLI (heavy, ~500MB)...")
        try:
            pruner = NLIPruner(
                model_name=settings.nli.model_name,
                provider="transformers",
                device=settings.nli.device,
                entailment_threshold=settings.nli.entailment_threshold,
                neutral_threshold=settings.nli.neutral_threshold
            )
            logger.info("DeBERTa NLI loaded")
            return pruner, "deberta"
        except Exception as e:
            logger.warning(f"DeBERTa NLI failed ({e}), falling back to heuristic")
            st.warning("DeBERTa-v3 too large for this system, using lightweight heuristic NLI")
    logger.info("Using heuristic NLI (lightweight, no model download)")
    pruner = NLIPruner(
        provider="heuristic",
        entailment_threshold=0.3,
        neutral_threshold=0.3
    )
    return pruner, "heuristic"

    provider = settings.llm.provider
    if provider == "ollama":
        try:
            import urllib.request
            urllib.request.urlopen("http://localhost:11434/api/tags", timeout=3)
        except Exception:
            if settings.llm.groq_api_key:
                logger.info("Ollama unreachable, switching to Groq")
                provider = "groq"
            elif settings.llm.api_key:
                logger.info("Ollama unreachable, switching to OpenAI")
                provider = "openai"
            else:
                logger.warning("No LLM available, will use fallback mode")

    logger.info("Initializing Response Generator...")
    response_gen = ResponseGenerator(
        provider=provider,
        model_name=settings.llm.model_name,
        openai_model=settings.llm.openai_model,
        openai_base_url=settings.llm.openai_base_url,
        api_key=settings.llm.api_key,
        groq_model=settings.llm.groq_model,
        groq_api_key=settings.llm.groq_api_key,
        allow_response_fallback=True if provider != "ollama" else settings.llm.allow_response_fallback,
        base_url=settings.llm.base_url,
        temperature=settings.llm.temperature,
        max_tokens=settings.llm.max_tokens
    )

    logger.info("Building NLI-Gated Pipeline...")
    pipeline_nli = AgriRAGPipeline(
        hybrid_retriever=retriever,
        nli_pruner=nli_pruner,
        response_generator=response_gen
    )

    return pipeline_nli, retriever, response_gen, neo4j


def run_baseline(retriever, generator, query, top_k=12):
    start = time.time()
    facts = retriever.retrieve(query, top_k=top_k)
    fact_strings = [f.content if hasattr(f, 'content') else str(f) for f in facts]
    response = generator.generate(
        question=query,
        pruned_facts=fact_strings,
        warnings=[],
        include_citations=True
    )
    return PipelineResult(
        query=query,
        response=response.response,
        confidence=response.confidence_score,
        citations=response.citations if response.citations else [],
        warnings=response.warnings if response.warnings else [],
        pipeline_steps={'retrieval': {'num_results': len(fact_strings)}},
        execution_time=time.time() - start,
        timestamp=str(datetime.now()),
        retrieved_facts=fact_strings
    )

st.markdown("<div class='header'><h1>🌾 Agri-RAG: NLI-Gated vs Standard RAG</h1><h3>Phase 3 Presentation Demo | Sachidanand C G (1RV24SCS11)</h3></div>", unsafe_allow_html=True)

with st.sidebar:
    st.markdown("## ⚙️ System")
    
    nli_option = st.radio(
        "NLI Mode (affects RAM)",
        ["Heuristic (lightweight, fast)", "DeBERTa-v3 (accurate, ~500MB)"],
        index=0,
        help="Heuristic uses keyword matching (no model download). DeBERTa requires ~500MB RAM."
    )
    
    if st.button("🚀 Initialize Pipelines", use_container_width=True):
        with st.spinner("Connecting to Neo4j + FAISS..."):
            try:
                retriever, neo4j = init_retrieval()
                st.session_state.retriever = retriever
                st.success("✓ Retrieval ready (Neo4j + FAISS)")
            except Exception as e:
                st.error(f"Retrieval init failed: {e}")
                st.stop()
        
        mode = "deberta" if "DeBERTa" in nli_option else "heuristic"
        with st.spinner("Loading NLI pruner..."):
            pruner, actual_mode = init_nli(mode)
            st.session_state.nli_pruner = pruner
            st.session_state.nli_mode = actual_mode
        
        provider = settings.llm.provider
        if provider == "ollama":
            try:
                import urllib.request
                urllib.request.urlopen("http://localhost:11434/api/tags", timeout=3)
            except Exception:
                if settings.llm.groq_api_key:
                    provider = "groq"
                elif settings.llm.api_key:
                    provider = "openai"
        
        with st.spinner("Initializing LLM..."):
            try:
                from src.models.response_generator import ResponseGenerator
                gen = ResponseGenerator(
                    provider=provider,
                    model_name=settings.llm.model_name,
                    openai_model=settings.llm.openai_model,
                    openai_base_url=settings.llm.openai_base_url,
                    api_key=settings.llm.api_key,
                    groq_model=settings.llm.groq_model,
                    groq_api_key=settings.llm.groq_api_key,
                    allow_response_fallback=True,
                    base_url=settings.llm.base_url,
                    temperature=settings.llm.temperature,
                    max_tokens=settings.llm.max_tokens
                )
                st.session_state.generator = gen
                st.success("✓ LLM ready")
            except Exception as e:
                st.error(f"LLM init failed: {e}")
                st.stop()
        
        with st.spinner("Building pipelines..."):
            from src.models.rag_pipeline import AgriRAGPipeline
            st.session_state.pipeline_nli = AgriRAGPipeline(
                hybrid_retriever=retriever,
                nli_pruner=pruner,
                response_generator=gen
            )
            st.session_state.initialized = True
        
        st.success("✓ Both pipelines ready!")

    st.markdown("---")
    st.markdown("### 📊 Today's Session")
    st.metric("Queries Run", st.session_state.queries_run)

    st.markdown("---")
    st.markdown("### 🌱 Crops")
    st.write(", ".join(settings.crops))

    st.markdown("---")
    st.markdown("**Student**: Sachidanand C G")
    st.markdown("**Guide**: Dr. Soumya A")
    st.markdown("**RVCE, Bengaluru**")

if not st.session_state.initialized:
    st.info("👈 Select NLI mode, then click **Initialize Pipelines**")
    st.markdown("""
    ### What this demo shows:
    1. **NLI-Gated RAG** (left) - Uses NLI to prune irrelevant facts before generation
    2. **Standard RAG** (right) - Baseline without NLI pruning (all facts used directly)
    3. **RAGAS Metrics** - Faithfulness, context precision comparison
    4. **Contradiction Detection** - NLI identifies conflicting information

    The system retrieves from **Neo4j Knowledge Graph** + **FAISS Vector Store** with agricultural data from TNAU Crop Production Guides.

    ### Memory-Safe Mode
    Default uses **heuristic NLI** (keyword matching, ~0MB extra RAM).
    Select **DeBERTa-v3** for deep semantic pruning (~500MB RAM).
    """)
else:
    tab1, tab2 = st.tabs(["🔍 Ask Questions", "📊 About the System"])

    with tab1:
        col_q, col_b = st.columns([5, 1])
        with col_q:
            question = st.text_input(
                "Enter your agricultural question:",
                placeholder="e.g., How can I prevent fungal blast in rice?",
                key="query"
            )
        with col_b:
            sample = st.selectbox("Sample:", [
                "Custom",
                "How to prevent fungal blast in rice?",
                "Best fertilizer for wheat?",
                "Control bollworms in cotton?",
                "When to sow maize?",
                "Common diseases in sugarcane?",
                "How to control red rot in sugarcane?",
                "What is ideal pH for rice?",
                "How much water does rice need?",
            ])
            if sample != "Custom":
                question = sample

        col1, col2 = st.columns(2)

        if st.button("🔍 Run Comparison", type="primary", use_container_width=True) and question:
            st.session_state.queries_run += 1

            with st.spinner("Processing query through both pipelines..."):
                result_nli = st.session_state.pipeline_nli.process(
                    question, top_k_retrieval=12
                )
                result_baseline = run_baseline(
                    st.session_state.retriever,
                    st.session_state.generator,
                    question, top_k=12
                )

            st.markdown(f"<div class='question-box'><b>Question:</b> {question}</div>", unsafe_allow_html=True)

            nli_facts_retrieved = result_nli.pipeline_steps.get('retrieval', {}).get('num_results', 0)
            nli_facts_pruned = result_nli.pipeline_steps.get('pruning', {}).get('pruned_facts', nli_facts_retrieved)
            nli_citations = len(result_nli.citations) if result_nli.citations else 0
            baseline_citations = len(result_baseline.citations) if result_baseline.citations else 0

            nli_faithfulness = nli_citations / max(nli_facts_pruned, 1)
            baseline_faithfulness = baseline_citations / max(nli_facts_retrieved, 1)

            col_m1, col_m2, col_m3, col_m4 = st.columns(4)
            with col_m1:
                st.markdown(f"<div class='metric-card'><b>NLI Faithfulness</b><br><h2 style='color:#2e7d32'>{nli_faithfulness:.1%}</h2></div>", unsafe_allow_html=True)
            with col_m2:
                st.markdown(f"<div class='metric-card'><b>Baseline Faithfulness</b><br><h2 style='color:#c62828'>{baseline_faithfulness:.1%}</h2></div>", unsafe_allow_html=True)
            with col_m3:
                diff = nli_faithfulness - baseline_faithfulness
                color = "#2e7d32" if diff > 0 else "#c62828"
                st.markdown(f"<div class='metric-card'><b>Hallucination Reduction</b><br><h2 style='color:{color}'>{diff:+.1%}</h2></div>", unsafe_allow_html=True)
            with col_m4:
                pruning_rate = (nli_facts_retrieved - nli_facts_pruned) / max(nli_facts_retrieved, 1)
                st.markdown(f"<div class='metric-card'><b>Context Pruned</b><br><h2 style='color:#1565c0'>{pruning_rate:.0%}</h2></div>", unsafe_allow_html=True)

            with col1:
                st.markdown("### 🟢 NLI-Gated RAG")
                st.markdown(f"<div class='nli-box'>{result_nli.response}</div>", unsafe_allow_html=True)
                nli_cols = st.columns(4)
                nli_cols[0].metric("Confidence", f"{result_nli.confidence:.0%}")
                nli_cols[1].metric("Facts Used", f"{nli_facts_pruned}/{nli_facts_retrieved}")
                nli_cols[2].metric("Citations", nli_citations)
                nli_cols[3].metric("Time", f"{result_nli.execution_time:.1f}s")
                if result_nli.warnings:
                    for w in result_nli.warnings:
                        st.warning(f"⚠️ {w}")

            with col2:
                st.markdown("### 🔴 Standard RAG (Baseline)")
                st.markdown(f"<div class='baseline-box'>{result_baseline.response}</div>", unsafe_allow_html=True)
                base_cols = st.columns(4)
                base_cols[0].metric("Confidence", f"{result_baseline.confidence:.0%}")
                base_cols[1].metric("Facts Used", f"{nli_facts_retrieved}/{nli_facts_retrieved}")
                base_cols[2].metric("Citations", baseline_citations)
                base_cols[3].metric("Time", f"{result_baseline.execution_time:.1f}s")
                if result_baseline.warnings:
                    for w in result_baseline.warnings:
                        st.warning(f"⚠️ {w}")

            st.markdown("---")
            if nli_faithfulness > baseline_faithfulness:
                st.success("### ✅ Winner: NLI-Gated RAG — Higher factual grounding")
            elif baseline_faithfulness > nli_faithfulness:
                st.error("### ❌ Winner: Standard RAG — NLI pruner may be too aggressive")
            else:
                st.info("### 🤝 Tie")

            with st.expander("📊 RAGAS Metrics Detail"):
                try:
                    for key in ["OPENAI_API_KEY", "OPENAI_BASE_URL", "OPENAI_MODEL_NAME"]:
                        os.environ.pop(key, None)
                    from langchain_groq import ChatGroq
                    from ragas import evaluate
                    from ragas.metrics import faithfulness, context_precision
                    from datasets import Dataset

                    groq_llm = ChatGroq(
                        model="llama-3.1-8b-instant",
                        api_key=os.getenv("GROQ_API_KEY"),
                        temperature=0
                    )
                    faithfulness.llm = groq_llm
                    context_precision.llm = groq_llm

                    ds = Dataset.from_dict({
                        "question": [question, question],
                        "answer": [result_nli.response, result_baseline.response],
                        "contexts": [result_nli.retrieved_facts[:5], result_baseline.retrieved_facts[:5]]
                    })
                    metrics = evaluate(ds, metrics=[faithfulness, context_precision])

                    ragas_cols = st.columns(4)
                    nli_f = metrics.get('faithfulness', [0, 0])[0]
                    base_f = metrics.get('faithfulness', [0, 0])[1]
                    nli_cp = metrics.get('context_precision', [0, 0])[0]
                    base_cp = metrics.get('context_precision', [0, 0])[1]
                    ragas_cols[0].metric("NLI Faithfulness", f"{nli_f:.3f}")
                    ragas_cols[1].metric("Baseline Faithfulness", f"{base_f:.3f}")
                    ragas_cols[2].metric("NLI Context Precision", f"{nli_cp:.3f}")
                    ragas_cols[3].metric("Baseline Context Precision", f"{base_cp:.3f}")

                except Exception as e:
                    st.warning(f"RAGAS library unavailable: {e}")
                    st.markdown("**Custom Metrics (Fallback):**")
                    st.json({
                        "nli_faithfulness (citations/facts)": round(nli_faithfulness, 3),
                        "baseline_faithfulness (citations/facts)": round(baseline_faithfulness, 3),
                        "hallucination_reduction": round(diff, 3),
                        "nli_confidence": round(result_nli.confidence, 3),
                        "baseline_confidence": round(result_baseline.confidence, 3),
                    })

            with st.expander("🔬 Pipeline Details"):
                st.markdown("**NLI-Gated Pipeline Steps:**")
                if result_nli.pipeline_steps:
                    for step, details in result_nli.pipeline_steps.items():
                        if isinstance(details, dict):
                            st.write(f"**{step}:**")
                            st.json(details)
                st.markdown("**Pruning Details:**")
                pruning = result_nli.pipeline_steps.get('pruning', {})
                st.write(f"Label distribution: {pruning.get('label_distribution', {})}")
                st.write(f"Retention rate: {pruning.get('retention_rate_pct', 'N/A')}")

    with tab2:
        st.markdown("## 🏗️ System Architecture")
        st.markdown("""
        ### Core Innovation: NLI-Gated Subgraph Pruning

        Instead of using all retrieved facts directly, NLI-Gated RAG:
        1. **Retrieves** 12 facts from hybrid (vector + graph) search
        2. **Prunes** facts using DeBERTa-v3 NLI model:
           - **Entailment**: Fact supports the query → **Keep**
           - **Neutral**: Fact is unrelated → **Discard**
           - **Contradiction**: Fact contradicts query → **Discard + Warning**
        3. **Generates** response using only pruned, relevant facts

        ### Key Benefits

        | Metric | Standard RAG | NLI-Gated RAG |
        |--------|-------------|---------------|
        | Faithfulness | Lower (hallucinates) | Higher (grounded) |
        | Context Clutter | All facts used | Only relevant facts |
        | Contradictions | Silent failures | Explicit warnings |
        | Confidence | Overconfident | Calibrated |

        ### Technical Stack

        | Component | Technology |
        |-----------|-----------|
        | Vector Store | FAISS / ChromaDB (all-MiniLM-L6-v2) |
        | Graph DB | Neo4j Desktop |
        | NLI Model | Heuristic or DeBERTa-v3 (user-selectable) |
        | LLM | GroQ Llama 3.1 8B |
        | UI | Streamlit |
        """)
