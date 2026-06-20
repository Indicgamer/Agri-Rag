"""
Agri-RAG Web UI: NLI-Gated vs Baseline RAG Comparison
Shows side-by-side answers with RAGAS metrics and hallucination rates
"""

import streamlit as st
import sys
import os
from pathlib import Path
import json
import time
import logging
from datetime import datetime
from dotenv import load_dotenv

load_dotenv()
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.retrieval.neo4j_retriever import KnowledgeGraph
from src.retrieval.faiss_retriever import FAISSVectorStore
from src.retrieval.hybrid_retriever import HybridRetriever
from src.pruning.nli_pruner import NLIPruner
from src.models.response_generator import ResponseGenerator
from src.models.rag_pipeline import AgriRAGPipeline, PipelineResult
from src.evaluation.ragas_evaluator import RAGASEvaluator, HallucinationDetector
from configs.settings import settings

logging.basicConfig(level=logging.WARNING)
logger = logging.getLogger(__name__)

# Page config
st.set_page_config(
    page_title="Agri-RAG: NLI vs Baseline Comparison",
    page_icon="🌾",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main { padding: 1rem; }
    .header { text-align: center; margin-bottom: 1rem; }
    .nli-box { background-color: #e8f5e9; padding: 1.2rem; border-radius: 0.5rem; border-left: 5px solid #2e7d32; margin: 0.5rem 0; }
    .baseline-box { background-color: #fce4ec; padding: 1.2rem; border-radius: 0.5rem; border-left: 5px solid #c62828; margin: 0.5rem 0; }
    .metric-good { color: #2e7d32; font-weight: bold; }
    .metric-bad { color: #c62828; font-weight: bold; }
    .metric-card { 
        background-color: #f5f5f5; 
        padding: 1rem; 
        border-radius: 0.5rem; 
        text-align: center;
        border: 2px solid #ddd;
    }
    .winner-badge { 
        background-color: #ffd700; 
        padding: 0.3rem 1rem; 
        border-radius: 1rem; 
        font-weight: bold;
        font-size: 0.9rem;
        margin: 0.5rem 0;
    }
    .question-box { 
        background-color: #fff8e1; 
        padding: 1.2rem; 
        border-radius: 0.5rem; 
        border-left: 5px solid #f9a825; 
        margin: 1rem 0; 
    }
</style>
""", unsafe_allow_html=True)

# Initialize session state
if 'nli_pipeline' not in st.session_state:
    st.session_state.nli_pipeline = None
    st.session_state.baseline_pipeline = None
    st.session_state.initialized = False
    st.session_state.evaluator = RAGASEvaluator()
    st.session_state.hallucination_detector = HallucinationDetector()


@st.cache_resource
def initialize_pipelines():
    """Initialize both pipelines"""
    try:
        # NLI Pipeline
        neo4j_nli = KnowledgeGraph(
            uri=settings.neo4j.uri,
            username=settings.neo4j.username,
            password=settings.neo4j.password,
            database=settings.neo4j.database
        )
        
        faiss_nli = FAISSVectorStore(
            collection_name=settings.faiss.collection_name,
            embedding_model=settings.faiss.embedding_model,
            persistence_dir=str(settings.faiss.persistence_dir)
        )
        
        retriever_nli = HybridRetriever(
            vector_store=faiss_nli,
            knowledge_graph=neo4j_nli,
            vector_weight=0.35,
            graph_weight=0.65,
            top_k_vector=5,
            top_k_graph=10
        )
        
        nli_pruner = NLIPruner(
            model_name="cross-encoder/nli-deberta-v3-base",
            provider="transformers",
            device="cpu",
            entailment_threshold=0.35,
            neutral_threshold=0.20
        )
        
        response_gen_nli = ResponseGenerator(
            provider="groq",
            model_name="llama-3.1-8b-instant",
            temperature=0.3,
            max_tokens=1000
        )
        
        nli_pipeline = AgriRAGPipeline(
            hybrid_retriever=retriever_nli,
            nli_pruner=nli_pruner,
            response_generator=response_gen_nli
        )
        
        # Baseline Pipeline
        neo4j_base = KnowledgeGraph(
            uri=settings.neo4j.uri,
            username=settings.neo4j.username,
            password=settings.neo4j.password,
            database=settings.neo4j.database
        )
        
        faiss_base = FAISSVectorStore(
            collection_name=settings.faiss.collection_name,
            embedding_model=settings.faiss.embedding_model,
            persistence_dir=str(settings.faiss.persistence_dir)
        )
        
        retriever_base = HybridRetriever(
            vector_store=faiss_base,
            knowledge_graph=neo4j_base,
            vector_weight=0.35,
            graph_weight=0.65,
            top_k_vector=5,
            top_k_graph=10
        )
        
        response_gen_base = ResponseGenerator(
            provider="groq",
            model_name="llama-3.1-8b-instant",
            temperature=0.3,
            max_tokens=1000
        )
        
        class BaselineRAG:
            def __init__(self, retriever, generator):
                self.retriever = retriever
                self.generator = generator
            
            def process(self, query, top_k=10):
                facts = self.retriever.retrieve(query, top_k=top_k)
                fact_texts = [f.content if hasattr(f, 'content') else str(f) for f in facts]
                response = self.generator.generate(query, fact_texts)
                
                return PipelineResult(
                    query=query,
                    response=response.response if hasattr(response, 'response') else str(response),
                    confidence=0.85,
                    citations=[{"text": f, "idx": i} for i, f in enumerate(fact_texts[:3])],
                    warnings=[],
                    pipeline_steps={'retrieval': {'num_results': len(facts)}},
                    execution_time=0.5,
                    timestamp=datetime.now().isoformat(),
                    retrieved_facts=fact_texts
                )
        
        baseline_pipeline = BaselineRAG(retriever_base, response_gen_base)
        
        return nli_pipeline, baseline_pipeline
    except Exception as e:
        logger.error(f"Pipeline initialization failed: {e}")
        return None, None


# Header
st.markdown("""
<div class="header">
<h1>🌾 Agri-RAG Comparison System</h1>
<h3>NLI-Gated vs Standard RAG</h3>
<p><i>Side-by-side evaluation with RAGAS metrics and hallucination detection</i></p>
</div>
""", unsafe_allow_html=True)

# Sidebar
with st.sidebar:
    st.markdown("## ⚙️ System Controls")
    
    if st.button("🚀 Initialize System", key="init_btn", use_container_width=True):
        with st.spinner("Initializing pipelines..."):
            nli_p, baseline_p = initialize_pipelines()
            if nli_p and baseline_p:
                st.session_state.nli_pipeline = nli_p
                st.session_state.baseline_pipeline = baseline_p
                st.session_state.initialized = True
                st.success("✓ System initialized!")
            else:
                st.error("✗ Initialization failed")
    
    st.markdown("---")
    st.markdown("### 📊 About This System")
    st.write("""
    **Compares two approaches:**
    - **NLI-Gated RAG**: Uses Natural Language Inference to prune unreliable facts
    - **Baseline RAG**: Standard retrieval without pruning
    
    **Metrics Evaluated:**
    - Faithfulness: How grounded is the answer?
    - Answer Relevance: Does it address the question?
    - Context Precision: Are retrieved facts relevant?
    - Hallucination Rate: % of response not grounded in facts
    """)


# Main content
if not st.session_state.initialized:
    st.warning("⚠️ Click 'Initialize System' in the sidebar to start")
else:
    # Query input
    st.markdown("""
    <div class="question-box">
    <h3>🔍 Enter Your Agricultural Question</h3>
    </div>
    """, unsafe_allow_html=True)
    
    query = st.text_input(
        "Question:",
        placeholder="e.g., How to control fungal blast in rice?",
        label_visibility="collapsed"
    )
    
    if st.button("💡 Get Answers", use_container_width=True, type="primary"):
        if query:
            with st.spinner("Generating answers from both systems..."):
                try:
                    start_time = time.time()
                    
                    # Get NLI response
                    nli_result = st.session_state.nli_pipeline.process(query, top_k_retrieval=10)
                    
                    # Get Baseline response
                    baseline_result = st.session_state.baseline_pipeline.process(query, top_k=10)
                    
                    execution_time = time.time() - start_time
                    
                    # Extract facts
                    nli_facts = getattr(nli_result, 'retrieved_facts', [])
                    baseline_facts = getattr(baseline_result, 'retrieved_facts', [])
                    
                    # Calculate metrics
                    nli_metrics = st.session_state.evaluator.evaluate_single(
                        question=query,
                        response=nli_result.response,
                        retrieved_facts=nli_facts,
                        citations=nli_result.citations or [],
                        ground_truth=""
                    )
                    
                    baseline_metrics = st.session_state.evaluator.evaluate_single(
                        question=query,
                        response=baseline_result.response,
                        retrieved_facts=baseline_facts,
                        citations=baseline_result.citations or [],
                        ground_truth=""
                    )
                    
                    # Calculate hallucination rates
                    nli_hallucination = st.session_state.hallucination_detector.calculate_hallucination_rate(
                        nli_result.response,
                        nli_result.citations or [],
                        len(nli_facts)
                    )
                    
                    baseline_hallucination = st.session_state.hallucination_detector.calculate_hallucination_rate(
                        baseline_result.response,
                        baseline_result.citations or [],
                        len(baseline_facts)
                    )
                    
                    # Determine winner
                    nli_score = nli_metrics.average_score()
                    baseline_score = baseline_metrics.average_score()
                    nli_wins = nli_score > baseline_score
                    
                    # Display results
                    st.markdown("---")
                    st.markdown("### 📋 Results")
                    
                    col1, col2 = st.columns(2)
                    
                    # NLI-Gated Column
                    with col1:
                        st.markdown("""
                        <div class="nli-box">
                        <h3>🧠 NLI-Gated RAG</h3>
                        """, unsafe_allow_html=True)
                        
                        if nli_wins:
                            st.markdown('<div class="winner-badge">🏆 BETTER ANSWER</div>', unsafe_allow_html=True)
                        
                        st.write("**Answer:**")
                        st.info(nli_result.response[:600] + "..." if len(nli_result.response) > 600 else nli_result.response)
                        
                        st.markdown("**RAGAS Metrics:**")
                        col_a, col_b = st.columns(2)
                        with col_a:
                            st.metric("Faithfulness", f"{nli_metrics.faithfulness:.1%}")
                            st.metric("Answer Relevance", f"{nli_metrics.answer_relevance:.1%}")
                        with col_b:
                            st.metric("Context Precision", f"{nli_metrics.context_precision:.1%}")
                            st.metric("Context Recall", f"{nli_metrics.context_recall:.1%}")
                        
                        st.markdown("**Quality Metrics:**")
                        col_c, col_d = st.columns(2)
                        with col_c:
                            hall_color = "normal" if nli_hallucination < 0.3 else "inverse"
                            st.metric("Hallucination Rate", f"{nli_hallucination:.1%}", delta="Lower is better", delta_color="inverse")
                        with col_d:
                            st.metric("Overall Score", f"{nli_score:.1%}")
                        
                        st.markdown("""</div>""", unsafe_allow_html=True)
                    
                    # Baseline Column
                    with col2:
                        st.markdown("""
                        <div class="baseline-box">
                        <h3>📚 Baseline RAG</h3>
                        """, unsafe_allow_html=True)
                        
                        if not nli_wins:
                            st.markdown('<div class="winner-badge">🏆 BETTER ANSWER</div>', unsafe_allow_html=True)
                        
                        st.write("**Answer:**")
                        st.info(baseline_result.response[:600] + "..." if len(baseline_result.response) > 600 else baseline_result.response)
                        
                        st.markdown("**RAGAS Metrics:**")
                        col_a, col_b = st.columns(2)
                        with col_a:
                            st.metric("Faithfulness", f"{baseline_metrics.faithfulness:.1%}")
                            st.metric("Answer Relevance", f"{baseline_metrics.answer_relevance:.1%}")
                        with col_b:
                            st.metric("Context Precision", f"{baseline_metrics.context_precision:.1%}")
                            st.metric("Context Recall", f"{baseline_metrics.context_recall:.1%}")
                        
                        st.markdown("**Quality Metrics:**")
                        col_c, col_d = st.columns(2)
                        with col_c:
                            hall_color = "normal" if baseline_hallucination < 0.3 else "inverse"
                            st.metric("Hallucination Rate", f"{baseline_hallucination:.1%}", delta="Lower is better", delta_color="inverse")
                        with col_d:
                            st.metric("Overall Score", f"{baseline_score:.1%}")
                        
                        st.markdown("""</div>""", unsafe_allow_html=True)
                    
                    # Comparison summary
                    st.markdown("---")
                    st.markdown("### 📊 Comparison Summary")
                    
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        diff = nli_metrics.faithfulness - baseline_metrics.faithfulness
                        st.metric("Faithfulness Improvement", f"{diff:+.1%}", 
                                 delta_color="normal" if diff > 0 else "inverse")
                    
                    with col2:
                        diff = baseline_hallucination - nli_hallucination
                        st.metric("Hallucination Reduction", f"{diff:+.1%}",
                                 delta_color="normal" if diff > 0 else "inverse")
                    
                    with col3:
                        diff = nli_score - baseline_score
                        st.metric("Overall Quality Improvement", f"{diff:+.1%}",
                                 delta_color="normal" if diff > 0 else "inverse")
                    
                    st.success(f"✓ Analysis completed in {execution_time:.2f}s")
                    
                except Exception as e:
                    st.error(f"Error: {str(e)}")
                    logger.exception(e)
        else:
            st.warning("Please enter a question")
    
    # Sample questions
    st.markdown("---")
    st.markdown("### 💡 Sample Questions")
    
    sample_questions = [
        "How to control fungal blast in rice?",
        "What nitrogen dose is recommended for rice?",
        "What are the best rice varieties in Tamil Nadu?",
        "How to control pests in cotton?",
        "What is the water requirement for rice?"
    ]
    
    cols = st.columns(2)
    for i, q in enumerate(sample_questions):
        with cols[i % 2]:
            if st.button(q, key=f"sample_{i}", use_container_width=True):
                st.session_state.sample_query = q
                st.rerun()
    
    if 'sample_query' in st.session_state:
        st.info(f"Selected: {st.session_state.sample_query}")
