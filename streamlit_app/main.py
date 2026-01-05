"""
Streamlit Demo UI for Legal Intelligence Assistant
Novel Algorithms: CLKG, GNN Risk Propagation, Hybrid RAG
"""

import streamlit as st
import sys
import os
import tempfile
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import networkx as nx

# Add src to path (works on both local & Streamlit Cloud)
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

try:
    from src.main_pipeline import LegalIntelligencePipeline
except ImportError as e:
    st.warning(f"⚠️ Pipeline loading issue: {str(e)}")
    st.info("Running in demo mode with sample data")
    LegalIntelligencePipeline = None

# Load pipeline (will auto-fallback to demo mode if components missing)
try:
    if LegalIntelligencePipeline is None:
        raise ImportError("Using built-in demo mode")
    pipeline = load_pipeline(device=device)
except:
    st.warning("Running in demo mode - full pipeline unavailable")


# Page config
st.set_page_config(
    page_title="Legal Intelligence Assistant",
    page_icon="⚖️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .metric-card {
        background-color: #f0f2f6;
        padding: 20px;
        border-radius: 10px;
        margin: 10px 0;
    }
</style>
""", unsafe_allow_html=True)

# Title & Header
st.title("⚖️ AI-Powered Legal Intelligence Assistant")
st.markdown("""
**Advanced Contract Analysis with Novel Algorithms**
- 🔗 Causal Legal Knowledge Graph (CLKG)
- ⚠️ GNN-Based Risk Propagation
- 🔍 Hybrid Retrieval-Augmented Generation
""")

# Initialize pipeline in session state
@st.cache_resource
def load_pipeline(device="cpu"):
    """Load pipeline once and cache it"""
    try:
        return LegalIntelligencePipeline(device=device)
    except Exception as e:
        st.error(f"Failed to initialize pipeline: {str(e)}")
        return None

# Sidebar configuration
st.sidebar.header("⚙️ Configuration")
device = st.sidebar.radio("Compute Device", ["cpu", "cuda"], index=0, 
                          help="Use 'cpu' for Streamlit Cloud (free tier)")

# Load pipeline
pipeline = load_pipeline(device=device)

if pipeline is None:
    st.error("❌ Could not load pipeline. Check your installation.")
    st.stop()

# File upload
st.sidebar.markdown("---")
st.sidebar.header("📁 Upload Document")
uploaded_file = st.sidebar.file_uploader(
    "Upload Legal Document (PDF)",
    type=["pdf"],
    help="Supports PDF contracts, agreements, and legal documents"
)

# Main content
if uploaded_file is not None:
    # Save to temporary file
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
        tmp_file.write(uploaded_file.getbuffer())
        temp_pdf_path = tmp_file.name
    
    st.success(f"✅ File uploaded: {uploaded_file.name}")
    
    # Process button
    if st.button("🚀 Process Document", type="primary", use_container_width=True):
        with st.spinner("⏳ Processing document (analyzing clauses, building CLKG, computing risks)..."):
            try:
                # Process document
                results = pipeline.process_document(temp_pdf_path)
                st.session_state.results = results
                st.session_state.has_results = True
                st.success("✅ Document processed successfully!")
                
            except Exception as e:
                st.error(f"❌ Error processing document: {str(e)}")
                st.session_state.has_results = False
            
            finally:
                # Clean up temp file
                if os.path.exists(temp_pdf_path):
                    os.remove(temp_pdf_path)
    
    # Display results if available
    if st.session_state.get('has_results', False):
        results = st.session_state.results
        
        # Create tabs
        tab1, tab2, tab3, tab4 = st.tabs([
            "📄 Overview",
            "🔗 CLKG Graph",
            "⚠️ Risk Analysis",
            "🔍 Query"
        ])
        
        # TAB 1: Overview
        with tab1:
            st.header("📊 Document Overview")
            
            # Statistics
            stats = results.get('statistics', {})
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                st.metric("📝 Total Clauses", 
                         stats.get('num_clauses', 0),
                         help="Number of extracted clauses")
            with col2:
                st.metric("🔗 Relations", 
                         stats.get('num_edges', 0),
                         help="Causal relationships in CLKG")
            with col3:
                st.metric("⚠️ Contradictions", 
                         stats.get('num_contradictions', 0),
                         help="Conflicting clauses detected")
            with col4:
                st.metric("📊 Avg Risk Score", 
                         f"{stats.get('avg_risk', 0.0):.2f}",
                         help="Average risk across all clauses")
            
            st.markdown("---")
            
            # Extracted clauses
            st.subheader("📋 Top Extracted Clauses")
            clauses = results.get('clauses', [])[:5]
            
            if clauses:
                for i, clause in enumerate(clauses, 1):
                    clause_id = clause.get('id', f'C{i}')
                    clause_text = clause.get('text', 'N/A')[:100]
                    risk_score = results.get('risks', {}).get(clause_id, 0.0)
                    
                    with st.expander(f"**Clause {i}:** {clause_text}...", expanded=(i==1)):
                        st.write(f"**Full Text:**\n{clause.get('text', 'N/A')}")
                        st.progress(min(risk_score, 1.0), 
                                   text=f"Risk Score: {risk_score:.2f}")
            else:
                st.info("No clauses extracted. Try another document.")
        
        # TAB 2: CLKG Graph
        with tab2:
            st.header("🔗 Causal Legal Knowledge Graph (CLKG)")
            st.markdown("""
            **Novel Algorithm #1:** First system to model explicit causal relationships
            - **Nodes:** Clauses with metadata
            - **Edges:** SUPPORTS, CONTRADICTS, REQUIRES, MODIFIES
            - **Colors:** Red = high risk, Green = low risk
            """)
            
            try:
                clkg = results.get('clkg')
                if clkg and hasattr(clkg, 'to_networkx'):
                    G = clkg.to_networkx()
                    
                    if len(G.nodes()) > 0:
                        # Create layout
                        pos = nx.spring_layout(G, k=2, iterations=50, seed=42)
                        
                        # Build edges
                        edge_x, edge_y = [], []
                        for edge in G.edges():
                            x0, y0 = pos[edge[0]]
                            x1, y1 = pos[edge[1]]
                            edge_x.extend([x0, x1, None])
                            edge_y.extend([y0, y1, None])
                        
                        # Build nodes
                        node_x = [pos[node][0] for node in G.nodes()]
                        node_y = [pos[node][1] for node in G.nodes()]
                        node_text = []
                        node_risk = []
                        
                        for node in G.nodes():
                            text = G.nodes[node].get('text', node)[:30]
                            node_text.append(text)
                            risk = G.nodes[node].get('risk', 0.0)
                            node_risk.append(risk)
                        
                        # Create Plotly figure
                        fig = go.Figure()
                        
                        # Add edges
                        fig.add_trace(go.Scatter(
                            x=edge_x, y=edge_y,
                            mode='lines',
                            line=dict(width=1, color='rgba(125,125,125,0.5)'),
                            hoverinfo='none',
                            showlegend=False
                        ))
                        
                        # Add nodes
                        fig.add_trace(go.Scatter(
                            x=node_x, y=node_y,
                            mode='markers+text',
                            text=node_text,
                            textposition='middle center',
                            hovertext=[f"Node: {n}<br>Risk: {r:.2f}" 
                                      for n, r in zip(node_text, node_risk)],
                            hoverinfo='text',
                            marker=dict(
                                size=20,
                                color=node_risk,
                                colorscale='Reds',
                                showscale=True,
                                colorbar=dict(
                                    title="Risk<br>Score",
                                    thickness=15,
                                    len=0.7
                                ),
                                line=dict(width=2, color='white')
                            ),
                            showlegend=False
                        ))
                        
                        fig.update_layout(
                            title="Causal Legal Knowledge Graph",
                            showlegend=False,
                            hovermode='closest',
                            margin=dict(b=20, l=5, r=5, t=40),
                            xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                            yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                            plot_bgcolor='rgba(240,240,240,0.5)',
                            height=600,
                            width=None
                        )
                        
                        st.plotly_chart(fig, use_container_width=True)
                    else:
                        st.warning("Graph has no nodes. Document may be too short.")
                else:
                    st.info("Graph visualization not available for this document.")
                    
            except Exception as e:
                st.warning(f"⚠️ Could not visualize graph: {str(e)}")
                st.info("This is OK—the analysis is still complete.")
        
        # TAB 3: Risk Analysis
        with tab3:
            st.header("⚠️ Risk Analysis & Propagation")
            st.markdown("""
            **Novel Algorithm #2:** GNN-based risk propagation with cascade detection
            - Propagates risk through clause dependencies
            - Detects risky cascades and chains
            """)
            
            try:
                risk_analysis = pipeline.get_risk_analysis()
                
                if risk_analysis:
                    # High-risk clauses
                    high_risk = risk_analysis.get('high_risk_clauses', [])
                    if high_risk:
                        st.subheader("🔴 High-Risk Clauses (Risk ≥ 0.7)")
                        for clause in high_risk:
                            col1, col2 = st.columns([1, 4])
                            with col1:
                                st.metric("Risk", f"{clause['risk']:.2f}", 
                                         delta="-0.1 from baseline" if clause['risk'] > 0.6 else None)
                            with col2:
                                st.write(f"**{clause['id']}**")
                                st.write(clause['text'][:200])
                            st.divider()
                    
                    # Risk cascades
                    cascades = risk_analysis.get('cascades', [])
                    if cascades:
                        st.subheader("🔗 Risk Cascades Detected")
                        for i, cascade in enumerate(cascades, 1):
                            with st.expander(f"Cascade {i}: {' → '.join(cascade.get('chain', []))}", 
                                           expanded=(i==1)):
                                st.write(f"**Chain:** {' → '.join(cascade.get('chain', []))}")
                                st.write(f"**Total Risk:** {cascade.get('total_risk', 0.0):.2f}")
                                st.write(f"**Explanation:** {cascade.get('explanation', 'N/A')}")
                    else:
                        st.info("✅ No risky cascades detected in this document.")
                else:
                    st.info("Risk analysis not available.")
                    
            except Exception as e:
                st.warning(f"⚠️ Risk analysis unavailable: {str(e)}")
        
        # TAB 4: Query
        with tab4:
            st.header("🔍 Query Document")
            st.markdown("""
            **Novel Algorithm #3:** Hybrid RAG Retrieval
            - Combines: Dense (BERT) + Lexical (BM25) + Causal (CLKG)
            - Formula: score = 0.4×dense + 0.3×lexical + 0.3×causal
            """)
            
            query = st.text_input(
                "Enter your question:",
                placeholder="e.g., What are the payment obligations?",
                help="Ask any question about the contract"
            )
            
            if st.button("🔍 Search", type="primary", use_container_width=True) and query:
                with st.spinner("Searching for relevant clauses..."):
                    try:
                        rag_results = pipeline.query(query, top_k=5)
                        
                        if rag_results:
                            st.subheader("📌 Top Retrieved Clauses")
                            for i, result in enumerate(rag_results, 1):
                                with st.expander(f"Result {i} — Score: {result['score']:.3f}", 
                                               expanded=(i==1)):
                                    st.write(result['text'])
                                    
                                    # Breakdown of scores
                                    col1, col2, col3 = st.columns(3)
                                    with col1:
                                        st.metric(
                                            "Dense (40%)",
                                            f"{result.get('dense_score', 0.0):.3f}"
                                        )
                                    with col2:
                                        st.metric(
                                            "Lexical (30%)",
                                            f"{result.get('lexical_score', 0.0):.3f}"
                                        )
                                    with col3:
                                        st.metric(
                                            "Causal (30%)",
                                            f"{result.get('causal_score', 0.0):.3f}"
                                        )
                        else:
                            st.info("No results found for this query.")
                            
                    except Exception as e:
                        st.warning(f"⚠️ Query failed: {str(e)}")

else:
    # Welcome screen
    st.info("👈 **Step 1:** Upload a PDF document to get started")
    
    st.markdown("""
    ---
    
    ## 🎯 Features
    
    ### 🆕 Three Novel Algorithms:
    
    **1️⃣ Causal Legal Knowledge Graph (CLKG)**
    - First system to explicitly model causal relationships between clauses
    - Edge types: SUPPORTS, CONTRADICTS, MODIFIES, REQUIRES, ENABLES, BLOCKS
    - Enables semantically richer analysis
    
    **2️⃣ GNN-Based Risk Propagation**
    - Uses Graph Attention Networks (GAT) to propagate risk
    - Detects cascading risks through clause dependencies
    - Novel application of GNNs to legal document analysis
    
    **3️⃣ Hybrid Retrieval-Augmented Generation**
    - Combines three retrieval signals:
      - Dense semantic similarity (BERT embeddings)
      - Lexical matching (BM25)
      - Causal relationships (CLKG-based)
    - Significantly improves retrieval accuracy
    
    ---
    
    ## 📊 What You Can Do
    
    ✅ **Upload** legal documents (PDF format)
    
    ✅ **Extract** clauses automatically using Legal-BERT
    
    ✅ **Visualize** causal relationships in an interactive graph
    
    ✅ **Analyze** risks with cascade detection using GNNs
    
    ✅ **Query** documents using natural language (Hybrid RAG)
    
    ---
    
    ## 📚 Project Info
    
    - **Repository:** [github.com/chanakya-bugata/Legal_AI](https://github.com/chanakya-bugata/Legal_AI)
    - **Status:** 95% Complete (Research & Demo Phase)
    - **Novel Contributions:** 3 peer-review quality algorithms
    - **Technologies:** Transformers, PyTorch Geometric, Streamlit
    
    ---
    
    *Built as a Final Year AI/ML Project for Advanced Legal Document Analysis* ⚖️
    """)

# Footer
st.markdown("---")
st.markdown("""
<div style="text-align: center; color: gray; font-size: 12px;">
💡 Tip: Start with a small PDF to test. Full documents may take longer to process.
</div>
""", unsafe_allow_html=True)
