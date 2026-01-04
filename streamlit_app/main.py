"""
Streamlit Demo UI for Legal Intelligence Assistant
"""

import streamlit as st
import sys
import os

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.main_pipeline import LegalIntelligencePipeline
import networkx as nx
import plotly.graph_objects as go
from plotly.subplots import make_subplots


st.set_page_config(
    page_title="Legal Intelligence Assistant",
    page_icon="⚖️",
    layout="wide"
)

st.title("⚖️ AI-Powered Legal Intelligence Assistant")
st.markdown("**Novel Clause Reasoning with CLKG, GNN Risk Propagation, and Hybrid RAG**")

# Sidebar
st.sidebar.header("Configuration")
device = st.sidebar.selectbox("Device", ["cpu", "cuda"], index=0)
uploaded_file = st.sidebar.file_uploader("Upload Legal Document (PDF)", type=["pdf"])

# Initialize pipeline
if 'pipeline' not in st.session_state:
    st.session_state.pipeline = LegalIntelligencePipeline(device=device)

# Main content
if uploaded_file is not None:
    # Save uploaded file temporarily
    with open("temp_document.pdf", "wb") as f:
        f.write(uploaded_file.getbuffer())
    
    # Process document
    if st.button("Process Document", type="primary"):
        with st.spinner("Processing document... This may take a few minutes."):
            try:
                results = st.session_state.pipeline.process_document("temp_document.pdf")
                st.session_state.results = results
                st.success("Document processed successfully!")
            except Exception as e:
                st.error(f"Error processing document: {str(e)}")
    
    # Display results
    if 'results' in st.session_state:
        results = st.session_state.results
        
        # Tabs
        tab1, tab2, tab3, tab4 = st.tabs([
            "📄 Overview",
            "🔗 Knowledge Graph",
            "⚠️ Risk Analysis",
            "🔍 Query"
        ])
        
        with tab1:
            st.header("Document Overview")
            
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("Clauses", results['statistics']['num_clauses'])
            with col2:
                st.metric("Relations", results['statistics']['num_edges'])
            with col3:
                st.metric("Contradictions", results['statistics']['num_contradictions'])
            with col4:
                st.metric("Avg Risk", f"{results['statistics']['avg_risk']:.2f}")
            
            st.subheader("Extracted Clauses")
            for i, clause in enumerate(results['clauses'][:10], 1):
                with st.expander(f"Clause {i}: {clause['text'][:50]}..."):
                    st.write(clause['text'])
                    risk = results['risks'].get(clause['id'], 0.0)
                    st.progress(risk, text=f"Risk Score: {risk:.2f}")
        
        with tab2:
            st.header("Causal Legal Knowledge Graph (CLKG)")
            st.markdown("**Novel Contribution:** First system to model explicit causal relationships")
            
            # Visualize graph
            try:
                G = results['clkg'].to_networkx()
                
                # Create network layout
                pos = nx.spring_layout(G, k=1, iterations=50)
                
                # Extract edge and node info
                edge_x = []
                edge_y = []
                for edge in G.edges():
                    x0, y0 = pos[edge[0]]
                    x1, y1 = pos[edge[1]]
                    edge_x.extend([x0, x1, None])
                    edge_y.extend([y0, y1, None])
                
                node_x = [pos[node][0] for node in G.nodes()]
                node_y = [pos[node][1] for node in G.nodes()]
                node_text = [G.nodes[node].get('text', node) for node in G.nodes()]
                
                # Create plotly figure
                fig = go.Figure()
                
                # Add edges
                fig.add_trace(go.Scatter(
                    x=edge_x, y=edge_y,
                    line=dict(width=0.5, color='#888'),
                    hoverinfo='none',
                    mode='lines'
                ))
                
                # Add nodes
                fig.add_trace(go.Scatter(
                    x=node_x, y=node_y,
                    mode='markers+text',
                    text=node_text,
                    textposition="middle right",
                    hoverinfo='text',
                    marker=dict(
                        size=10,
                        color=[G.nodes[node].get('risk', 0.0) for node in G.nodes()],
                        colorscale='Reds',
                        showscale=True,
                        colorbar=dict(title="Risk Score")
                    )
                ))
                
                fig.update_layout(
                    title="CLKG Visualization",
                    showlegend=False,
                    hovermode='closest',
                    margin=dict(b=20, l=5, r=5, t=40),
                    xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                    yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                    height=600
                )
                
                st.plotly_chart(fig, use_container_width=True)
            except Exception as e:
                st.warning(f"Graph visualization not available: {str(e)}")
                st.write("Graph Statistics:")
                st.json(results['statistics'])
        
        with tab3:
            st.header("Risk Analysis")
            st.markdown("**Novel Contribution:** GNN-based risk propagation with cascade detection")
            
            risk_analysis = st.session_state.pipeline.get_risk_analysis()
            
            st.subheader("High-Risk Clauses (Risk ≥ 0.7)")
            for clause in risk_analysis['high_risk_clauses']:
                st.write(f"**{clause['id']}** (Risk: {clause['risk']:.2f})")
                st.write(clause['text'])
                st.divider()
            
            st.subheader("Risk Cascades")
            if risk_analysis['cascades']:
                for cascade in risk_analysis['cascades']:
                    st.write(f"**Chain:** {' → '.join(cascade['chain'])}")
                    st.write(f"**Total Risk:** {cascade['total_risk']:.2f}")
                    st.write(f"**Explanation:** {cascade['explanation']}")
                    st.divider()
            else:
                st.info("No risk cascades detected.")
        
        with tab4:
            st.header("Query Document")
            st.markdown("**Novel Contribution:** Hybrid RAG (dense + lexical + causal)")
            
            query = st.text_input("Enter your query:", placeholder="e.g., What are the payment obligations?")
            
            if st.button("Search", type="primary") and query:
                with st.spinner("Searching..."):
                    results_rag = st.session_state.pipeline.query(query, top_k=5)
                
                st.subheader("Retrieved Clauses")
                for i, result in enumerate(results_rag, 1):
                    with st.expander(f"Result {i}: Score {result['score']:.3f}"):
                        st.write(result['text'])
                        col1, col2, col3 = st.columns(3)
                        with col1:
                            st.metric("Dense", f"{result['dense_score']:.3f}")
                        with col2:
                            st.metric("Lexical", f"{result['lexical_score']:.3f}")
                        with col3:
                            st.metric("Causal", f"{result['causal_score']:.3f}")

else:
    st.info("👈 Please upload a PDF document to get started")
    
    st.markdown("""
    ## Features
    
    ### 🆕 Three Novel Algorithms:
    
    1. **Causal Legal Knowledge Graph (CLKG)**
       - First system to model explicit causal relationships
       - Edge types: SUPPORTS, CONTRADICTS, MODIFIES, etc.
    
    2. **GNN-Based Risk Propagation**
       - Detects cascading risks through dependencies
       - Novel application of Graph Neural Networks
    
    3. **Hybrid Retrieval-Augmented Generation**
       - Combines dense + lexical + causal retrieval
       - Improves accuracy over single-method approaches
    
    ### 📊 What You Can Do:
    
    - Upload legal documents (PDF)
    - Extract clauses automatically
    - Visualize causal relationships
    - Analyze risks with cascade detection
    - Query documents using natural language
    """)

