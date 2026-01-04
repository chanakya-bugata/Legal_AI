# AI-Powered Context-Aware Legal Intelligence Assistant with Novel Clause Reasoning

## 🎯 Research Contribution

This project introduces **three novel algorithmic contributions** to legal AI:

1. **Causal Legal Knowledge Graph (CLKG)** - First system to model explicit causal relationships between clauses (SUPPORTS, CONTRADICTS, MODIFIES, etc.)
2. **GNN-Based Risk Propagation** - Novel application of Graph Neural Networks to detect cascading risks through clause dependencies
3. **Hybrid Retrieval-Augmented Generation** - Combines dense + lexical + causal retrieval signals for improved legal document understanding

## 📁 Project Structure

```
legal-intelligence-assistant/
├── src/
│   ├── document_processing/     # Component 1: Multi-modal document encoder
│   ├── clause_extraction/        # Component 2: Clause & relation extraction
│   ├── clkg/                     # Component 3: Causal Legal Knowledge Graph (NOVEL)
│   ├── risk_propagation/         # Component 4: GNN risk propagation (NOVEL)
│   ├── rag/                      # Component 5: Hybrid RAG pipeline (NOVEL)
│   ├── generation/               # Component 6: Legal QA & drafting
│   └── utils/                    # Shared utilities
├── data/
│   ├── raw/                      # Raw legal documents
│   ├── processed/                # Processed documents
│   └── annotations/              # Manual annotations
├── models/
│   ├── checkpoints/              # Model checkpoints
│   └── embeddings/               # Pre-computed embeddings
├── experiments/
│   ├── baseline/                 # Baseline comparison experiments
│   ├── ablation/                 # Ablation studies
│   └── results/                  # Experimental results
├── evaluation/
│   ├── legalbench/               # LegalBench evaluation
│   └── metrics/                  # Custom metrics
├── notebooks/                    # Jupyter notebooks for exploration
├── tests/                        # Unit tests
├── docs/                         # Documentation
└── streamlit_app/                # Streamlit demo UI
```

## 🚀 Quick Start

### Installation

```bash
pip install -r requirements.txt
```

### Run Demo

```bash
streamlit run streamlit_app/main.py
```

## 📊 Novel Algorithms Explained

### 1. Causal Legal Knowledge Graph (CLKG)

**Problem:** Existing systems use semantic similarity, missing causal relationships between clauses.

**Solution:** Knowledge graph with explicit causal edge types:
- SUPPORTS, CONTRADICTS, MODIFIES, OVERTURNS, ENABLES, BLOCKS, REQUIRES

**Novelty:** First legal AI system to model explicit causality (not just similarity).

### 2. GNN-Based Risk Propagation

**Problem:** Baseline systems score clauses independently, missing cascade effects.

**Solution:** Graph Attention Network propagates risk through dependency graph.

**Novelty:** Novel application of GNNs to legal risk analysis with cascade detection.

### 3. Hybrid RAG

**Problem:** Single retrieval method (dense or lexical) misses relevant clauses.

**Solution:** Combines three signals: dense (semantic), lexical (BM25), causal (graph-based).

**Novelty:** First hybrid retrieval combining semantic, lexical, and causal signals for legal documents.

## 📈 Evaluation

- **Baseline Comparison:** Standard RAG vs. Hybrid RAG
- **Ablation Studies:** Impact of each component (CLKG, GNN, hybrid retrieval)
- **LegalBench:** Evaluation on 162 legal understanding tasks
- **Metrics:** F1 (clause extraction), Recall (risk detection), MAP@5 (retrieval)

## 🔬 Research Methodology

1. **Baseline:** Standard clause extraction + dense RAG
2. **Proposed:** CLKG + GNN + Hybrid RAG
3. **Ablation:** Remove each component to measure contribution
4. **Evaluation:** LegalBench + custom metrics

## 📝 Citation

If you use this work, please cite:

```bibtex
@misc{legal-intelligence-assistant,
  title={AI-Powered Context-Aware Legal Intelligence Assistant with Novel Clause Reasoning},
  author={Your Name},
  year={2025},
  note={Final Year Major Project}
}
```

## 📄 License

MIT License - Open Source

## 🙏 Acknowledgments

- Legal-BERT (NLPAUEB)
- LayoutLMv3 (Microsoft)
- LegalBench (HazyResearch)
- ContractNLI (Stanford)

