import streamlit as st
import sys
from pathlib import Path

# root path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))


from single_mode import render_single_mode
from batch_mode import render_batch_mode

def main():
    st.set_page_config(
        page_title="Dual-Mode Resume Screening Platform",
        layout="wide",
        initial_sidebar_state="expanded"
    )
    
    st.title("Dual-Mode Intelligent Resume Screening Platform")
    st.markdown("""
    **AI-Powered Resume Analysis using RAG Technology**
    
    This platform uses advanced embedding similarity and semantic analysis to evaluate resumes 
    against job descriptions without restrictive keyword matching.
    """)
    
    # Sidebar for mode selection
    st.sidebar.title("Analysis Mode")
    mode = st.sidebar.radio(
        "Select operation mode:",
        ["Single Resume Analysis", "Batch Shortlisting"],
        help="Choose between analyzing one resume in detail or ranking multiple resumes"
    )
    
    # Mode-specific rendering
    if mode == "Single Resume Analysis":
        render_single_mode()
    else:
        render_batch_mode()
    
    # Footer
    st.markdown("---")
    st.markdown("""
    **Technology Stack:** LangChain • ChromaDB • Hugging Face Transformers • Sentence-BERT
    
    *Built for semantic understanding and bias-free candidate evaluation*
    """)

if __name__ == "__main__":
    main()