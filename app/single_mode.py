import streamlit as st
import tempfile
import os
from pathlib import Path
import json
import uuid

from src.services.resume_analyzer import ResumeAnalyzer
from src.models.job_description_model import JobDescription

def render_single_mode():
    """Render the single resume analysis interface."""
    st.header("Single Resume Analysis")
    st.markdown("Upload a resume and job description for detailed compatibility analysis.")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Job Description")
        
        # Job title input
        job_title = st.text_input("Job Title", placeholder="e.g., Senior Software Engineer")
        
        # Job description input
        job_description_text = st.text_area(
            "Job Description",
            height=300,
            placeholder="Paste the complete job description here..."
        )
    
    with col2:
        st.subheader("Resume Upload")
        
        # Resume file upload
        uploaded_file = st.file_uploader(
            "Upload Resume",
            type=['pdf', 'txt'],
            help="Upload a PDF or text file containing the candidate's resume"
        )
        
        if uploaded_file:
            st.success(f"File uploaded: {uploaded_file.name}")
            
            # Show file details
            file_size = len(uploaded_file.getvalue()) / 1024
            st.info(f"File size: {file_size:.1f} KB")
    
    # Analysis button
    st.markdown("---")
    
    if st.button("🔍 Analyze Resume", type="primary", use_container_width=True):
        if not job_description_text.strip():
            st.error("Please provide a job description")
            return
        
        if not uploaded_file:
            st.error("Please upload a resume file")
            return
        
        # Show progress
        with st.spinner("Analyzing resume using AI..."):
            try:
                # Save uploaded file temporarily
                with tempfile.NamedTemporaryFile(delete=False, suffix=f".{uploaded_file.name.split('.')[-1]}") as tmp_file:
                    tmp_file.write(uploaded_file.getvalue())
                    tmp_file_path = tmp_file.name
                
                # Create job description object
                job_desc = JobDescription(
                    id=str(uuid.uuid4()),
                    title=job_title or "Job Analysis",
                    content=job_description_text,
                    sections={"full_content": job_description_text},
                    requirements=[]
                )
                
                # Initialize analyzer and run analysis
                analyzer = ResumeAnalyzer()
                result = analyzer.analyze_single_resume(tmp_file_path, job_desc)
                
                # Display results
                display_single_analysis_results(result)
                
                # Clean up temp file
                os.unlink(tmp_file_path)
                
            except Exception as e:
                st.error(f"Analysis failed: {str(e)}")
                if 'tmp_file_path' in locals():
                    try:
                        os.unlink(tmp_file_path)
                    except:
                        pass

def display_single_analysis_results(result):
    """Display the results of single resume analysis."""
    st.success("Analysis completed!")
    
    # Overall score display
    col1, col2, col3 = st.columns(3)
    
    with col1:
        score_pct = result.similarity_score.get_score_percentage()
        st.metric("Overall Match", f"{score_pct:.1f}%", f"Grade: {result.similarity_score.get_grade()}")
    
    with col2:
        st.metric("Confidence", f"{result.similarity_score.confidence:.1f}", "Analysis Quality")
    
    with col3:
        match_count = len(result.key_matches)
        st.metric("Key Matches", match_count, "Strong Points")
    
    # Detailed breakdown
    st.markdown("---")
    
    # Section scores
    st.subheader("Section-wise Analysis")
    
    if result.similarity_score.section_scores:
        for section, score in result.similarity_score.section_scores.items():
            score_pct = score * 100
            st.progress(score, text=f"{section.title()}: {score_pct:.1f}%")
    
    # Insights tabs
    tab1, tab2, tab3, tab4 = st.tabs(["Key Matches", "Missing Skills", "Recommendations", "AI Analysis"])
    
    with tab1:
        st.subheader("Strong Matching Points")
        if result.key_matches:
            for i, match in enumerate(result.key_matches, 1):
                st.write(f"{i}. {match}")
        else:
            st.info("No specific key matches identified.")
    
    with tab2:
        st.subheader("Areas for Improvement")
        if result.missing_skills:
            for i, skill in enumerate(result.missing_skills, 1):
                st.write(f"{i}. {skill}")
        else:
            st.success("No major skill gaps identified!")
    
    with tab3:
        st.subheader("Actionable Recommendations")
        if result.recommendations:
            for i, rec in enumerate(result.recommendations, 1):
                st.write(f"{i}. {rec}")
        else:
            st.info("No specific recommendations at this time.")
    
    with tab4:
        st.subheader("AI Analysis Reasoning")
        st.write(result.similarity_score.reasoning)
        
        # Show raw analysis data
        with st.expander("Detailed Analysis Data"):
            analysis_data = result.to_dict()
            st.json(analysis_data)
    
    # Download results
    st.markdown("---")
    if st.button("Download Analysis Report", use_container_width=True):
        # Create downloadable report
        report_data = result.to_dict()
        report_json = json.dumps(report_data, indent=2)
        
        st.download_button(
            label="Download JSON Report",
            data=report_json,
            file_name=f"resume_analysis_{result.resume_id[:8]}.json",
            mime="application/json"
        )