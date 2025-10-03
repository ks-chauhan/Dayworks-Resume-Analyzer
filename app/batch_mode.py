import streamlit as st
import tempfile
import os
from pathlib import Path
import json
import uuid
import zipfile
from typing import List

from src.services.batch_processor import BatchProcessor
from src.models.job_description_model import JobDescription

def render_batch_mode():
    """Render the batch processing interface."""
    st.header("Batch Resume Shortlisting")
    st.markdown("Upload multiple resumes to get a ranked shortlist of top candidates.")
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.subheader("Job Description")
        
        # Job title input
        job_title = st.text_input("Position Title", placeholder="e.g., Data Scientist")
        
        # Job description input
        job_description_text = st.text_area(
            "Complete Job Description",
            height=250,
            placeholder="Paste the complete job description including requirements, responsibilities, and qualifications..."
        )
    
    with col2:
        st.subheader("Analysis Settings")
        
        # Top N selection
        top_n = st.number_input(
            "Number of top candidates to shortlist",
            min_value=1,
            max_value=50,
            value=10,
            help="Select how many top-ranked candidates you want in the results"
        )
        
        # Processing options
        st.checkbox("Include detailed section analysis", value=True)
        st.checkbox("Generate candidate summaries", value=True)
    
    # Resume upload section
    st.markdown("---")
    st.subheader("Resume Upload")
    
    upload_method = st.radio(
        "Choose upload method:",
        ["Multiple Files", "ZIP Archive"],
        horizontal=True
    )
    
    uploaded_files = []
    
    if upload_method == "Multiple Files":
        uploaded_files = st.file_uploader(
            "Upload Resume Files",
            type=['pdf', 'txt'],
            accept_multiple_files=True,
            help="Select multiple resume files (PDF or text format)"
        )
        
        if uploaded_files:
            st.success(f"{len(uploaded_files)} files uploaded")
            
            # Show file list
            with st.expander("Uploaded Files"):
                for i, file in enumerate(uploaded_files, 1):
                    file_size = len(file.getvalue()) / 1024
                    st.write(f"{i}. {file.name} ({file_size:.1f} KB)")
    
    else:
        zip_file = st.file_uploader(
            "Upload ZIP Archive",
            type=['zip'],
            help="Upload a ZIP file containing multiple resume files"
        )
        
        if zip_file:
            # Extract files from ZIP
            with tempfile.TemporaryDirectory() as temp_dir:
                zip_path = os.path.join(temp_dir, "resumes.zip")
                with open(zip_path, "wb") as f:
                    f.write(zip_file.getvalue())
                
                try:
                    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
                        file_list = [name for name in zip_ref.namelist() 
                                   if name.lower().endswith(('.pdf', '.txt')) and not name.startswith('__MACOSX')]
                        
                        st.success(f"ZIP file contains {len(file_list)} resume files")
                        
                        # Show file list
                        with st.expander("Files in ZIP"):
                            for i, filename in enumerate(file_list, 1):
                                st.write(f"{i}. {filename}")
                        
                        # Store file info for processing
                        st.session_state['zip_file'] = zip_file
                        st.session_state['zip_files'] = file_list
                
                except Exception as e:
                    st.error(f"Error reading ZIP file: {str(e)}")
    
    # Analysis button
    st.markdown("---")
    
    if st.button("Start Batch Analysis", type="primary", use_container_width=True):
        if not job_description_text.strip():
            st.error("Please provide a job description")
            return
        
        # Check if we have files to process
        files_to_process = []
        
        if upload_method == "Multiple Files":
            if not uploaded_files:
                st.error("Please upload resume files")
                return
            files_to_process = uploaded_files
        else:
            if 'zip_file' not in st.session_state:
                st.error("Please upload a ZIP file")
                return
            # We'll handle ZIP extraction in the processing function
        
        # Show progress
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        try:
            # Create job description object
            job_desc = JobDescription(
                id=str(uuid.uuid4()),
                title=job_title or "Batch Analysis",
                content=job_description_text,
                sections={"full_content": job_description_text},
                requirements=[]
            )
            
            # Process files
            temp_file_paths = []
            
            status_text.text("Preparing files for analysis...")
            progress_bar.progress(10)
            
            if upload_method == "Multiple Files":
                # Save uploaded files temporarily
                for i, uploaded_file in enumerate(files_to_process):
                    with tempfile.NamedTemporaryFile(delete=False, suffix=f".{uploaded_file.name.split('.')[-1]}") as tmp_file:
                        tmp_file.write(uploaded_file.getvalue())
                        temp_file_paths.append(tmp_file.name)
            else:
                # Extract ZIP files
                temp_file_paths = extract_zip_files(st.session_state['zip_file'], st.session_state['zip_files'])
            
            progress_bar.progress(25)
            status_text.text("Running AI analysis on all resumes...")
            
            # Initialize batch processor and run analysis
            processor = BatchProcessor()
            result = processor.process_batch_resumes(temp_file_paths, job_desc, top_n)
            
            progress_bar.progress(100)
            status_text.text("Analysis completed!")
            
            # Display results
            display_batch_results(result)
            
            # Clean up temp files
            for temp_path in temp_file_paths:
                try:
                    os.unlink(temp_path)
                except:
                    pass
        
        except Exception as e:
            st.error(f"Batch analysis failed: {str(e)}")
            # Clean up on error
            if 'temp_file_paths' in locals():
                for temp_path in temp_file_paths:
                    try:
                        os.unlink(temp_path)
                    except:
                        pass

def extract_zip_files(zip_file, file_list: List[str]) -> List[str]:
    """Extract files from ZIP and return temp file paths."""
    temp_paths = []
    
    with tempfile.TemporaryDirectory() as temp_dir:
        zip_path = os.path.join(temp_dir, "resumes.zip")
        with open(zip_path, "wb") as f:
            f.write(zip_file.getvalue())
        
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            for filename in file_list:
                # Extract to temp directory
                zip_ref.extract(filename, temp_dir)
                extracted_path = os.path.join(temp_dir, filename)
                
                # Create a permanent temp file
                with tempfile.NamedTemporaryFile(delete=False, suffix=f".{filename.split('.')[-1]}") as tmp_file:
                    with open(extracted_path, 'rb') as src_file:
                        tmp_file.write(src_file.read())
                    temp_paths.append(tmp_file.name)
    
    return temp_paths

def display_batch_results(result):
    """Display the results of batch analysis."""
    st.success("Batch analysis completed successfully!")
    
    # Summary metrics
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Total Candidates", result.total_candidates)
    
    with col2:
        st.metric("Processed", result.analysis_summary.get('total_processed', 0))
    
    with col3:
        avg_score = result.analysis_summary.get('average_score', 0) * 100
        st.metric("Average Score", f"{avg_score:.1f}%")
    
    with col4:
        max_score = result.analysis_summary.get('max_score', 0) * 100
        st.metric("Top Score", f"{max_score:.1f}%")
    
    # Score distribution
    st.markdown("---")
    st.subheader("Score Distribution")
    
    if 'score_distribution' in result.analysis_summary:
        dist = result.analysis_summary['score_distribution']
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Excellent (>80%)", dist.get('excellent (>0.8)', 0), "")
        with col2:
            st.metric("Good (60-80%)", dist.get('good (0.6-0.8)', 0), "")
        with col3:
            st.metric("Fair (40-60%)", dist.get('fair (0.4-0.6)', 0), "")
        with col4:
            st.metric("Poor (<40%)", dist.get('poor (<0.4)', 0), "")
    
    # Top candidates ranking
    st.markdown("---")
    st.subheader(f"Top {len(result.rankings)} Candidates")
    
    for i, ranking in enumerate(result.rankings):
        with st.expander(f"#{ranking.rank} - {ranking.candidate_name or f'Candidate {ranking.rank}'} ({ranking.similarity_score.get_score_percentage():.1f}%)"):
            
            col1, col2 = st.columns([2, 1])
            
            with col1:
                st.write("**Key Highlights:**")
                for highlight in ranking.key_highlights:
                    st.write(f"• {highlight}")
                
                if ranking.similarity_score.section_scores:
                    st.write("**Section Scores:**")
                    for section, score in ranking.similarity_score.section_scores.items():
                        st.progress(score, text=f"{section.title()}: {score*100:.1f}%")
            
            with col2:
                score_pct = ranking.similarity_score.get_score_percentage()
                grade = ranking.similarity_score.get_grade()
                
                st.metric("Match Score", f"{score_pct:.1f}%", f"Grade: {grade}")
                st.write(f"**Resume ID:** `{ranking.resume_id[:8]}...`")
    
    # Download results
    st.markdown("---")
    col1, col2 = st.columns(2)
    
    with col1:
        if st.button("📥 Download Detailed Report", use_container_width=True):
            # Create comprehensive report
            report_data = result.to_dict()
            report_json = json.dumps(report_data, indent=2)
            
            st.download_button(
                label="Download JSON Report",
                data=report_json,
                file_name=f"batch_analysis_report_{result.job_description_id[:8]}.json",
                mime="application/json"
            )
    
    with col2:
        if st.button("Download Candidate Summary", use_container_width=True):
            # Create simplified CSV-like summary
            summary_lines = ["Rank,Candidate,Score,Grade,Key Highlights"]
            
            for ranking in result.rankings:
                name = ranking.candidate_name or f"Candidate {ranking.rank}"
                score = f"{ranking.similarity_score.get_score_percentage():.1f}%"
                grade = ranking.similarity_score.get_grade()
                highlights = "; ".join(ranking.key_highlights[:2])  # First 2 highlights
                
                summary_lines.append(f"{ranking.rank},{name},{score},{grade},\"{highlights}\"")
            
            summary_csv = "\n".join(summary_lines)
            
            st.download_button(
                label="Download CSV Summary",
                data=summary_csv,
                file_name=f"candidate_summary_{result.job_description_id[:8]}.csv",
                mime="text/csv"
            )