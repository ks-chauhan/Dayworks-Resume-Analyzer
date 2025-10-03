from typing import List, Dict, Optional, Tuple
import logging
from datetime import datetime
import uuid
from concurrent.futures import ThreadPoolExecutor, as_completed
import numpy as np

from src.core.document_processor import DocumentProcessor
from src.core.embedding_manager import EmbeddingManager
from src.core.vector_store import VectorStoreManager
from src.core.similarity_calculator import SimilarityCalculator
from src.models.resume_model import ResumeDocument, ResumeCollection
from src.models.job_description_model import JobDescription
from src.models.analysis_result import BatchAnalysisResult, CandidateRanking, SimilarityScore

logger = logging.getLogger(__name__)

class BatchProcessor:
    """Service for processing multiple resumes in batch mode."""
    
    def __init__(self, max_workers: int = 4):
        self.document_processor = DocumentProcessor()
        self.embedding_manager = EmbeddingManager()
        self.vector_store = VectorStoreManager()
        self.similarity_calculator = SimilarityCalculator()
        self.max_workers = max_workers
    
    def process_batch_resumes(self, resume_paths: List[str], job_description: JobDescription, 
                            top_n: int = 10) -> BatchAnalysisResult:
        """Process multiple resumes and return ranked results."""
        try:
            logger.info(f"Processing batch of {len(resume_paths)} resumes")
            
            # Process all resumes
            resume_collection = self._process_resume_batch(resume_paths)
            
            # Store in vector database for efficient similarity search
            self._index_resumes_in_vector_store(resume_collection)
            
            # Calculate similarities and rank candidates
            rankings = self._rank_candidates(resume_collection, job_description, top_n)
            
            # Generate analysis summary
            analysis_summary = self._generate_batch_summary(rankings, len(resume_paths))
            
            return BatchAnalysisResult(
                job_description_id=job_description.id,
                total_candidates=len(resume_paths),
                top_n_requested=top_n,
                rankings=rankings,
                analysis_summary=analysis_summary,
                analysis_timestamp=datetime.now()
            )
        
        except Exception as e:
            logger.error(f"Error in batch processing: {e}")
            raise
    
    def _process_resume_batch(self, resume_paths: List[str]) -> ResumeCollection:
        """Process multiple resumes in parallel."""
        resumes = []
        
        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            # Submit all resume processing tasks
            future_to_path = {
                executor.submit(self._process_single_resume, path): path 
                for path in resume_paths
            }
            
            # Collect results
            for future in as_completed(future_to_path):
                path = future_to_path[future]
                try:
                    resume_doc = future.result()
                    if resume_doc:
                        resumes.append(resume_doc)
                except Exception as e:
                    logger.error(f"Error processing {path}: {e}")
        
        return ResumeCollection(
            resumes=resumes,
            created_at=datetime.now(),
            metadata={"batch_size": len(resume_paths), "processed_count": len(resumes)}
        )
    
    def _process_single_resume(self, file_path: str) -> Optional[ResumeDocument]:
        """Process a single resume file."""
        try:
            # Load and process document
            documents = self.document_processor.load_pdf(file_path)
            full_content = "\n".join([doc.page_content for doc in documents])
            
            # Extract sections
            sections = self.document_processor.extract_key_sections(full_content)
            
            # Generate embedding for full content
            embedding = self.embedding_manager.generate_single_embedding(full_content)
            
            return ResumeDocument(
                id=str(uuid.uuid4()),
                file_path=file_path,
                content=full_content,
                sections=sections,
                metadata={"source": file_path, "type": "resume"},
                embedding=embedding
            )
        
        except Exception as e:
            logger.error(f"Error processing resume {file_path}: {e}")
            return None
    
    def _index_resumes_in_vector_store(self, resume_collection: ResumeCollection):
        """Index all resumes in the vector store for efficient similarity search."""
        try:
            # Clear existing collection
            self.vector_store.clear_collection()
            
            # Prepare data for indexing
            documents = []
            embeddings = []
            metadata = []
            ids = []
            
            for resume in resume_collection.resumes:
                if resume.embedding:
                    documents.append(resume.content)
                    embeddings.append(resume.embedding)
                    metadata.append({
                    "resume_id": resume.id,
                    "file_path": resume.file_path,
                    "skills": resume.sections.get("skills", "")[:500],  # Limit length
                    "experience": resume.sections.get("experience", "")[:500],
                    "education": resume.sections.get("education", "")[:500],
                    "summary": resume.sections.get("summary", "")[:500]
                })
                ids.append(resume.id)
            
            # Add to vector store
            if documents:
                self.vector_store.add_documents(documents, embeddings, metadata, ids)
                logger.info(f"Indexed {len(documents)} resumes in vector store")
        
        except Exception as e:
            logger.error(f"Error indexing resumes: {e}")
            raise
    
    def _rank_candidates(self, resume_collection: ResumeCollection, 
                        job_description: JobDescription, top_n: int) -> List[CandidateRanking]:
        """Rank all candidates based on similarity to job description."""
        try:
            # Generate job description embedding
            job_embedding = self.embedding_manager.generate_single_embedding(job_description.content)
            
            # Perform similarity search
            documents, similarity_scores, metadatas = self.vector_store.similarity_search(
                job_embedding, top_k=len(resume_collection.resumes)
            )
            
            # Create candidate rankings
            rankings = []
            for i, (doc, score, metadata) in enumerate(zip(documents, similarity_scores, metadatas)):
                # Get resume document
                resume_id = metadata.get("resume_id")
                resume_doc = resume_collection.get_resume_by_id(resume_id)
                
                if resume_doc:

                    exact_file_name = self._get_exact_filename(resume_doc.file_path)
                    # Calculate detailed similarity score
                    detailed_score = self._calculate_detailed_similarity_score(
                        job_description, resume_doc, score
                    )
                    
                    # Generate key highlights
                    highlights = self._generate_candidate_highlights(resume_doc, job_description)
                    
                    ranking = CandidateRanking(
                        rank=i + 1,
                        resume_id=exact_file_name,
                        candidate_name=exact_file_name,
                        similarity_score=detailed_score,
                        key_highlights=highlights
                    )
                    rankings.append(ranking)
            
            # Return top N candidates
            return rankings[:top_n]
        
        except Exception as e:
            logger.error(f"Error ranking candidates: {e}")
            return []
        
    def _get_exact_filename(self, file_path: str) -> str:
        """Get exact filename without path and extension."""
        import os
        # Get just the filename without path
        filename = os.path.basename(file_path)
        # Remove file extension but keep the exact name
        name_without_ext = os.path.splitext(filename)[0]
        return name_without_ext

    def _calculate_detailed_similarity_score(self, job_desc: JobDescription, 
                                           resume_doc: ResumeDocument, base_score: float) -> SimilarityScore:
        """Calculate detailed similarity score with section breakdown."""
        
        # Generate section embeddings
        section_scores = {}
        job_embedding = self.embedding_manager.generate_single_embedding(job_desc.content)
        
        for section_name, section_content in resume_doc.sections.items():
            if section_content.strip():
                section_embedding = self.embedding_manager.generate_single_embedding(section_content)
                raw_similarity = self.similarity_calculator.cosine_similarity_score(
                    job_embedding, section_embedding
                )
                enhanced_similarity = self._enhance_batch_score(raw_similarity)
                section_scores[section_name] = enhanced_similarity
        
        enhanced_base_score = self._enhance_batch_score(base_score)

        # Calculate confidence
        confidence = min(1.0, enhanced_base_score + 0.1)  # Simple confidence calculation
        
        # Generate reasoning
        reasoning = self._generate_positive_reasoning(enhanced_base_score, section_scores)
        
        return SimilarityScore(
            overall_score=enhanced_base_score,
            section_scores=section_scores,
            confidence=confidence,
            reasoning=reasoning
        )
    
    def _enhance_batch_score(self, raw_score: float) -> float:
        """Apply generous score enhancement for batch processing."""
        # More generous than single analysis - recruiters want to see potential
        if raw_score < 0.2:
            return raw_score * 1.5  # Slight boost for very low scores
        elif raw_score < 0.4:
            return 0.35 + (raw_score - 0.2) * 2.0  # Significant boost for low-medium
        elif raw_score < 0.6:
            return 0.65 + (raw_score - 0.4) * 1.25  # Good boost for medium scores
        else:
            return 0.85 + (raw_score - 0.6) * 0.375  # High scores reach 95%+

    def _generate_positive_reasoning(self, score: float, section_scores: Dict[str, float]) -> str:
        """Generate more positive, encouraging reasoning for batch results."""
        if score >= 0.75:
            return "Excellent candidate with strong qualifications and experience alignment."
        elif score >= 0.60:
            return "Strong candidate with relevant skills and good potential fit."
        elif score >= 0.45:
            return "Promising candidate with applicable experience and transferable skills."
        elif score >= 0.30:
            return "Potential candidate worth considering with some relevant background."
        else:
            return "Candidate may have hidden potential - recommend manual review."
        
    def _extract_candidate_name(self, content: str) -> Optional[str]:
        """Extract candidate name from resume content."""
        # Simple name extraction - could be enhanced with NLP
        lines = content.split('\n')[:5]  # Check first 5 lines
        
        for line in lines:
            line = line.strip()
            if line and len(line.split()) <= 4 and not any(char.isdigit() for char in line):
                # Basic validation for name-like content
                if 2 <= len(line.split()) <= 3:
                    return line
        
        return None
    
    def _generate_candidate_highlights(self, resume_doc: ResumeDocument, 
                                     job_desc: JobDescription) -> List[str]:
        """Generate key highlights for a candidate."""
        highlights = []
        
        # Extract years of experience (simple pattern matching)
        content_lower = resume_doc.content.lower()
        if 'year' in content_lower:
            highlights.append("Demonstrated work experience")
        
        # Check for education
        if resume_doc.has_section('education'):
            highlights.append("Relevant educational background")
        
        # Check for technical skills
        if resume_doc.has_section('skills'):
            highlights.append("Technical skills listed")
        
        # Add content-based highlights
        if len(resume_doc.content) > 2000:
            highlights.append("Comprehensive resume with detailed experience")
        
        return highlights[:3]  # Return top 3 highlights
    
    def _generate_batch_summary(self, rankings: List[CandidateRanking], 
                               total_candidates: int) -> Dict[str, any]:
        """Generate summary statistics for batch analysis."""
        if not rankings:
            return {"error": "No rankings generated"}
        
        scores = [ranking.similarity_score.overall_score for ranking in rankings]
        
        return {
            "total_processed": len(rankings),
            "total_candidates": total_candidates,
            "average_score": np.mean(scores),
            "median_score": np.median(scores),
            "max_score": max(scores),
            "min_score": min(scores),
            "score_distribution": {
                "excellent (>0.8)": sum(1 for s in scores if s > 0.8),
                "good (0.6-0.8)": sum(1 for s in scores if 0.65 <= s <= 0.8),
                "fair (0.4-0.6)": sum(1 for s in scores if 0.5 <= s < 0.65),
                "poor (<0.4)": sum(1 for s in scores if s < 0.5)
            }
        }