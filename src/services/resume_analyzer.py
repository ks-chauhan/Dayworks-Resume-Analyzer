from typing import List, Dict, Optional, Tuple
import logging
from datetime import datetime
import uuid
import numpy as np

from src.core.document_processor import DocumentProcessor
from src.core.embedding_manager import EmbeddingManager
from src.core.similarity_calculator import SimilarityCalculator
from src.models.resume_model import ResumeDocument
from src.models.job_description_model import JobDescription
from src.models.analysis_result import SingleAnalysisResult, SimilarityScore

logger = logging.getLogger(__name__)

class ResumeAnalyzer:
    """Core service for analyzing individual resumes against job descriptions."""
    
    def __init__(self):
        self.document_processor = DocumentProcessor()
        self.embedding_manager = EmbeddingManager()
        self.similarity_calculator = SimilarityCalculator()
    
    def analyze_single_resume(self, resume_path: str, job_description: JobDescription) -> SingleAnalysisResult:
        """Perform comprehensive analysis of a single resume against a job description."""
        try:
            # Process resume
            resume_doc = self._process_resume_file(resume_path)
            
            # Generate embeddings for resume sections
            resume_section_embeddings = self._generate_section_embeddings(resume_doc)
            
            # Calculate similarity scores
            similarity_score = self._calculate_comprehensive_similarity(
                job_description, resume_doc, resume_section_embeddings
            )
            
            # Generate insights
            key_matches = self._identify_key_matches(resume_doc, job_description)
            missing_skills = self._identify_missing_skills(resume_doc, job_description)
            recommendations = self._generate_recommendations(similarity_score, missing_skills)
            
            return SingleAnalysisResult(
                resume_id=resume_doc.id,
                job_description_id=job_description.id,
                similarity_score=similarity_score,
                key_matches=key_matches,
                missing_skills=missing_skills,
                recommendations=recommendations,
                analysis_timestamp=datetime.now()
            )
        
        except Exception as e:
            logger.error(f"Error analyzing resume: {e}")
            raise
    
    def _process_resume_file(self, file_path: str) -> ResumeDocument:
        """Process resume file and extract structured information."""
        # Load document
        documents = self.document_processor.load_pdf(file_path)
        
        # Combine all content
        full_content = "\n".join([doc.page_content for doc in documents])
        
        # Extract sections
        sections = self.document_processor.extract_key_sections(full_content)
        
        # Create resume document
        resume_doc = ResumeDocument(
            id=str(uuid.uuid4()),
            file_path=file_path,
            content=full_content,
            sections=sections,
            metadata={"source": file_path, "type": "resume"}
        )
        
        return resume_doc
    
    def _generate_section_embeddings(self, resume_doc: ResumeDocument) -> Dict[str, List[float]]:
        """Generate embeddings for different resume sections."""
        section_embeddings = {}
        
        for section_name, section_content in resume_doc.sections.items():
            if section_content.strip():  # Only process non-empty sections
                embedding = self.embedding_manager.generate_single_embedding(section_content)
                section_embeddings[section_name] = embedding
        
        return section_embeddings
    
    def _calculate_comprehensive_similarity(self, job_desc: JobDescription, 
                                          resume_doc: ResumeDocument,
                                          resume_embeddings: Dict[str, List[float]]) -> SimilarityScore:
        """Calculate comprehensive similarity score using multiple factors."""
        
        # Generate job description embedding
        job_embedding = self.embedding_manager.generate_single_embedding(job_desc.content)
        
        # Calculate weighted similarity across sections
        section_weights = {
            'skills': 0.35,
            'experience': 0.30,
            'full_content': 0.25,
            'education': 0.10
        }
        
        overall_score, section_scores = self.similarity_calculator.weighted_similarity_score(
            job_embedding, resume_embeddings, section_weights
        )
        
        # Calculate confidence based on content availability and score distribution
        confidence = self._calculate_confidence(section_scores, resume_doc)
        
        # Generate reasoning
        reasoning = self._generate_reasoning(section_scores, overall_score)
        
        return SimilarityScore(
            overall_score=overall_score,
            section_scores=section_scores,
            confidence=confidence,
            reasoning=reasoning
        )
    
    def _calculate_confidence(self, section_scores: Dict[str, float], resume_doc: ResumeDocument) -> float:
        """Calculate confidence in the analysis based on available data."""
        # Base confidence on number of sections available
        available_sections = sum(1 for section in ['skills', 'experience', 'education'] 
                               if resume_doc.has_section(section))
        
        section_coverage = available_sections / 3.0  # Normalize to 0-1
        
        # Factor in score consistency (lower variance = higher confidence)
        if len(section_scores) > 1:
            scores = list(section_scores.values())
            score_variance = np.var(scores) if len(scores) > 1 else 0
            consistency_factor = max(0, 1 - score_variance)
        else:
            consistency_factor = 0.5
        
        confidence = (section_coverage * 0.6) + (consistency_factor * 0.4)
        return min(1.0, max(0.1, confidence))
    
    def _generate_reasoning(self, section_scores: Dict[str, float], overall_score: float) -> str:
        """Generate human-readable reasoning for the score."""
        reasoning_parts = []
        
        if overall_score >= 0.8:
            reasoning_parts.append("Excellent match with strong alignment across multiple areas.")
        elif overall_score >= 0.6:
            reasoning_parts.append("Good match with solid qualifications.")
        elif overall_score >= 0.4:
            reasoning_parts.append("Moderate match with some relevant qualifications.")
        else:
            reasoning_parts.append("Limited match with few relevant qualifications.")
        
        # Add section-specific insights
        if 'skills' in section_scores:
            skills_score = section_scores['skills']
            if skills_score >= 0.7:
                reasoning_parts.append("Strong technical skill alignment.")
            elif skills_score >= 0.4:
                reasoning_parts.append("Moderate technical skill match.")
            else:
                reasoning_parts.append("Limited technical skill overlap.")
        
        if 'experience' in section_scores:
            exp_score = section_scores['experience']
            if exp_score >= 0.7:
                reasoning_parts.append("Relevant work experience demonstrated.")
            elif exp_score >= 0.4:
                reasoning_parts.append("Some relevant work experience.")
            else:
                reasoning_parts.append("Limited relevant work experience.")
        
        return " ".join(reasoning_parts)
    
    def _identify_key_matches(self, resume_doc: ResumeDocument, job_desc: JobDescription) -> List[str]:
        """Identify key matching points between resume and job description."""
        matches = []
        
        # Simple keyword-based matching for demonstration
        # In a production system, this would use more sophisticated NLP
        job_content_lower = job_desc.content.lower()
        resume_content_lower = resume_doc.content.lower()
        
        # Common technical terms and skills
        common_terms = [
            'python', 'java', 'javascript', 'machine learning', 'data science',
            'sql', 'aws', 'docker', 'kubernetes', 'react', 'angular', 'node.js',
            'project management', 'agile', 'scrum', 'leadership', 'communication'
        ]
        
        for term in common_terms:
            if term in job_content_lower and term in resume_content_lower:
                matches.append(f"Relevant experience with {term}")
        
        return matches[:5]  # Return top 5 matches
    
    def _identify_missing_skills(self, resume_doc: ResumeDocument, job_desc: JobDescription) -> List[str]:
        """Identify skills mentioned in job description but not in resume."""
        missing = []
        
        job_content_lower = job_desc.content.lower()
        resume_content_lower = resume_doc.content.lower()
        
        # Extract potential skills from job description
        potential_skills = [
            'python', 'java', 'javascript', 'react', 'angular', 'vue.js',
            'machine learning', 'deep learning', 'tensorflow', 'pytorch',
            'aws', 'azure', 'gcp', 'docker', 'kubernetes', 'jenkins',
            'sql', 'mongodb', 'postgresql', 'redis', 'elasticsearch'
        ]
        
        for skill in potential_skills:
            if skill in job_content_lower and skill not in resume_content_lower:
                missing.append(skill.title())
        
        return missing[:5]  # Return top 5 missing skills
    
    def _generate_recommendations(self, similarity_score: SimilarityScore, missing_skills: List[str]) -> List[str]:
        """Generate actionable recommendations based on analysis."""
        recommendations = []
        
        overall_score = similarity_score.overall_score
        
        if overall_score < 0.6:
            recommendations.append("Consider gaining more relevant experience in the required domain.")
        
        if missing_skills:
            recommendations.append(f"Consider developing skills in: {', '.join(missing_skills[:3])}")
        
        if 'skills' in similarity_score.section_scores and similarity_score.section_scores['skills'] < 0.5:
            recommendations.append("Enhance the technical skills section with more relevant technologies.")
        
        if 'experience' in similarity_score.section_scores and similarity_score.section_scores['experience'] < 0.5:
            recommendations.append("Highlight more relevant work experience and achievements.")
        
        if overall_score >= 0.8:
            recommendations.append("Excellent candidate profile. Consider for immediate interview.")
        elif overall_score >= 0.6:
            recommendations.append("Good candidate profile. Suitable for further consideration.")
        
        return recommendations[:3]  # Return top 3 recommendations