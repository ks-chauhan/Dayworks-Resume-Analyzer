from dataclasses import dataclass
from typing import List, Dict, Optional
from datetime import datetime

@dataclass
class SimilarityScore:
    """Model for similarity score details."""
    
    overall_score: float
    section_scores: Dict[str, float]
    confidence: float
    reasoning: str
    
    def get_score_percentage(self) -> float:
        """Get score as percentage."""
        return round(self.overall_score * 100, 2)
    
    def get_grade(self) -> str:
        """Get letter grade based on score."""
        score_pct = self.get_score_percentage()
        if score_pct >= 80:
            return "A"
        elif score_pct >= 70:
            return "B"
        elif score_pct >= 60:
            return "C"
        elif score_pct >= 50:
            return "D"
        else:
            return "F"

@dataclass
class SingleAnalysisResult:
    """Model for single resume analysis result."""
    
    resume_id: str
    job_description_id: str
    similarity_score: SimilarityScore
    key_matches: List[str]
    missing_skills: List[str]
    recommendations: List[str]
    analysis_timestamp: datetime
    
    def __post_init__(self):
        if not hasattr(self, 'analysis_timestamp') or self.analysis_timestamp is None:
            self.analysis_timestamp = datetime.now()
    
    def to_dict(self) -> Dict:
        """Convert to dictionary representation."""
        return {
            "resume_id": self.resume_id,
            "job_description_id": self.job_description_id,
            "similarity_score": {
                "overall_score": self.similarity_score.overall_score,
                "section_scores": self.similarity_score.section_scores,
                "confidence": self.similarity_score.confidence,
                "reasoning": self.similarity_score.reasoning,
                "percentage": self.similarity_score.get_score_percentage(),
                "grade": self.similarity_score.get_grade()
            },
            "key_matches": self.key_matches,
            "missing_skills": self.missing_skills,
            "recommendations": self.recommendations,
            "analysis_timestamp": self.analysis_timestamp.isoformat()
        }

@dataclass
class CandidateRanking:
    """Model for candidate ranking in batch mode."""
    
    rank: int
    resume_id: str
    candidate_name: Optional[str]
    similarity_score: SimilarityScore
    key_highlights: List[str]
    
    def to_dict(self) -> Dict:
        """Convert to dictionary representation."""
        return {
            "rank": self.rank,
            "resume_id": self.resume_id,
            "candidate_name": self.candidate_name,
            "similarity_score": {
                "overall_score": self.similarity_score.overall_score,
                "percentage": self.similarity_score.get_score_percentage(),
                "grade": self.similarity_score.get_grade()
            },
            "key_highlights": self.key_highlights
        }

@dataclass
class BatchAnalysisResult:
    """Model for batch analysis result."""
    
    job_description_id: str
    total_candidates: int
    top_n_requested: int
    rankings: List[CandidateRanking]
    analysis_summary: Dict[str, any]
    analysis_timestamp: datetime
    
    def __post_init__(self):
        if not hasattr(self, 'analysis_timestamp') or self.analysis_timestamp is None:
            self.analysis_timestamp = datetime.now()
    
    def get_top_candidates(self, n: int) -> List[CandidateRanking]:
        """Get top N candidates."""
        return self.rankings[:min(n, len(self.rankings))]
    
    def to_dict(self) -> Dict:
        """Convert to dictionary representation."""
        return {
            "job_description_id": self.job_description_id,
            "total_candidates": self.total_candidates,
            "top_n_requested": self.top_n_requested,
            "rankings": [ranking.to_dict() for ranking in self.rankings],
            "analysis_summary": self.analysis_summary,
            "analysis_timestamp": self.analysis_timestamp.isoformat()
        }