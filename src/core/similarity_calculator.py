import numpy as np
from typing import List, Dict, Tuple
from sklearn.metrics.pairwise import cosine_similarity
import logging

logger = logging.getLogger(__name__)

class SimilarityCalculator:
    """Calculates various similarity metrics for embeddings and text analysis."""
    
    def __init__(self):
        pass
    
    def cosine_similarity_score(self, embedding1: List[float], embedding2: List[float]) -> float:
        """Calculate cosine similarity between two embeddings."""
        try:
            # Convert to numpy arrays
            vec1 = np.array(embedding1).reshape(1, -1)
            vec2 = np.array(embedding2).reshape(1, -1)
            
            # Calculate cosine similarity
            similarity = cosine_similarity(vec1, vec2)[0][0]
            return float(similarity)
        
        except Exception as e:
            logger.error(f"Error calculating cosine similarity: {e}")
            return 0.0
    
    def batch_cosine_similarity(self, query_embedding: List[float], 
                               candidate_embeddings: List[List[float]]) -> List[float]:
        """Calculate cosine similarity between query and multiple candidate embeddings."""
        try:
            query_vec = np.array(query_embedding).reshape(1, -1)
            candidate_matrix = np.array(candidate_embeddings)
            
            similarities = cosine_similarity(query_vec, candidate_matrix)[0]
            return similarities.tolist()
        
        except Exception as e:
            logger.error(f"Error in batch cosine similarity calculation: {e}")
            return [0.0] * len(candidate_embeddings)
    
    def weighted_similarity_score(self, job_embedding: List[float], 
                                 resume_sections: Dict[str, List[float]], 
                                 section_weights: Dict[str, float] = None) -> Tuple[float, Dict[str, float]]:
        """Calculate weighted similarity score based on different resume sections."""
        
        if section_weights is None:
            section_weights = {
                'skills': 0.35,
                'experience': 0.30,
                'education': 0.25,
                'summary': 0.10
            }
        
        total_score = 0.0
        section_scores = {}
        
        try:
            for section, embedding in resume_sections.items():
                if section in section_weights and embedding:
                    raw_similarity = self.cosine_similarity_score(job_embedding, embedding)

                    enhanced_similarity = self.enhance_similarity_score(raw_similarity)
                    weighted_score = enhanced_similarity * section_weights[section]
                    total_score += weighted_score
                    section_scores[section] = enhanced_similarity
            
            final_score = self.apply_final_score_boost(total_score)
            
            return final_score, section_scores
        
        except Exception as e:
            logger.error(f"Error calculating weighted similarity: {e}")
            return 0.0, {}
        
    def enhance_similarity_score(self, raw_score: float) -> float:
        """Enhance similarity score to be more generous and realistic."""
        # Shift the scoring range to be more generous
        # Transform 0.2-0.8 range to 0.4-0.95 range
        
        if raw_score < 0.15:
            return raw_score * 2  # Very low scores remain low
        elif raw_score < 0.3:
            return 0.3 + (raw_score - 0.15) * 2  # Boost low-medium scores
        elif raw_score < 0.6:
            return 0.45 + (raw_score - 0.3) * 1.5  # Generous boost for medium scores
        else:
            return 0.9 + (raw_score - 0.6) * 0.25  # High scores get to near perfect
        
    def apply_final_score_boost(self, score: float) -> float:
        """Apply final boost to make scoring more realistic."""
        # Apply a square root transformation to boost lower scores more
        import math
        boosted = math.sqrt(score) * 0.85 + score * 0.15
        
        # Ensure minimum viable scores for decent matches
        if boosted > 0.25:
            boosted = boosted * 1.2  # 20% boost for any reasonable match
        
        return min(1.0, boosted)

    def normalize_scores(self, scores: List[float]) -> List[float]:
        """Normalize scores to 0-1 range using min-max normalization."""
        if not scores or len(scores) == 1:
            return scores
        
        min_score = min(scores)
        max_score = max(scores)
        
        if max_score == min_score:
            return [1.0] * len(scores)
        
        return [(score - min_score) / (max_score - min_score) for score in scores]
    
    def calculate_percentile_rank(self, score: float, all_scores: List[float]) -> float:
        """Calculate percentile rank of a score within a list of scores."""
        if not all_scores:
            return 0.0
        
        rank = sum(1 for s in all_scores if s <= score)
        return (rank / len(all_scores)) * 100
