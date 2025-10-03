from dataclasses import dataclass
from typing import List, Dict, Optional
from datetime import datetime

@dataclass
class JobDescription:
    """Model representing a job description."""
    
    id: str
    title: str
    content: str
    sections: Dict[str, str]
    requirements: List[str]
    embedding: Optional[List[float]] = None
    created_at: Optional[datetime] = None
    
    def __post_init__(self):
        if self.created_at is None:
            self.created_at = datetime.now()
    
    def get_section_content(self, section: str) -> str:
        """Get content of a specific section."""
        return self.sections.get(section, "")
    
    def extract_requirements(self) -> List[str]:
        """Extract key requirements from job description."""
        # This could be enhanced with NLP techniques
        req_section = self.get_section_content('requirements')
        if req_section:
            # Simple extraction by splitting on common delimiters
            requirements = [req.strip() for req in req_section.split('\n') if req.strip()]
            return [req for req in requirements if len(req) > 10]  # Filter short items
        return self.requirements
    
    def to_dict(self) -> Dict:
        """Convert to dictionary representation."""
        return {
            "id": self.id,
            "title": self.title,
            "content": self.content,
            "sections": self.sections,
            "requirements": self.requirements,
            "created_at": self.created_at.isoformat() if self.created_at else None
        }