from dataclasses import dataclass
from typing import List, Dict, Optional
from datetime import datetime

@dataclass
class ResumeDocument:
    """Model representing a resume document."""
    
    id: str
    file_path: str
    content: str
    sections: Dict[str, str]
    metadata: Dict
    embedding: Optional[List[float]] = None
    processed_at: Optional[datetime] = None
    
    def __post_init__(self):
        if self.processed_at is None:
            self.processed_at = datetime.now()
    
    def get_section_content(self, section: str) -> str:
        """Get content of a specific section."""
        return self.sections.get(section, "")
    
    def has_section(self, section: str) -> bool:
        """Check if resume has a specific section."""
        return section in self.sections and bool(self.sections[section].strip())
    
    def to_dict(self) -> Dict:
        """Convert to dictionary representation."""
        return {
            "id": self.id,
            "file_path": self.file_path,
            "content": self.content,
            "sections": self.sections,
            "metadata": self.metadata,
            "processed_at": self.processed_at.isoformat() if self.processed_at else None
        }

@dataclass
class ResumeCollection:
    """Model representing a collection of resumes."""
    
    resumes: List[ResumeDocument]
    created_at: datetime
    metadata: Dict
    
    def __post_init__(self):
        if not hasattr(self, 'created_at') or self.created_at is None:
            self.created_at = datetime.now()
    
    def add_resume(self, resume: ResumeDocument):
        """Add a resume to the collection."""
        self.resumes.append(resume)
    
    def get_resume_by_id(self, resume_id: str) -> Optional[ResumeDocument]:
        """Get resume by ID."""
        return next((r for r in self.resumes if r.id == resume_id), None)
    
    def get_resume_count(self) -> int:
        """Get total number of resumes."""
        return len(self.resumes)
    
    def to_dict(self) -> Dict:
        """Convert to dictionary representation."""
        return {
            "resumes": [r.to_dict() for r in self.resumes],
            "created_at": self.created_at.isoformat(),
            "metadata": self.metadata,
            "count": len(self.resumes)
        }
