from typing import List, Dict, Optional
import logging
from pathlib import Path
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.schema import Document

logger = logging.getLogger(__name__)

class DocumentProcessor:
    """Processes and extracts text from various document formats."""
    
    def __init__(self, chunk_size: int = 1000, chunk_overlap: int = 200):
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            separators=["\n\n", "\n", " ", ""]
        )
    
    def load_pdf(self, file_path: str) -> List[Document]:
        """Load and process PDF documents."""
        try:
            loader = PyPDFLoader(file_path)
            documents = loader.load()
            return self._process_documents(documents)
        except Exception as e:
            logger.error(f"Error loading PDF {file_path}: {e}")
            raise
    
    def load_text_file(self, file_path: str) -> List[Document]:
        """Load and process text files."""
        try:
            with open(file_path, 'r', encoding='utf-8') as file:
                content = file.read()
            
            doc = Document(
                page_content=content,
                metadata={"source": file_path, "type": "text"}
            )
            return self._process_documents([doc])
        except Exception as e:
            logger.error(f"Error loading text file {file_path}: {e}")
            raise
    
    def process_text_content(self, content: str, metadata: Optional[Dict] = None) -> List[Document]:
        """Process raw text content."""
        doc = Document(
            page_content=content,
            metadata=metadata or {"type": "raw_text"}
        )
        return self._process_documents([doc])
    
    def _process_documents(self, documents: List[Document]) -> List[Document]:
        """Split documents into chunks and clean text."""
        processed_docs = []
        
        for doc in documents:
            # Clean and preprocess text
            cleaned_content = self._clean_text(doc.page_content)
            doc.page_content = cleaned_content
            
            # Split into chunks if content is large
            if len(cleaned_content) > 1000:
                chunks = self.text_splitter.split_documents([doc])
                processed_docs.extend(chunks)
            else:
                processed_docs.append(doc)
        
        return processed_docs
    
    def _clean_text(self, text: str) -> str:
        """Clean and normalize text content."""
        # Remove extra whitespace and normalize
        text = ' '.join(text.split())
        # Remove special characters that might interfere with processing
        text = text.replace('\x00', '')  # Remove null characters
        return text.strip()
    
    def extract_key_sections(self, content: str) -> Dict[str, str]:
        """Extract key sections from resume content using pattern matching."""
        sections = {
            'skills': '',
            'experience': '',
            'education': '',
            'summary': '',
            'full_content': content
        }
        
        # Simple section extraction based on common patterns
        content_lower = content.lower()
        
        # Extract skills section
        skills_patterns = ['skills', 'technical skills', 'competencies', 'technologies']
        sections['skills'] = self._extract_section_by_patterns(content, skills_patterns)
        
        # Extract experience section
        exp_patterns = ['experience', 'work experience', 'employment', 'professional experience']
        sections['experience'] = self._extract_section_by_patterns(content, exp_patterns)
        
        # Extract education section
        edu_patterns = ['education', 'academic', 'qualifications', 'degrees']
        sections['education'] = self._extract_section_by_patterns(content, edu_patterns)
        
        return sections
    
    def _extract_section_by_patterns(self, content: str, patterns: List[str]) -> str:
        """Extract text section based on header patterns."""
        content_lines = content.split('\n')
        section_content = []
        in_section = False
        
        for line in content_lines:
            line_lower = line.lower().strip()
            
            # Check if line contains any of the section patterns
            if any(pattern in line_lower for pattern in patterns):
                in_section = True
                continue
            
            # Check if we've hit another section (common section headers)
            common_sections = ['summary', 'objective', 'skills', 'experience', 'education', 
                             'projects', 'certifications', 'awards', 'references']
            if in_section and any(section in line_lower for section in common_sections):
                if not any(pattern in line_lower for pattern in patterns):
                    break
            
            if in_section and line.strip():
                section_content.append(line)
        
        return '\n'.join(section_content).strip()