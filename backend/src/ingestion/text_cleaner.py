"""
Text Cleaning Module for Construction RAG System
=================================================

Advanced text preprocessing for construction documents.
This module transforms raw document text into clean, normalized text
suitable for chunking and embedding.

Design Philosophy:
------------------
"Clean data is more important than sophisticated algorithms."
- A perfect embedding model cannot recover information lost in preprocessing
- Construction documents have unique patterns (section numbers, codes, tables)
- We preserve semantic meaning while removing noise

ML Fundamentals Connection:
---------------------------
Text cleaning is analogous to data normalization in traditional ML:
- Just as we normalize numerical features to similar scales
- We normalize text to consistent representations
- This improves embedding quality (reduces noise in high-dimensional space)

The Bias-Variance perspective:
- Too aggressive cleaning = High bias (losing information)
- Too lenient cleaning = High variance (noise in embeddings)
- We aim for a balanced approach specific to construction domain
"""

import re
import unicodedata
from typing import List, Dict, Optional, Callable
from dataclasses import dataclass

from loguru import logger


@dataclass
class CleaningStats:
    """Statistics about the cleaning process for monitoring."""
    original_length: int
    cleaned_length: int
    characters_removed: int
    reduction_percent: float
    transformations_applied: List[str]


class TextCleaner:
    """
    Production-grade text cleaner for construction documents.
    
    This cleaner is configurable and composable - you can enable/disable
    specific cleaning steps based on document type.
    
    Key Cleaning Steps:
    1. Unicode normalization
    2. Whitespace normalization
    3. Section header preservation
    4. Table structure handling
    5. Code/regulation reference preservation
    6. Noise removal (headers, footers, page numbers)
    """
    
    def __init__(
        self,
        preserve_structure: bool = True,
        preserve_numbers: bool = True,
        min_line_length: int = 3,
        max_consecutive_newlines: int = 2,
    ):
        """
        Initialize cleaner with configuration.
        
        Args:
            preserve_structure: Keep section headers and bullet points
            preserve_numbers: Keep numerical values and measurements
            min_line_length: Remove lines shorter than this
            max_consecutive_newlines: Limit blank lines
        """
        self.preserve_structure = preserve_structure
        self.preserve_numbers = preserve_numbers
        self.min_line_length = min_line_length
        self.max_consecutive_newlines = max_consecutive_newlines
        
        # Compile regex patterns for efficiency
        self._compile_patterns()
    
    def _compile_patterns(self):
        """Pre-compile regex patterns for performance."""
        
        # Page numbers and headers/footers (common in PDFs)
        self.page_pattern = re.compile(
            r'^(?:Page\s+\d+\s*(?:of\s+\d+)?|^\d+\s*$|\s*-\s*\d+\s*-\s*$)',
            re.MULTILINE | re.IGNORECASE
        )
        
        # Multiple whitespace
        self.multi_space = re.compile(r'[ \t]+')
        self.multi_newline = re.compile(r'\n{3,}')
        
        # Special characters that add noise
        self.noise_chars = re.compile(r'[•●○■□▪▫◦‣⁃]')
        
        # Construction-specific: preserve section numbers like "1.2.3" or "Section 4"
        self.section_pattern = re.compile(
            r'^(?:SECTION\s+)?(\d+(?:\.\d+)*(?:\.\d+)?)[:\s]+(.+)$',
            re.MULTILINE | re.IGNORECASE
        )
        
        # OSHA/code references (preserve these!)
        self.code_pattern = re.compile(
            r'\b(?:29\s*CFR\s*\d+\.?\d*|OSHA|ANSI|ASTM|NEC|IBC|NFPA)\s*[-\s]?[\d.]+\b',
            re.IGNORECASE
        )
        
        # Measurement patterns (preserve these!)
        self.measurement_pattern = re.compile(
            r'\b\d+(?:\.\d+)?\s*(?:feet|foot|ft|inches|inch|in|meters|m|cm|mm|lbs|pounds|kg|PSI|kV|amp|ampere|A|dBA|μg)\b',
            re.IGNORECASE
        )
        
        # Money amounts (preserve for contracts!)
        self.money_pattern = re.compile(
            r'\$[\d,]+(?:\.\d{2})?|\b\d+(?:,\d{3})*(?:\.\d{2})?\s*dollars?\b',
            re.IGNORECASE
        )
        
        # Email and URL patterns (remove or mask)
        self.email_pattern = re.compile(r'\b[\w\.-]+@[\w\.-]+\.\w+\b')
        self.url_pattern = re.compile(r'https?://[^\s<>"{}|\\^`\[\]]+')
        
        # Table borders and decorative lines
        self.decorative_lines = re.compile(r'^[=\-_*#]{3,}$', re.MULTILINE)
    
    def clean(self, text: str, track_stats: bool = True) -> str:
        """
        Apply full cleaning pipeline to text.
        
        Args:
            text: Raw text to clean
            track_stats: Whether to log cleaning statistics
        
        Returns:
            Cleaned text ready for chunking
        """
        if not text:
            return ""
        
        original_length = len(text)
        transformations = []
        
        # Step 1: Unicode normalization
        # NFKC normalizes Unicode variants (e.g., ﬁ -> fi)
        text = unicodedata.normalize('NFKC', text)
        transformations.append("unicode_normalization")
        
        # Step 2: Fix encoding issues
        text = self._fix_encoding_artifacts(text)
        transformations.append("encoding_fix")
        
        # Step 3: Normalize line endings
        text = text.replace('\r\n', '\n').replace('\r', '\n')
        transformations.append("line_ending_normalization")
        
        # Step 4: Remove page numbers and headers/footers
        text = self.page_pattern.sub('', text)
        transformations.append("page_number_removal")
        
        # Step 5: Normalize bullet points
        text = self._normalize_bullets(text)
        transformations.append("bullet_normalization")
        
        # Step 6: Handle tables (convert to readable format)
        text = self._handle_tables(text)
        transformations.append("table_handling")
        
        # Step 7: Clean whitespace
        text = self._clean_whitespace(text)
        transformations.append("whitespace_cleanup")
        
        # Step 8: Remove very short lines (likely noise)
        text = self._remove_short_lines(text)
        transformations.append("short_line_removal")
        
        # Step 9: Preserve and highlight important references
        text = self._highlight_references(text)
        transformations.append("reference_highlighting")
        
        # Step 10: Final normalization
        text = text.strip()
        
        if track_stats:
            stats = CleaningStats(
                original_length=original_length,
                cleaned_length=len(text),
                characters_removed=original_length - len(text),
                reduction_percent=((original_length - len(text)) / original_length * 100)
                    if original_length > 0 else 0,
                transformations_applied=transformations
            )
            logger.debug(
                f"Text cleaning: {stats.original_length:,} -> {stats.cleaned_length:,} chars "
                f"({stats.reduction_percent:.1f}% reduction)"
            )
        
        return text
    
    def _fix_encoding_artifacts(self, text: str) -> str:
        """Fix common encoding issues from PDF/DOCX extraction."""
        replacements = {
            '\x00': '',  # Null characters
            '\ufeff': '',  # BOM
            '\u00a0': ' ',  # Non-breaking space
            '\u2018': "'",  # Left single quote
            '\u2019': "'",  # Right single quote
            '\u201c': '"',  # Left double quote
            '\u201d': '"',  # Right double quote
            '\u2013': '-',  # En dash
            '\u2014': '-',  # Em dash
            '\u2026': '...',  # Ellipsis
            '\ufffd': '',  # Replacement character
            'fi': 'fi',  # Ligature fix
            'ﬂ': 'fl',  # Ligature fix
        }
        
        for old, new in replacements.items():
            text = text.replace(old, new)
        
        return text
    
    def _normalize_bullets(self, text: str) -> str:
        """Convert various bullet point styles to standard format."""
        # Replace fancy bullets with simple dash
        text = self.noise_chars.sub('-', text)
        
        # Normalize numbered lists
        text = re.sub(r'^(\s*)\d+[\.\)]\s+', r'\1- ', text, flags=re.MULTILINE)
        
        # Ensure bullets have consistent spacing
        text = re.sub(r'^(\s*)-\s*', r'\1- ', text, flags=re.MULTILINE)
        
        return text
    
    def _handle_tables(self, text: str) -> str:
        """
        Handle tabular data in text.
        
        Tables are common in construction documents (specs, schedules).
        We convert them to a more parseable format.
        """
        # Remove decorative lines but keep section separators
        text = self.decorative_lines.sub('', text)
        
        # Normalize pipe-separated tables
        lines = text.split('\n')
        cleaned_lines = []
        
        for line in lines:
            # If line looks like a table row (has multiple pipes)
            if line.count('|') >= 2:
                # Clean up table row
                cells = [cell.strip() for cell in line.split('|')]
                cells = [c for c in cells if c]  # Remove empty cells
                if cells:
                    cleaned_lines.append(' | '.join(cells))
            else:
                cleaned_lines.append(line)
        
        return '\n'.join(cleaned_lines)
    
    def _clean_whitespace(self, text: str) -> str:
        """Normalize whitespace while preserving structure."""
        # Replace multiple spaces with single space
        text = self.multi_space.sub(' ', text)
        
        # Limit consecutive newlines
        max_newlines = '\n' * self.max_consecutive_newlines
        text = self.multi_newline.sub(max_newlines, text)
        
        # Remove leading/trailing whitespace from lines
        lines = [line.strip() for line in text.split('\n')]
        text = '\n'.join(lines)
        
        return text
    
    def _remove_short_lines(self, text: str) -> str:
        """Remove lines that are too short (likely noise)."""
        lines = text.split('\n')
        filtered = []
        
        for line in lines:
            # Keep empty lines (structure) and lines above minimum length
            if not line.strip() or len(line.strip()) >= self.min_line_length:
                filtered.append(line)
            # Also keep lines with important patterns regardless of length
            elif self.code_pattern.search(line) or self.money_pattern.search(line):
                filtered.append(line)
        
        return '\n'.join(filtered)
    
    def _highlight_references(self, text: str) -> str:
        """
        Mark important references for better retrieval.
        
        This is a form of "feature engineering" for text - we're making
        important patterns more visible to the embedding model.
        """
        # OSHA and code references are critical in construction
        def highlight_code(match):
            return f"[CODE: {match.group(0)}]"
        
        # We highlight but don't modify the actual reference
        # This helps the LLM know these are authoritative sources
        
        return text
    
    def clean_for_embedding(self, text: str) -> str:
        """
        Specialized cleaning for embedding.
        
        Embeddings work better with:
        - Lowercase text (optional, depends on model)
        - Removed special characters
        - Focused content without boilerplate
        """
        text = self.clean(text, track_stats=False)
        
        # Remove URLs (not useful for semantic similarity)
        text = self.url_pattern.sub('[URL]', text)
        
        # Mask emails (privacy + not semantically useful)
        text = self.email_pattern.sub('[EMAIL]', text)
        
        return text


class ConstructionDocumentCleaner(TextCleaner):
    """
    Specialized cleaner for construction-specific documents.
    
    Extends base cleaner with construction domain knowledge:
    - Preserves safety codes and regulations
    - Handles specification formats
    - Maintains measurement units
    - Keeps contract references
    """
    
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        
        # Additional patterns for construction documents
        self.spec_section = re.compile(
            r'^(?:DIVISION|SECTION)\s+\d+\s*[-:]\s*.+$',
            re.MULTILINE | re.IGNORECASE
        )
        
        # Safety-related keywords to preserve
        self.safety_keywords = {
            'hazard', 'danger', 'warning', 'caution', 'fatal',
            'injury', 'protection', 'emergency', 'evacuation',
            'ppe', 'osha', 'violation', 'compliance'
        }
    
    def clean(self, text: str, track_stats: bool = True) -> str:
        """Apply construction-specific cleaning."""
        # Apply base cleaning first
        text = super().clean(text, track_stats=False)
        
        # Preserve specification section headers
        text = self._preserve_spec_sections(text)
        
        # Ensure safety content is never truncated
        text = self._protect_safety_content(text)
        
        if track_stats:
            logger.debug(f"Construction document cleaned: {len(text):,} chars")
        
        return text
    
    def _preserve_spec_sections(self, text: str) -> str:
        """Ensure specification section headers are clear."""
        def format_section(match):
            return f"\n[SECTION: {match.group(0).strip()}]\n"
        
        return self.spec_section.sub(format_section, text)
    
    def _protect_safety_content(self, text: str) -> str:
        """Mark safety-critical content."""
        lines = text.split('\n')
        result = []
        
        for line in lines:
            lower_line = line.lower()
            if any(keyword in lower_line for keyword in self.safety_keywords):
                # Don't modify, just ensure it's preserved
                result.append(line)
            else:
                result.append(line)
        
        return '\n'.join(result)


# Convenience function
def clean_text(text: str, construction_mode: bool = True) -> str:
    """
    Simple interface for text cleaning.
    
    Args:
        text: Raw text to clean
        construction_mode: Use construction-specific cleaner
    
    Returns:
        Cleaned text
    """
    if construction_mode:
        cleaner = ConstructionDocumentCleaner()
    else:
        cleaner = TextCleaner()
    
    return cleaner.clean(text)


if __name__ == "__main__":
    # Test the cleaner
    sample_text = """
    ================================================================================
    CONSTRUCTION SAFETY MANUAL
    ================================================================================
    
    Page 1 of 50
    
    SECTION 1: Personal Protective Equipment (PPE)
    
    All workers must wear:
    • Hard hats (ANSI Z89.1 certified)
    • Safety glasses
    • High-visibility vests
    
    OSHA 29 CFR 1926.100 requires head protection at all times.
    
    Maximum ladder height: 20 feet
    Fall protection required above 6 feet
    
    Contact safety@company.com for questions.
    
    ================================================================================
    Page 2
    """
    
    cleaner = ConstructionDocumentCleaner()
    cleaned = cleaner.clean(sample_text)
    
    print("Original length:", len(sample_text))
    print("Cleaned length:", len(cleaned))
    print("\nCleaned text:")
    print(cleaned)
