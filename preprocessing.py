import re
import string
import unicodedata
from typing import List


class TextPreprocessor:
    """
    Simplified pipeline for text preprocessing.
    Includes only basic cleaning: ToLower, RemoveAccents, RemoveEscapeSequences, 
    RemoveEmojis, RemovePunctuation, RemoveBlankSpaces.
    """
    
    def __init__(self):
        """
        Initializes the simplified text preprocessor.
        """
        pass
        
    def to_lowercase(self, text: str) -> str:
        """Converts text to lowercase."""
        return text.lower()
    
    def remove_accents(self, text: str) -> str:
        """Removes accents and special characters."""
        return unicodedata.normalize('NFD', text).encode('ascii', 'ignore').decode('utf-8')
    
    def remove_escape_sequences(self, text: str) -> str:
        """Removes escape sequences like \n, \t, \r, etc."""
        return text.replace('\\n', ' ').replace('\\t', ' ').replace('\\r', ' ').replace('\\', ' ')
    
    def remove_emojis(self, text: str) -> str:
        """Removes emojis from text."""
        emoji_pattern = re.compile("["
                                   u"\U0001F600-\U0001F64F"  # emoticons
                                   u"\U0001F300-\U0001F5FF"  # symbols & pictographs
                                   u"\U0001F680-\U0001F6FF"  # transport & map symbols
                                   u"\U0001F1E0-\U0001F1FF"  # flags (iOS)
                                   u"\U00002702-\U000027B0"
                                   u"\U000024C2-\U0001F251"
                                   "]+", flags=re.UNICODE)
        return emoji_pattern.sub('', text)
    
    def remove_punctuation(self, text: str) -> str:
        """Removes punctuation from text."""
        return text.translate(str.maketrans('', '', string.punctuation))
    
    def remove_blank_spaces(self, text: str) -> str:
        """Removes extra blank spaces and normalizes spaces."""
        return re.sub(r'\s+', ' ', text).strip()
    
    def preprocess(self, text: str) -> str:
        """
        Applies the simplified preprocessing pipeline.
        """
        if not text or not isinstance(text, str):
            return ""
        
        # 1. ToLower
        text = self.to_lowercase(text)
        
        # 2. RemoveAccents
        text = self.remove_accents(text)
        
        # 3. RemoveEscapeSequences
        text = self.remove_escape_sequences(text)
        
        # 4. RemoveEmojis
        text = self.remove_emojis(text)
        
        # 5. RemovePunctuation
        text = self.remove_punctuation(text)
        
        # 6. RemoveBlankSpaces
        text = self.remove_blank_spaces(text)
        
        return text
    
    def preprocess_batch(self, texts: List[str]) -> List[str]:
        """
        Preprocesses a list of texts.
        """
        return [self.preprocess(text) for text in texts]

