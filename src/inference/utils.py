import re
import string
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
import spacy

# Download required NLTK data
nltk.download('punkt', quiet=True)
nltk.download('stopwords', quiet=True)
nltk.download('wordnet', quiet=True)

# Load spaCy model
try:
    nlp = spacy.load('en_core_web_sm')
except:
    print("Downloading spaCy model...")
    import os
    os.system('python -m spacy download en_core_web_sm')
    nlp = spacy.load('en_core_web_sm')

class TextCleaner:
    def __init__(self):
        self.stop_words = set(stopwords.words('english'))
        self.lemmatizer = WordNetLemmatizer()
    
    def clean_text(self, text):
        """Clean and preprocess text"""
        if not isinstance(text, str):
            return ""
        
        # Convert to lowercase
        text = text.lower()
        
        # Remove URLs
        text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE)
        
        # Remove email addresses
        text = re.sub(r'\S+@\S+', '', text)
        
        # Remove special characters and digits
        text = re.sub(r'[^a-zA-Z\s]', ' ', text)
        
        # Remove extra whitespaces
        text = re.sub(r'\s+', ' ', text).strip()
        
        return text
    
    def tokenize_and_lemmatize(self, text):
        """Tokenize and lemmatize text"""
        tokens = word_tokenize(text)
        
        # Remove stopwords and lemmatize
        tokens = [self.lemmatizer.lemmatize(token) 
                  for token in tokens 
                  if token not in self.stop_words and len(token) > 2]
        
        return ' '.join(tokens)
    
    def extract_skills(self, text):
        """Extract skills using NER and keyword matching"""
        doc = nlp(text)
        
        # Common technical skills keywords
        skills_keywords = [
            'python', 'java', 'javascript', 'c++', 'sql', 'html', 'css',
            'react', 'angular', 'node', 'django', 'flask', 'tensorflow',
            'pytorch', 'machine learning', 'deep learning', 'nlp', 'cv',
            'aws', 'azure', 'docker', 'kubernetes', 'git', 'agile'
        ]
        
        skills = []
        text_lower = text.lower()
        
        for skill in skills_keywords:
            if skill in text_lower:
                skills.append(skill)
        
        return list(set(skills))
    
    def extract_experience_years(self, text):
        """Extract years of experience"""
        # Pattern to match "X years" or "X+ years"
        pattern = r'(\d+)\+?\s*(?:years?|yrs?)'
        matches = re.findall(pattern, text.lower())
        
        if matches:
            return max([int(year) for year in matches])
        return 0
