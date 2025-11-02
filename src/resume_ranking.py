import numpy as np
import pandas as pd
from src.feature_extraction import FeatureExtractor
from src.utils import TextCleaner
from src.config import Config

class ResumeRanker:
    def __init__(self):
        self.feature_extractor = FeatureExtractor()
        self.cleaner = TextCleaner()
    
    def calculate_keyword_match_score(self, resume_text, job_description):
        """Calculate keyword matching score"""
        resume_keywords = set(resume_text.lower().split())
        job_keywords = set(job_description.lower().split())
        
        if len(job_keywords) == 0:
            return 0.0
        
        common_keywords = resume_keywords.intersection(job_keywords)
        score = len(common_keywords) / len(job_keywords)
        
        return min(score, 1.0)
    
    def calculate_experience_score(self, resume_experience, required_experience):
        """Calculate experience matching score"""
        if required_experience == 0:
            return 1.0
        
        if resume_experience >= required_experience:
            return 1.0
        else:
            return resume_experience / required_experience
    
    def calculate_education_score(self, resume_text, required_education):
        """Calculate education matching score"""
        resume_lower = resume_text.lower()
        required_lower = required_education.lower()
        
        education_keywords = ['bachelor', 'master', 'phd', 'degree', 'diploma']
        
        score = 0.0
        for keyword in education_keywords:
            if keyword in required_lower and keyword in resume_lower:
                score += 0.2
        
        return min(score, 1.0)
    
    def rank_candidates(self, resumes_df, job_description, 
                       required_experience=0, required_education=""):
        """Rank candidates based on job description"""
        print(f"\nRanking {len(resumes_df)} candidates...")
        print(f"Job Description: {job_description[:100]}...")
        
        # Clean job description
        cleaned_job_desc = self.cleaner.clean_text(job_description)
        processed_job_desc = self.cleaner.tokenize_and_lemmatize(cleaned_job_desc)
        
        # Generate embeddings for job description
        job_embedding = self.feature_extractor.encode_texts([processed_job_desc])
        
        # Generate embeddings for resumes (use processed resume text)
        resume_texts = resumes_df['processed_resume'].tolist()
        resume_embeddings = self.feature_extractor.encode_texts(resume_texts)
        
        # Calculate semantic similarity scores
        semantic_scores = self.feature_extractor.calculate_cosine_similarity(
            resume_embeddings, job_embedding
        ).flatten()
        
        # Calculate other scores
        scores_data = []
        
        for idx, row in resumes_df.iterrows():
            # Semantic similarity score
            semantic_score = semantic_scores[idx]
            
            # Keyword matching score
            keyword_score = self.calculate_keyword_match_score(
                row['Resume'], job_description
            )
            
            # Experience score
            experience_score = self.calculate_experience_score(
                row['experience_years'], required_experience
            )
            
            # Education score
            education_score = self.calculate_education_score(
                row['Resume'], required_education
            )
            
            # Calculate final weighted score
            final_score = (
                Config.SEMANTIC_WEIGHT * semantic_score +
                Config.KEYWORD_WEIGHT * keyword_score +
                Config.EXPERIENCE_WEIGHT * experience_score +
                Config.EDUCATION_WEIGHT * education_score
            )
            
            scores_data.append({
                'resume_id': idx,
                'category': row.get('Category', 'Unknown'),
                'semantic_score': semantic_score,
                'keyword_score': keyword_score,
                'experience_score': experience_score,
                'education_score': education_score,
                'final_score': final_score,
                'skills': row.get('extracted_skills', []),
                'experience_years': row['experience_years']
            })
        
        # Create scores dataframe
        scores_df = pd.DataFrame(scores_data)
        
        # Sort by final score
        scores_df = scores_df.sort_values('final_score', ascending=False).reset_index(drop=True)
        scores_df['rank'] = scores_df.index + 1
        
        # Mark shortlisted candidates
        scores_df['shortlisted'] = scores_df['final_score'] >= Config.SHORTLIST_THRESHOLD
        
        print(f"\nRanking complete!")
        print(f"Candidates shortlisted: {scores_df['shortlisted'].sum()}")
        
        return scores_df
    
    def display_top_candidates(self, scores_df, top_n=10):
        """Display top N candidates"""
        print(f"\n{'='*80}")
        print(f"TOP {top_n} CANDIDATES")
        print(f"{'='*80}\n")
        
        top_candidates = scores_df.head(top_n)
        
        for _, row in top_candidates.iterrows():
            print(f"Rank: {row['rank']}")
            print(f"Category: {row['category']}")
            print(f"Final Score: {row['final_score']:.4f}")
            print(f"  - Semantic Similarity: {row['semantic_score']:.4f}")
            print(f"  - Keyword Match: {row['keyword_score']:.4f}")
            print(f"  - Experience Match: {row['experience_score']:.4f}")
            print(f"  - Education Match: {row['education_score']:.4f}")
            print(f"Skills: {row['skills']}")
            print(f"Experience: {row['experience_years']} years")
            print(f"Shortlisted: {'YES' if row['shortlisted'] else 'NO'}")
            print(f"{'-'*80}\n")
