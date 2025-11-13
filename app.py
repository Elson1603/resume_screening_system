import os
import sys
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from flask import Flask, render_template, request, redirect, url_for, flash, jsonify
from werkzeug.utils import secure_filename
import pandas as pd
from datetime import datetime
import PyPDF2
from docx import Document

from src.config import Config
from src.data_preprocessing import DataPreprocessor
from src.resume_ranking import ResumeRanker
from src.utils import TextCleaner

app = Flask(__name__)
app.secret_key = 'your-secret-key-here'
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max file size

ALLOWED_EXTENSIONS = {'pdf', 'docx', 'txt'}

# Create upload folder
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

# Initialize components
preprocessor = DataPreprocessor()
ranker = ResumeRanker()
cleaner = TextCleaner()

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def extract_text_from_pdf(filepath):
    """Extract text from PDF"""
    text = ""
    with open(filepath, 'rb') as file:
        pdf_reader = PyPDF2.PdfReader(file)
        for page in pdf_reader.pages:
            text += page.extract_text()
    return text

def extract_text_from_docx(filepath):
    """Extract text from DOCX"""
    doc = Document(filepath)
    text = "\n".join([paragraph.text for paragraph in doc.paragraphs])
    return text

def extract_text_from_file(filepath):
    """Extract text from various file formats"""
    ext = filepath.rsplit('.', 1)[1].lower()
    
    if ext == 'pdf':
        return extract_text_from_pdf(filepath)
    elif ext == 'docx':
        return extract_text_from_docx(filepath)
    elif ext == 'txt':
        with open(filepath, 'r', encoding='utf-8') as file:
            return file.read()
    return ""

@app.route('/')
def index():
    """Home page"""
    return render_template('index.html')

@app.route('/upload', methods=['GET', 'POST'])
def upload():
    """Upload resumes and job description"""
    if request.method == 'POST':
        # Get job description
        job_description = request.form.get('job_description')
        required_experience = int(request.form.get('required_experience', 0))
        required_education = request.form.get('required_education', '')
        
        if not job_description:
            flash('Please enter a job description', 'error')
            return redirect(url_for('upload'))
        
        # Get uploaded files
        files = request.files.getlist('resumes')
        
        if not files or files[0].filename == '':
            flash('Please upload at least one resume', 'error')
            return redirect(url_for('upload'))
        
        # Process uploaded resumes
        resumes_data = []
        
        for file in files:
            if file and allowed_file(file.filename):
                filename = secure_filename(file.filename)
                filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
                file.save(filepath)
                
                # Extract text
                resume_text = extract_text_from_file(filepath)
                
                if resume_text:
                    # Clean and process
                    cleaned_text = cleaner.clean_text(resume_text)
                    processed_text = cleaner.tokenize_and_lemmatize(cleaned_text)
                    skills = cleaner.extract_skills(resume_text)
                    experience = cleaner.extract_experience_years(resume_text)
                    
                    resumes_data.append({
                        'filename': filename,
                        'Resume': resume_text,
                        'cleaned_resume': cleaned_text,
                        'processed_resume': processed_text,
                        'extracted_skills': skills,
                        'experience_years': experience,
                        'Category': 'Uploaded'
                    })
                
                # Clean up uploaded file
                os.remove(filepath)
        
        if not resumes_data:
            flash('No valid resumes were processed', 'error')
            return redirect(url_for('upload'))
        
        # Create DataFrame
        df_resumes = pd.DataFrame(resumes_data)
        
        # Rank candidates
        scores_df = ranker.rank_candidates(
            df_resumes,
            job_description,
            required_experience,
            required_education
        )
        
        # Merge with resume data
        results_df = pd.merge(
            scores_df,
            df_resumes[['filename', 'extracted_skills', 'experience_years']],
            left_on=scores_df.index,
            right_on=df_resumes.index,
            how='left'
        )
        
        # Convert to dict for template
        results = results_df.to_dict('records')
        
        return render_template('results.html',
                             results=results,
                             job_description=job_description,
                             total_candidates=len(results),
                             shortlisted_count=scores_df['shortlisted'].sum())
    
    return render_template('upload.html')

@app.route('/demo', methods=['GET', 'POST'])
def demo():
    """Demo with existing dataset"""
    if request.method == 'POST':
        # Load processed dataset
        df_processed = preprocessor.load_processed_data()
        
        if df_processed is None:
            flash('Dataset not found. Please run train.py first.', 'error')
            return redirect(url_for('demo'))
        
        # Get job description from form
        job_description = request.form.get('job_description')
        required_experience = int(request.form.get('required_experience', 0))
        required_education = request.form.get('required_education', '')
        
        # Sample resumes (limit to 50 for speed)
        df_sample = df_processed.sample(n=min(50, len(df_processed)), random_state=42)
        
        # Rank candidates
        scores_df = ranker.rank_candidates(
            df_sample,
            job_description,
            required_experience,
            required_education
        )
        
        # Prepare results
        results = []
        for idx, row in scores_df.iterrows():
            # Parse skills if they're stored as string
            skills = row['skills']
            if isinstance(skills, str):
                # Try to parse as list if it's a string representation
                try:
                    import ast
                    skills = ast.literal_eval(skills)
                except:
                    skills = []
            elif not isinstance(skills, list):
                skills = []
            
            results.append({
                'rank': row['rank'],
                'filename': f"Resume_{row['resume_id']}",
                'category': row['category'],
                'final_score': round(row['final_score'], 4),
                'semantic_score': round(row['semantic_score'], 4),
                'keyword_score': round(row['keyword_score'], 4),
                'experience_score': round(row['experience_score'], 4),
                'education_score': round(row['education_score'], 4),
                'skills': skills,
                'experience_years': row['experience_years'],
                'shortlisted': row['shortlisted']
            })
        
        return render_template('results.html',
                             results=results,
                             job_description=job_description,
                             total_candidates=len(results),
                             shortlisted_count=scores_df['shortlisted'].sum())
    
    # Sample job descriptions for demo
    sample_jobs = {
        'Python Developer': 'We are looking for a Python Developer with experience in Machine Learning, NLP, Django, Flask, and web development. Minimum 2 years experience required.',
        'Data Scientist': 'Seeking a Data Scientist skilled in Python, R, TensorFlow, PyTorch, SQL, and statistical analysis. Strong background in machine learning required.',
        'Java Developer': 'Looking for Java Developer with Spring Boot, Hibernate, microservices, and REST API experience. 3+ years required.',
    }
    
    return render_template('demo.html', sample_jobs=sample_jobs)

@app.route('/api/rank', methods=['POST'])
def api_rank():
    """API endpoint for ranking"""
    data = request.get_json()
    
    # Process and return JSON response
    return jsonify({'status': 'success'})

if __name__ == '__main__':
    port = int(os.environ.get("PORT", 5000))
    app.run(host='0.0.0.0', port=port)

