# Resume Classification and Career Domain Prediction System Using NLP and Naive Bayes Algorithm

## Cover Page

**College Name:** [Enter College Name]  
**Department Name:** Department of Computer Applications  
**Subject Name:** Artificial Intelligence and Machine Learning AAT  
**Project Title:** Resume Classification and Career Domain Prediction System Using NLP and Multinomial Naive Bayes Algorithm  
**Domain:** Natural Language Processing (NLP)  
**Machine Learning Algorithm:** Multinomial Naive Bayes  
**Feature Extraction Technique:** TF-IDF (Term Frequency-Inverse Document Frequency)  
**Programming Language:** Python  
**Development Environment:** Google Colab  
**Libraries Used:** Pandas, NumPy, NLTK, Scikit-Learn, Matplotlib, Seaborn  
**Team Members:** [Enter Team Member Names]  
**Guide Name:** [Enter Guide Name]  
**Academic Year:** 2025-2026  

---

## Abstract

Career selection is an important decision for students in computer science, information technology, and related professional courses. Many students learn skills from more than one technical area, such as Python, SQL, JavaScript, cloud platforms, data analytics, machine learning, DevOps, and cybersecurity. Because of this overlap, students may find it difficult to identify the most suitable career domain based only on their current resume or skill profile. Manual resume analysis by faculty members, career counselors, or human resource professionals can be useful, but it is also time-consuming, subjective, and difficult to scale when many resumes must be evaluated.

This project presents a Resume Classification and Career Domain Prediction System using Natural Language Processing and the Multinomial Naive Bayes algorithm. The system accepts resume text or skill-based profile text as input, performs text preprocessing, converts the cleaned text into numerical features using TF-IDF, and predicts the most suitable technical career domain. The preprocessing stage includes lowercasing, tokenization, stopword removal, and stemming. TF-IDF is used to represent important words while reducing the weight of common words. Multinomial Naive Bayes is selected because it is efficient, interpretable, and widely used for text classification tasks.

The proposed system supports career domains such as Data Science, Machine Learning, Web Development, Backend Development, Frontend Development, DevOps, Cloud Computing, Cybersecurity, Android Development, Data Analytics, AI/ML, and other technical areas. The system is implemented in Python using libraries such as Pandas, NumPy, NLTK, Scikit-Learn, Matplotlib, and Seaborn. The model is trained and tested using labeled resume or skill-text data. Evaluation is performed using accuracy, precision, recall, F1 score, and confusion matrix. The expected model accuracy lies between 88% and 95% depending on dataset quality, category balance, and preprocessing decisions.

The project demonstrates how machine learning can support automated career guidance by analyzing skills present in resume text. It does not replace expert counseling, but it provides a consistent, scalable, and data-driven first-level recommendation system for students.

**Keywords:** Resume Classification, Career Domain Prediction, NLP, TF-IDF, Multinomial Naive Bayes, Machine Learning, Text Classification, Career Guidance

---

# 1. Introduction

## 1.1 Natural Language Processing

Natural Language Processing, commonly called NLP, is a branch of artificial intelligence that focuses on enabling computers to understand, process, analyze, and generate human language. Human language is highly flexible and complex. The same idea may be expressed in many different ways, and the meaning of a sentence often depends on context, grammar, vocabulary, and domain knowledge. NLP attempts to bridge the gap between human communication and machine understanding.

In traditional programming, input data is usually structured and predictable. For example, a database table may contain columns such as name, age, marks, and department. However, resumes, job descriptions, emails, feedback forms, social media posts, and documents are mostly unstructured text. Such text cannot be directly processed by machine learning algorithms without converting it into a structured numerical format. NLP provides methods for cleaning, transforming, and extracting useful information from text.

Common NLP tasks include tokenization, stopword removal, stemming, lemmatization, part-of-speech tagging, named entity recognition, sentiment analysis, machine translation, text summarization, document classification, and information retrieval. In this project, NLP is used mainly for text preprocessing and classification. The system analyzes resume text, identifies important technical keywords, converts the text into TF-IDF vectors, and predicts the most suitable career domain.

## 1.2 Importance of Resume Analysis

A resume is a concise document that represents a candidate's education, skills, projects, internships, certifications, and work experience. For students, a resume is often the first professional document used to communicate their capabilities to recruiters, placement officers, mentors, and academic guides. A well-written resume can show whether a student is more aligned with data science, web development, machine learning, cybersecurity, cloud computing, mobile development, or another technical domain.

Resume analysis is important because resumes contain evidence of a candidate's technical direction. For example, a resume containing words such as "Python", "Pandas", "NumPy", "visualization", "statistics", and "machine learning" may indicate suitability for Data Science or Machine Learning. A resume containing "HTML", "CSS", "JavaScript", "React", and "responsive design" may indicate Frontend Development. Similarly, "Docker", "Kubernetes", "CI/CD", "AWS", and "Linux" may point toward DevOps or Cloud Computing.

... (file truncated for brevity)