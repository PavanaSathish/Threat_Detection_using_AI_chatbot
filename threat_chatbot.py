import streamlit as st
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import pandas as pd
import random

# Sample threat dataset
def create_threat_dataset():
    data = {
        "threat_name": [
            "Phishing Attack", "Malware Infection", "DDoS Attack", 
            "SQL Injection", "Cross-Site Scripting", "Insider Threat",
            "Ransomware", "Man-in-the-Middle", "Zero-Day Exploit",
            "Brute Force Attack", "Data Breach", "Social Engineering"
        ],
        "description": [
            "Deceptive emails or websites tricking users to reveal sensitive information",
            "Malicious software infecting systems to damage or gain unauthorized access",
            "Overwhelming a system with traffic to make it unavailable to users",
            "Injecting malicious SQL code to manipulate databases",
            "Injecting client-side scripts into web pages viewed by others",
            "Malicious actions by authorized users within an organization",
            "Malware that encrypts files and demands payment for decryption",
            "Secretly intercepting and possibly altering communications",
            "Attack exploiting unknown vulnerability before patch is available",
            "Repeated login attempts to guess credentials and gain access",
            "Unauthorized access and retrieval of sensitive data",
            "Psychological manipulation to trick people into giving up information"
        ],
        "prevention": [
            "Educate users, use email filters, verify sender authenticity",
            "Use antivirus, keep systems updated, don't open suspicious attachments",
            "Implement traffic filtering, use CDN, increase bandwidth capacity",
            "Use parameterized queries, input validation, web application firewalls",
            "Validate/sanitize inputs, implement Content Security Policy",
            "Implement least privilege access, monitor user activities",
            "Regular backups, email filtering, keep systems updated",
            "Use encryption, VPNs, certificate pinning, secure protocols",
            "Prompt patching, intrusion detection systems, network segmentation",
            "Account lockouts, CAPTCHAs, strong password policies, 2FA",
            "Encrypt sensitive data, access controls, regular audits",
            "Security awareness training, verification procedures"
        ],
        "severity": ["High"]*6 + ["Critical"]*6
    }
    return pd.DataFrame(data)

# Initialize the app
st.title("🔒 AI Threat Analysis Chatbot")
st.write("Describe a potential security threat and I'll analyze it for you.")

# Load or create dataset
threat_df = create_threat_dataset()

# Initialize TF-IDF Vectorizer
tfidf = TfidfVectorizer(stop_words='english')
tfidf_matrix = tfidf.fit_transform(threat_df['description'] + " " + threat_df['threat_name'])

# Initialize chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display chat messages
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Chat input
if prompt := st.chat_input("Describe a potential threat..."):
    # Add user message to chat history
    st.session_state.messages.append({"role": "user", "content": prompt})
    
    # Display user message
    with st.chat_message("user"):
        st.markdown(prompt)
    
    # Analyze the threat
    prompt_vec = tfidf.transform([prompt])
    similarities = cosine_similarity(prompt_vec, tfidf_matrix)
    most_similar_idx = np.argmax(similarities)
    similarity_score = similarities[0, most_similar_idx]
    
    if similarity_score > 0.3:  # Threshold for matching
        threat = threat_df.iloc[most_similar_idx]
        response = f"""🚨 **Potential Threat Identified**: {threat['threat_name']} ({threat['severity']} severity)
        
📝 **Description**: {threat['description']}
        
🛡️ **Prevention**: {threat['prevention']}
        
🔍 **Confidence**: {similarity_score:.0%} match"""
    else:
        response = """⚠️ **Unknown Threat Pattern** 
        
I couldn't find a close match in my database. This might be a new or complex threat. 
Consider consulting with a security specialist and provide more details if possible."""

    # Display assistant response
    with st.chat_message("assistant"):
        st.markdown(response)
    
    # Add assistant response to chat history
    st.session_state.messages.append({"role": "assistant", "content": response})