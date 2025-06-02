import streamlit as st
import pandas as pd
import PyPDF2
import numpy as np
from sentence_transformers import SentenceTransformer
import faiss
import ollama
import json
import re
from datetime import datetime
import io

# Initialize session state
def initialize_session_state():
    """Initialize all session state variables"""
    if 'stage' not in st.session_state:
        st.session_state.stage = 'greeting'
    if 'user_data' not in st.session_state:
        st.session_state.user_data = {}
    if 'chat_history' not in st.session_state:
        st.session_state.chat_history = []
    if 'vectorizer' not in st.session_state:
        st.session_state.vectorizer = None
    if 'pdf_content' not in st.session_state:
        st.session_state.pdf_content = None
    if 'faiss_index' not in st.session_state:
        st.session_state.faiss_index = None

class ChatbotVectorizer:
    def __init__(self):
        # Initialize sentence transformer for embeddings
        self.model = SentenceTransformer('all-MiniLM-L6-v2')
        self.index = None
        self.documents = []
        self.embeddings = None
    
    def load_pdf_content(self, pdf_file):
        """Extract text from PDF file"""
        try:
            pdf_reader = PyPDF2.PdfReader(pdf_file)
            text = ""
            for page in pdf_reader.pages:
                text += page.extract_text() + "\n"
            return text
        except Exception as e:
            st.error(f"Error reading PDF: {str(e)}")
            return None
    
    def chunk_text(self, text, chunk_size=500):
        """Split text into chunks for better vector search"""
        sentences = re.split(r'[.!?]+', text)
        chunks = []
        current_chunk = ""
        
        for sentence in sentences:
            if len(current_chunk + sentence) < chunk_size:
                current_chunk += sentence + ". "
            else:
                if current_chunk:
                    chunks.append(current_chunk.strip())
                current_chunk = sentence + ". "
        
        if current_chunk:
            chunks.append(current_chunk.strip())
        
        return chunks
    
    def create_vector_index(self, text_chunks):
        """Create FAISS index from text chunks"""
        try:
            self.documents = text_chunks
            self.embeddings = self.model.encode(text_chunks)
            
            # Create FAISS index
            dimension = self.embeddings.shape[1]
            self.index = faiss.IndexFlatIP(dimension)  # Inner product for similarity
            self.index.add(self.embeddings.astype('float32'))
            
            return True
        except Exception as e:
            st.error(f"Error creating vector index: {str(e)}")
            return False
    
    def search_similar(self, query, top_k=3):
        """Search for similar content using vector similarity"""
        if self.index is None:
            return []
        
        try:
            query_embedding = self.model.encode([query])
            scores, indices = self.index.search(query_embedding.astype('float32'), top_k)
            
            results = []
            for i, idx in enumerate(indices[0]):
                if scores[0][i] > 0.3:  # Threshold for relevance
                    results.append({
                        'text': self.documents[idx],
                        'score': float(scores[0][i])
                    })
            
            return results
        except Exception as e:
            st.error(f"Error in similarity search: {str(e)}")
            return []

def initialize_vectorizer():
    """Initialize the vectorizer with sample data"""
    if st.session_state.vectorizer is None:
        st.session_state.vectorizer = ChatbotVectorizer()

def load_csv_codes(csv_content):
    """Load CSV codes for validation"""
    try:
        df = pd.read_csv(io.StringIO(csv_content))
        return df
    except Exception as e:
        st.error(f"Error loading CSV: {str(e)}")
        return None

def validate_email(email):
    """Validate email format"""
    pattern = r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$'
    return re.match(pattern, email) is not None

def validate_mobile(mobile):
    """Validate mobile number format"""
    pattern = r'^[+]?[\d\s\-()]{10,15}$'
    return re.match(pattern, mobile) is not None

def check_ollama_status():
    """Check if Ollama is running and what models are available"""
    try:
        # Test if Ollama is accessible
        models = ollama.list()
        return {
            'status': 'active',
            'models': [model['name'] for model in models['models']],
            'error': None
        }
    except Exception as e:
        return {
            'status': 'inactive',
            'models': [],
            'error': str(e)
        }

def generate_response_with_ollama(context, query, model_name="llama2"):
    """Generate response using Ollama LLM"""
    try:
        prompt = f"""Based on the following context, answer the user's question about BRAND'S® Essence of Chicken products.

Context: {context}

Question: {query}

Please provide a helpful and accurate answer based on the context provided. If the information is not in the context, politely say you don't have that specific information."""

        response = ollama.generate(
            model=model_name,
            prompt=prompt,
            options={
                'temperature': 0.7,
                'max_tokens': 200
            }
        )
        return response['response']
    except Exception as e:
        return f"I apologize, but I'm having trouble accessing the Ollama service. Error: {str(e)}"

def render_greeting_stage():
    """Render the greeting stage"""
    st.markdown("### Hello! Welcome to BRAND'S® Customer Support! 👋")
    st.markdown("I'm here to help you with your questions. Let's start by getting to know you better.")
    
    if st.button("Let's Get Started!", key="start_btn"):
        st.session_state.stage = 'registration'
        st.rerun()

def render_registration_stage():
    """Render the registration stage"""
    st.markdown("### Please provide your details:")
    
    with st.form("registration_form"):
        first_name = st.text_input("First Name *", key="first_name")
        last_name = st.text_input("Last Name *", key="last_name")
        email = st.text_input("Email *", key="email")
        dob = st.date_input("Date of Birth *", key="dob")
        mobile = st.text_input("Mobile Number *", key="mobile")
        lucky_draw = st.selectbox("Enroll for Lucky Draw?", ["Yes", "No"], key="lucky_draw")
        
        submit_btn = st.form_submit_button("Submit Registration")
        
        if submit_btn:
            errors = []
            
            if not first_name.strip():
                errors.append("First name is required")
            if not last_name.strip():
                errors.append("Last name is required")
            if not email.strip():
                errors.append("Email is required")
            elif not validate_email(email):
                errors.append("Please enter a valid email address")
            if not mobile.strip():
                errors.append("Mobile number is required")
            elif not validate_mobile(mobile):
                errors.append("Please enter a valid mobile number")
            
            if errors:
                for error in errors:
                    st.error(error)
            else:
                st.session_state.user_data = {
                    'first_name': first_name,
                    'last_name': last_name,
                    'email': email,
                    'dob': dob.strftime('%Y-%m-%d'),
                    'mobile': mobile,
                    'lucky_draw': lucky_draw,
                    'registration_time': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
                }
                st.success("Registration successful! ✅")
                st.session_state.stage = 'crm_question'
                st.rerun()

def render_crm_question_stage():
    """Render the CRM question stage"""
    st.markdown(f"### Thank you, {st.session_state.user_data['first_name']}! 🎉")
    st.markdown("Would you like to know about our CRM system and customer support?")
    
    col1, col2 = st.columns(2)
    with col1:
        if st.button("Yes, tell me about CRM", key="yes_crm"):
            st.session_state.stage = 'pdf_setup'
            st.rerun()
    
    with col2:
        if st.button("No, skip to chat", key="no_crm"):
            st.session_state.stage = 'chat'
            st.rerun()

def render_pdf_setup_stage():
    """Render the PDF setup stage"""
    st.markdown("### Setting up Customer Support Knowledge Base")
    st.markdown("I'm loading the customer support information...")
    
    # Here you would load your actual PDF file
    # For demonstration, I'll show how to handle file upload
    st.markdown("**Note:** In your implementation, replace this section with:")
    st.code("""
# Load your PDF file
with open('Customer_Bot_Answer.pdf', 'rb') as file:
    pdf_content = st.session_state.vectorizer.load_pdf_content(file)
    
# Load your CSV file
with open('Sample_13_Digit_Codes.csv', 'r') as file:
    csv_content = file.read()
""")
    
    # Simulate PDF loading with sample content
    sample_content = """
BRAND'S® Essence of Chicken is a premium health supplement made from high-quality chicken essence. It provides essential nutrients and amino acids to support overall health and vitality.

We offer different flavors including Original, Reduced Sugar, and Cordyceps varieties to suit different preferences and health needs.

To join our rewards program, simply register on our website or mobile app with your personal details. You'll start earning points immediately with every purchase.

Registration is required to start collecting points. Once registered, you'll receive a unique member ID and can track your points balance.

You earn 10 points per bottle purchased. Bonus points are available during special promotions.

If your code isn't working, please check that you've entered it correctly. Each code can only be used once. Contact customer service if the problem persists.

No, each code can only be scanned once. Duplicate scanning will not award additional points.

Cashback is available through our partnership with TrueMoney Wallet. Link your account to receive automatic cashback on purchases.

First-time users receive a welcome bonus of 50 points and 5% cashback on their first purchase.

Link your LINE account through the app settings to receive exclusive offers and notifications.

You can sign up for TrueMoney Wallet through their app or website. It's free and provides secure payment options.

Prizes include product vouchers, cash rewards, exclusive merchandise, and grand prizes like travel packages.

Winners are announced monthly on our website and social media channels.

Purchase more products and participate in special campaigns to increase your winning chances.

BRAND'S® products are available at major supermarkets, pharmacies, and online stores nationwide.

Yes, we offer home delivery through our website and partner delivery services.

Our products are generally safe, but we recommend consulting healthcare providers for children under 12 and pregnant women.

Our products are manufactured in Thailand under strict quality control standards and international certifications.
"""
    
    if st.button("Load Knowledge Base", key="load_kb"):
        with st.spinner("Processing customer support data..."):
            # Create text chunks
            chunks = st.session_state.vectorizer.chunk_text(sample_content)
            
            # Create vector index
            if st.session_state.vectorizer.create_vector_index(chunks):
                st.session_state.pdf_content = sample_content
                st.success("Knowledge base loaded successfully! ✅")
                st.session_state.stage = 'chat'
                st.rerun()
            else:
                st.error("Failed to create vector index")

def render_chat_stage():
    """Render the chat stage"""
    st.markdown(f"### Chat with Customer Support Bot")
    st.markdown(f"Hello {st.session_state.user_data['first_name']}! Ask me anything about BRAND'S® products.")
    
    # Display chat history
    for msg in st.session_state.chat_history:
        if msg['sender'] == 'user':
            with st.chat_message("user"):
                st.write(msg['message'])
        else:
            with st.chat_message("assistant"):
                st.write(msg['message'])
    
    # Chat input
    user_query = st.chat_input("Type your question here...")
    
    if user_query:
        # Add user message to history
        st.session_state.chat_history.append({
            'sender': 'user',
            'message': user_query,
            'timestamp': datetime.now().strftime('%H:%M:%S')
        })
        
        # Search for relevant context
        if st.session_state.vectorizer and st.session_state.vectorizer.index is not None:
            search_results = st.session_state.vectorizer.search_similar(user_query, top_k=3)
            
            if search_results:
                context = "\n".join([result['text'] for result in search_results])
                
                # Generate response using Ollama
                try:
                    # Check if Ollama is available
                    ollama_status = check_ollama_status()
                    if ollama_status['status'] == 'active' and ollama_status['models']:
                        response = generate_response_with_ollama(context, user_query)
                    else:
                        response = f"Based on our knowledge base: {search_results[0]['text'][:300]}..."
                except Exception as e:
                    response = f"I found relevant information: {search_results[0]['text'][:300]}..."
            else:
                response = "I apologize, but I couldn't find specific information about that in our current database. Please contact our customer service for detailed assistance."
        else:
            response = "The knowledge base is still loading. Please try again in a moment."
        
        # Add bot response to history
        st.session_state.chat_history.append({
            'sender': 'bot',
            'message': response,
            'timestamp': datetime.now().strftime('%H:%M:%S')
        })
        
        st.rerun()

def render_sidebar():
    """Render the sidebar with user information and system status"""
    with st.sidebar:
        st.markdown("### System Status")
        
        # Check Ollama status
        ollama_status = check_ollama_status()
        
        if ollama_status['status'] == 'active':
            st.success("🟢 Ollama: Active")
            if ollama_status['models']:
                st.markdown("**Available Models:**")
                for model in ollama_status['models']:
                    st.markdown(f"• {model}")
            else:
                st.warning("No models found. Please pull a model first.")
                st.markdown("Run: `ollama pull llama2`")
        else:
            st.error("🔴 Ollama: Inactive")
            st.markdown(f"**Error:** {ollama_status['error']}")
            st.markdown("**To start Ollama:**")
            st.markdown("1. Install Ollama from https://ollama.ai")
            st.markdown("2. Run `ollama serve` in terminal")
            st.markdown("3. Pull a model: `ollama pull llama2`")
        
        st.markdown("---")
        
        # Vectorizer status
        if st.session_state.vectorizer and st.session_state.vectorizer.index is not None:
            st.success("🟢 Vector Search: Ready")
        else:
            st.warning("🟡 Vector Search: Not Ready")
        
        st.markdown("---")
        
        st.markdown("### User Information")
        if st.session_state.user_data:
            st.json(st.session_state.user_data)
        
        if st.button("Reset Chat", key="reset_chat"):
            st.session_state.stage = 'greeting'
            st.session_state.user_data = {}
            st.session_state.chat_history = []
            st.rerun()
            
        # Test Ollama button
        if st.button("Test Ollama Connection", key="test_ollama"):
            with st.spinner("Testing Ollama..."):
                test_response = generate_response_with_ollama("BRAND'S Essence of Chicken is a health supplement.", "What is BRAND'S?", "llama2")
                st.text_area("Ollama Test Response:", test_response, height=150)

def main():
    """Main function to run the chatbot"""
    st.title("🤖 BRAND'S® Customer Support Chatbot")
    st.markdown("---")
    
    # Initialize session state
    initialize_session_state()
    initialize_vectorizer()
    
    # Render appropriate stage
    if st.session_state.stage == 'greeting':
        render_greeting_stage()
    elif st.session_state.stage == 'registration':
        render_registration_stage()
    elif st.session_state.stage == 'crm_question':
        render_crm_question_stage()
    elif st.session_state.stage == 'pdf_setup':
        render_pdf_setup_stage()
    elif st.session_state.stage == 'chat':
        render_chat_stage()
        render_sidebar()

if __name__ == "__main__":
    main()