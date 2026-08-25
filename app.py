import streamlit as st
import google.generativeai as genai
from PIL import Image

# --- 1. CONFIGURATION ---
def configure_system():
    # Replace with your actual API key
    api_key = "AIzaSyDi6hqrY-p71aNWjqcYXF21NMoNdrlg7JQ" 
    
    try:
        genai.configure(api_key=api_key)
    except Exception as e:
        st.error(f"Configuration Error: {e}")

def get_skin_analysis(image_input, user_profile):
    """
    Sends the image and user data to Gemini for analysis.
    """
    # Use the correct model version
    model = genai.GenerativeModel('gemini-2.5-flash')
    
    # PROFESSIONAL & FRIENDLY PROMPT
    analysis_prompt = f"""

    You are a friendly skin care assistant.



    USER DATA:

    - Age/Gender: {user_profile['age']} / {user_profile['gender']}

    - Skin Type: {user_profile['skin_type']}

    - Concerns: {user_profile['concerns']}

   

    INSTRUCTIONS:

    1. Analyze the face in the image.

    2. Speak in SIMPLE, everyday English. Do not use complex medical words (e.g., instead of "erythema", say "redness").

    3. Keep sentences short and easy to read.

   

    OUTPUT FORMAT:

    ### 1. What I See

    (Describe the issue simply. Max 2 sentences.)

   

    ### 2. Simple Home Fixes

    (3 easy steps using ingredients like Turmeric, Aloe Vera, or basic hygiene. Explain WHY it works in 5 words.)

   

    ### 3. What to Buy

    (Suggest 2 generic items. Example: "A gentle foam cleanser" or "Sunscreen with SPF 50".)

   

    ### Note

    (Standard AI disclaimer: Not a doctor, for information only.)

    """
    
    try:
        response = model.generate_content([analysis_prompt, image_input])
        return response.text
    except Exception as e:
        return f"⚠️ Service Unavailable: {str(e)}"

# --- 2. UI SETUP ---
st.set_page_config(page_title="Derm-Scan AI", page_icon="✨", layout="wide")

# Initialize System
try:
    configure_system()
except:
    pass

# Custom Styling
st.markdown("""
    <style>
    .main-title {font-size: 3rem; color: #4CAF50; text-align: center; font-weight: 700;}
    .sub-text {font-size: 1.2rem; color: #555; text-align: center; margin-bottom: 20px;}
    .report-container {background-color: #f8f9fa; padding: 30px; border-radius: 12px; border-left: 6px solid #4CAF50; box-shadow: 0 4px 12px rgba(0,0,0,0.05);}
    </style>
""", unsafe_allow_html=True)

# Header
st.markdown('<div class="main-title">✨ AI Skincare Advisor</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-text">Personalized dermatological analysis powered by Artificial Intelligence</div>', unsafe_allow_html=True)
st.write("---")

# --- 3. SIDEBAR INPUTS ---
with st.sidebar:
    st.header("👤 Your Profile")
    st.info("Please provide accurate details for the best advice.")
    
    gender = st.radio("Gender", ["Female", "Male", "Other"], horizontal=True)
    age = st.slider("Age", 12, 80, 25)
    
    st.markdown("### Skin Characteristics")
    skin_type = st.selectbox(
        "Skin Type",
        ["Normal", "Oily", "Dry", "Combination", "Sensitive"]
    )

    concerns = st.multiselect(
        "Current Concerns",
        ["Acne / Pimples", "Blackheads", "Wrinkles / Aging", "Dark Spots / Pigmentation", "Redness", "Uneven Texture"]
    )
    
    profile_data = {
        "gender": gender,
        "age": age,
        "skin_type": skin_type,
        "concerns": ", ".join(concerns)
    }

# --- 4. MAIN INTERFACE ---
col1, col2 = st.columns([1, 1], gap="large")

with col1:
    st.subheader("📸 Step 1: Capture Photo")
    st.markdown("Ensure good lighting and remove glasses if possible.")
    img_file = st.camera_input("Tap to Take Photo")

with col2:
    st.subheader("📝 Step 2: Expert Analysis")
    
    if img_file is not None:
        image = Image.open(img_file)
        
        with st.spinner("🔍 Dr. AI is analyzing your skin texture..."):
            # Get the result
            result = get_skin_analysis(image, profile_data)
            
            # Display Result in a styled box
            st.markdown('<div class="report-container">', unsafe_allow_html=True)
            st.markdown(result)
            st.markdown('</div>', unsafe_allow_html=True)
            
    else:
        st.info("👋 Waiting for your photo... The analysis will appear here instantly.")