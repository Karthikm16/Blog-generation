import streamlit as st
from langchain.prompts import PromptTemplate
from langchain.llms import CTransformers


# Function to interact with the LLaMA model and generate the blog
def get_llama_response(input_text, no_words, blog_style):
    try:
        # Model loading (with proper path and error handling)
        st.info("Loading LLaMA model... This might take a few moments.")
        llm = CTransformers(model='models/llama-2-7b-chat.ggmlv3.q8_0.bin',
                            model_type='llama',
                            config={'max_new_tokens': 256, 'temperature': 0.01})

        # Prompt template for blog generation
        template = """
        Write a professional blog for the {blog_style} job profile on the topic "{input_text}" within {no_words} words.
        """
        prompt = PromptTemplate(input_variables=["blog_style", "input_text", "no_words"], template=template)

        # Generate response
        response = llm(prompt.format(blog_style=blog_style, input_text=input_text, no_words=no_words))

        # Log response
        print(response)

        return response

    except Exception as e:
        st.error(f"An error occurred: {str(e)}")
        return None


# Function to validate input fields
def validate_inputs(input_text, no_words):
    if not input_text:
        st.warning("Please enter a blog topic.")
        return False
    if not no_words or int(no_words) <= 0:
        st.warning("Please specify a valid number of words.")
        return False
    return True


# Set Streamlit page configuration
st.set_page_config(page_title="AI Blog Generator", page_icon='♉', layout='centered')

# Header and introduction
st.title("Professional Blog Generator")
st.write("""
Welcome to the AI-powered blog generation tool. You can generate blog content based on a job profile 
(researchers, data scientists, students) and a given topic.
""")

# Blog input section
st.subheader("Enter Blog Details")
input_text = st.text_input('Blog Topic', placeholder="Enter the blog topic here...")

# Columns for word count and blog style
col1, col2 = st.columns(2)

with col1:
    # Word limit slider instead of text input for ease
    no_words = st.slider("Number of Words", min_value=100, max_value=1000, step=50, value=300)

with col2:
    # Select blog writing style
    blog_style = st.selectbox("Target Audience", ('Researchers', 'Data Scientists', 'Students'))

# Generate button
submit = st.button('Generate Blog')

# Process blog generation on click
if submit:
    if validate_inputs(input_text, no_words):
        with st.spinner("Generating your blog..."):
            response = get_llama_response(input_text, no_words, blog_style)
            if response:
                st.success("Blog generated successfully!")
                # Display the generated blog content in markdown format
                st.markdown(response)
            else:
                st.error("Failed to generate the blog. Please try again.")

# Footer with GitHub and LinkedIn icons
st.write("---")
st.write("Powered by [LLaMA Model](https://github.com/ggerganov/llama.cpp).")

# HTML code for GitHub and LinkedIn icons
st.markdown("""
<div style="display: flex; justify-content: center; align-items: center;">
    <a href="https://github.com/Karthikm16" target="_blank">
        <img src="https://img.icons8.com/ios-glyphs/30/000000/github.png" alt="GitHub" style="margin-right: 10px;">
    </a>
    <a href="https://www.linkedin.com/in/karthik-m-170788253/" target="_blank">
        <img src="https://img.icons8.com/ios-filled/30/0077B5/linkedin.png" alt="LinkedIn" style="margin-left: 10px;">
    </a>
</div>
""", unsafe_allow_html=True)

# Additional styling (optional)
st.markdown("""
<style>
    .stButton>button {
        background-color: #4CAF50;
        color: white;
    }
    .stTextInput>div>input {
        font-size: 16px;
    }
</style>
""", unsafe_allow_html=True)
