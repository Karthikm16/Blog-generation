import gradio as gr
from langchain_community.llms import CTransformers  # Updated import

# Function to interact with the LLaMA model and generate the blog
def get_llama_response(input_text, no_words, blog_style):
    try:
        # Model loading (with proper path and error handling)
        llm = CTransformers(model='models/llama-2-7b-chat.ggmlv3.q8_0.bin',
                            model_type='llama',
                            config={'max_new_tokens': 256, 'temperature': 0.01})

        # Prompt template for blog generation
        template = """
        Write a professional blog for the {blog_style} job profile on the topic "{input_text}" within {no_words} words.
        """
        response = llm(template.format(blog_style=blog_style, input_text=input_text, no_words=no_words))
        return response

    except Exception as e:
        return f"An error occurred: {str(e)}"


# Gradio interface function to trigger blog generation
def generate_blog(input_text, no_words, blog_style):
    if not input_text:
        return "Please enter a valid blog topic."
    if not no_words or int(no_words) <= 0:
        return "Please enter a valid word count."
    
    response = get_llama_response(input_text, no_words, blog_style)
    if response:
        return response
    else:
        return "Failed to generate the blog. Please try again."


# Gradio Blocks Layout (Professional UI)
with gr.Blocks(css=".footer {text-align: center;}") as app:
    # Header
    gr.Markdown(
        """
        # AI Blog Generator
        **Create professional blogs** based on specific job profiles and topics in just a few seconds.
        """
    )
    
    # Input Elements
    with gr.Row():
        input_text = gr.Textbox(label="Blog Topic", placeholder="Enter the blog topic here...", lines=2)
    
    with gr.Row():
        no_words = gr.Slider(minimum=100, maximum=1000, step=50, value=300, label="Number of Words")
        blog_style = gr.Dropdown(choices=["Researchers", "Data Scientists", "Students"], label="Target Audience")

    # Generate Button and Output
    with gr.Row():
        generate_button = gr.Button("Generate Blog")
    
    blog_output = gr.Textbox(label="Generated Blog", placeholder="Your blog will appear here...", lines=10)

    # GitHub and LinkedIn Icons
    gr.Markdown(
        """
        <div class="footer">
            <a href="https://github.com/Karthikm16" target="_blank">
                <img src="https://img.icons8.com/ios-glyphs/30/000000/github.png" alt="GitHub" style="margin-right: 10px;">
            </a>
            <a href="https://www.linkedin.com/in/karthik-m-170788253/" target="_blank">
                <img src="https://img.icons8.com/ios-filled/30/0077B5/linkedin.png" alt="LinkedIn" style="margin-left: 10px;">
            </a>
        </div>
        """, 
        visible=True
    )

    # Bind Functionality
    generate_button.click(generate_blog, inputs=[input_text, no_words, blog_style], outputs=blog_output)

# Launch the app
if __name__ == "__main__":
    app.launch(share=True)
