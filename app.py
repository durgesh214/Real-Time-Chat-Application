import gradio as gr
from openai import OpenAI
import json
import time
from datetime import datetime
import threading
import queue

# Initialize OpenAI client
client = OpenAI(
    api_key="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"  
)

# Global variables for chat management
chat_sessions = {}
active_models = ["gpt-4", "gpt-3.5-turbo", "gpt-4-turbo-preview"]

class ChatManager:
    def __init__(self):
        self.sessions = {}
        self.current_session_id = None
        
    def create_session(self, session_name="New Chat"):
        session_id = f"session_{int(time.time())}"
        self.sessions[session_id] = {
            "name": session_name,
            "messages": [],
            "created_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "model": "gpt-3.5-turbo",
            "system_prompt": "You are a helpful, professional AI assistant. Provide clear, accurate, and well-structured responses."
        }
        self.current_session_id = session_id
        return session_id
    
    def get_session_list(self):
        return [(f"{session['name']} ({session['created_at']})", session_id) 
                for session_id, session in self.sessions.items()]
    
    def get_current_session(self):
        if self.current_session_id and self.current_session_id in self.sessions:
            return self.sessions[self.current_session_id]
        return None

# Initialize chat manager
chat_manager = ChatManager()

def generate_ai_response(message: str, model: str = "gpt-3.5-turbo", system_prompt: str = "", chat_history: list = []) -> tuple:
    """Generate AI response with enhanced error handling and features"""
    try:
        if not message.strip():
            return "Please enter a message to continue the conversation.", []
        
        # Build messages for API
        messages = []
        
        # Add system prompt if provided
        if system_prompt.strip():
            messages.append({"role": "system", "content": system_prompt})
        else:
            messages.append({"role": "system", "content": "You are a helpful, professional AI assistant. Provide clear, accurate, and well-structured responses."})
        
        # Add chat history
        for user_msg, ai_msg in chat_history:
            if user_msg:
                messages.append({"role": "user", "content": user_msg})
            if ai_msg:
                messages.append({"role": "assistant", "content": ai_msg})
        
        # Add current message
        messages.append({"role": "user", "content": message})
        
        # Generate response
        response = client.chat.completions.create(
            model=model,
            messages=messages,
            max_tokens=2000,
            temperature=0.7,
            top_p=0.9,
            frequency_penalty=0.1,
            presence_penalty=0.1
        )
        
        ai_response = response.choices[0].message.content
        
        # Update chat history
        updated_history = chat_history + [[message, ai_response]]
        
        return ai_response, updated_history
        
    except Exception as e:
        error_msg = f"⚠️ Error: {str(e)}"
        
        # Specific error handling
        if "api_key" in str(e).lower() or "authentication" in str(e).lower():
            error_msg = "🔑 Authentication Error: Please check your OpenAI API key configuration."
        elif "quota" in str(e).lower() or "billing" in str(e).lower():
            error_msg = "💳 Quota Exceeded: Please check your OpenAI billing and usage limits."
        elif "rate_limit" in str(e).lower():
            error_msg = "⏱️ Rate Limited: Please wait a moment before sending another message."
        elif "model" in str(e).lower():
            error_msg = f"🤖 Model Error: The selected model '{model}' may not be available."
        
        return error_msg, chat_history

def chat_interface(message, history, model, system_prompt, temperature):
    """Main chat interface function"""
    if not message.strip():
        return history, ""
    
    try:
        # Generate response
        messages = []
        
        if system_prompt.strip():
            messages.append({"role": "system", "content": system_prompt})
        else:
            messages.append({"role": "system", "content": "You are a helpful, professional AI assistant."})
        
        # Add chat history
        for user_msg, ai_msg in history:
            messages.append({"role": "user", "content": user_msg})
            messages.append({"role": "assistant", "content": ai_msg})
        
        messages.append({"role": "user", "content": message})
        
        # API call
        response = client.chat.completions.create(
            model=model,
            messages=messages,
            max_tokens=2000,
            temperature=temperature,
            top_p=0.9
        )
        
        ai_response = response.choices[0].message.content
        history.append([message, ai_response])
        
        return history, ""
        
    except Exception as e:
        error_response = f"⚠️ Error: {str(e)}"
        history.append([message, error_response])
        return history, ""

def clear_chat():
    """Clear chat history"""
    return [], ""

def export_chat(history):
    """Export chat history to text format"""
    if not history:
        return "No chat history to export."
    
    export_text = f"# AI Chat Export - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n"
    
    for i, (user_msg, ai_msg) in enumerate(history, 1):
        export_text += f"## Message {i}\n"
        export_text += f"**User:** {user_msg}\n\n"
        export_text += f"**AI:** {ai_msg}\n\n"
        export_text += "---\n\n"
    
    return export_text

def load_preset_prompt(preset):
    """Load preset system prompts"""
    presets = {
        "Default Assistant": "You are a helpful, professional AI assistant. Provide clear, accurate, and well-structured responses.",
        "Code Expert": "You are an expert software developer and programmer. Provide detailed, well-commented code solutions and explanations. Focus on best practices, efficiency, and maintainability.",
        "Creative Writer": "You are a creative writing assistant. Help with storytelling, character development, plot creation, and creative expression. Be imaginative and inspiring.",
        "Business Analyst": "You are a professional business analyst. Provide strategic insights, data-driven recommendations, and help with business planning and analysis.",
        "Research Assistant": "You are a thorough research assistant. Provide well-researched, accurate information with proper context and multiple perspectives on topics.",
        "Technical Educator": "You are an expert technical educator. Explain complex concepts clearly, use examples and analogies, and adapt your teaching style to the user's level."
    }
    return presets.get(preset, presets["Default Assistant"])

# Create the advanced Gradio interface
def create_interface():
    with gr.Blocks(
        title="Professional AI Chat Assistant",
        theme=gr.themes.Soft(),
        css="""
        .gradio-container {
            max-width: 1200px !important;
            margin: auto !important;
        }
        .chat-container {
            border-radius: 12px;
            box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
        }
        .header-text {
            text-align: center;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            font-weight: bold;
        }
        """
    ) as demo:
        
        # Header
        gr.HTML("""
        <div style="text-align: center; padding: 20px;">
            <h1 class="header-text" style="font-size: 2.5em; margin-bottom: 10px;">
                🚀 Professional AI Chat Assistant
            </h1>
            <p style="font-size: 1.1em; color: #666; margin: 0;">
                Advanced conversational AI powered by OpenAI GPT models
            </p>
        </div>
        """)
        
        with gr.Row():
            # Main chat area
            with gr.Column(scale=3):
                chatbot = gr.Chatbot(
                    label="💬 AI Conversation",
                    height=500,
                    placeholder="Start a conversation with your AI assistant...",
                    show_label=True,
                    container=True,
                    elem_classes=["chat-container"]
                )
                
                with gr.Row():
                    message_input = gr.Textbox(
                        label="Your Message",
                        placeholder="Type your message here... (Press Shift+Enter for new line, Enter to send)",
                        lines=3,
                        max_lines=10,
                        scale=4,
                        show_label=False
                    )
                
                with gr.Row():
                    send_btn = gr.Button("📤 Send Message", variant="primary", scale=2)
                    clear_btn = gr.Button("🗑️ Clear Chat", variant="secondary", scale=1)
                    export_btn = gr.Button("📥 Export Chat", variant="secondary", scale=1)
            
            # Settings and controls sidebar
            with gr.Column(scale=1, min_width=300):
                gr.Markdown("### 🔧 **Chat Settings**")
                
                model_dropdown = gr.Dropdown(
                    choices=active_models,
                    value="gpt-3.5-turbo",
                    label="🤖 AI Model",
                    info="Select the AI model to use"
                )
                
                temperature_slider = gr.Slider(
                    minimum=0.0,
                    maximum=2.0,
                    value=0.7,
                    step=0.1,
                    label="🌡️ Creativity (Temperature)",
                    info="Higher values = more creative responses"
                )
                
                gr.Markdown("### 🎯 **System Prompt**")
                
                preset_dropdown = gr.Dropdown(
                    choices=[
                        "Default Assistant",
                        "Code Expert", 
                        "Creative Writer",
                        "Business Analyst",
                        "Research Assistant",
                        "Technical Educator"
                    ],
                    value="Default Assistant",
                    label="📋 Preset Prompts",
                    info="Quick-select common assistant roles"
                )
                
                system_prompt = gr.Textbox(
                    label="Custom System Prompt",
                    placeholder="Define how the AI should behave...",
                    lines=4,
                    value="You are a helpful, professional AI assistant. Provide clear, accurate, and well-structured responses."
                )
                
                # Chat export area
                gr.Markdown("### 📊 **Chat Export**")
                export_output = gr.Textbox(
                    label="Exported Chat",
                    placeholder="Exported chat will appear here...",
                    lines=6,
                    max_lines=15,
                    visible=False
                )
        
        # Statistics and info
        with gr.Row():
            gr.HTML("""
            <div style="text-align: center; padding: 15px; background: #f8f9fa; border-radius: 8px; margin-top: 20px;">
                <p style="margin: 5px; color: #666;">
                    <strong>💡 Pro Tips:</strong> 
                    Use Shift+Enter for line breaks • Try different models for varied responses • 
                    Customize system prompts for specialized tasks • Export important conversations
                </p>
            </div>
            """)
        
        # Event handlers
        def handle_send(message, history, model, system_prompt, temperature):
            return chat_interface(message, history, model, system_prompt, temperature)
        
        def handle_preset_change(preset):
            return load_preset_prompt(preset)
        
        def handle_export(history):
            if history:
                export_text = export_chat(history)
                return gr.update(value=export_text, visible=True)
            return gr.update(value="No chat history to export.", visible=True)
        
        # Wire up the events
        send_btn.click(
            fn=handle_send,
            inputs=[message_input, chatbot, model_dropdown, system_prompt, temperature_slider],
            outputs=[chatbot, message_input]
        )
        
        message_input.submit(
            fn=handle_send,
            inputs=[message_input, chatbot, model_dropdown, system_prompt, temperature_slider],
            outputs=[chatbot, message_input]
        )
        
        clear_btn.click(
            fn=clear_chat,
            outputs=[chatbot, message_input]
        )
        
        preset_dropdown.change(
            fn=handle_preset_change,
            inputs=[preset_dropdown],
            outputs=[system_prompt]
        )
        
        export_btn.click(
            fn=handle_export,
            inputs=[chatbot],
            outputs=[export_output]
        )
    
    return demo

# Launch the application
if __name__ == "__main__":
    print("🚀 Launching Professional AI Chat Assistant...")
    print("🔧 Features: Multiple models, custom prompts, chat export")
    print("🌐 Access the interface through your web browser")
    
    demo = create_interface()
    demo.launch(
        share=False,
        server_port=7860
    )