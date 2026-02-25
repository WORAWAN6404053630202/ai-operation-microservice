# code/main_gradio.py
"""
Gradio Chat Interface for PersonaSupervisor
"""

import gradio as gr
import uuid
from typing import List, Dict
from model.state_manager import StateManager
from model.conversation_state import ConversationState
from model.persona_supervisor import PersonaSupervisor
from service.local_vector_store import get_retriever


class PersonaChatInterface:
    def __init__(self):
        self.state_manager = StateManager()
        self.retriever = get_retriever(fail_if_empty=True)
        self.supervisor = PersonaSupervisor(retriever=self.retriever)
        self.session_id = str(uuid.uuid4())[:8]
        
        # Initialize state
        self.state = self._create_initial_state()
        
        # Get greeting
        self.state, greeting = self.supervisor.handle(
            state=self.state, 
            user_input=""
        )
        self.state_manager.save(self.session_id, self.state)
        self.greeting = greeting or "Hello! How can I help you today?"
    
    def _create_initial_state(self) -> ConversationState:
        return ConversationState(
            session_id=self.session_id,
            persona_id="practical",
            context={},
        )
    
    def chat(
        self, 
        message: str, 
        history: List[Dict[str, str]]
    ) -> tuple:
        """Handle chat interaction"""
        
        if not message.strip():
            return "", history
        
        try:
            # Process user input through supervisor
            self.state, reply = self.supervisor.handle(
                state=self.state,
                user_input=message
            )
            
            # Save state
            self.state_manager.save(self.session_id, self.state)
            
            # Update history
            history.append({"role": "user", "content": message})
            history.append({"role": "assistant", "content": reply})
            
            return "", history
            
        except Exception as e:
            error_msg = f"Error: {str(e)}"
            history.append({"role": "user", "content": message})
            history.append({"role": "assistant", "content": error_msg})
            return "", history
    
    def reset_chat(self):
        """Reset conversation and create new session"""
        # Delete old session
        self.state_manager.delete(self.session_id)
        
        # Create new session
        self.session_id = str(uuid.uuid4())[:8]
        self.state = self._create_initial_state()
        
        # Get new greeting
        self.state, greeting = self.supervisor.handle(
            state=self.state,
            user_input=""
        )
        self.state_manager.save(self.session_id, self.state)
        
        # Return greeting in history
        initial_history = [
            {"role": "assistant", "content": greeting or self.greeting}
        ]
        
        return initial_history, self._get_info()
    
    def _get_info(self) -> str:
        """Display current state info"""
        snapshot = self.state.snapshot()
        
        return f"""
**Session ID:** {self.session_id}
**Persona:** {snapshot['persona_id']}
**Round:** {snapshot['round']}
**Messages:** {snapshot['num_messages']}
**Docs:** {snapshot['num_docs']}
**Last Action:** {self.state.last_action or 'None'}
        """.strip()
    
    def get_initial_history(self):
        """Get initial history with greeting"""
        return [{"role": "assistant", "content": self.greeting}]


def create_app():
    chat_interface = PersonaChatInterface()
    
    with gr.Blocks(title="Persona AI Chat") as app:
        gr.Markdown("# 🤖 Persona AI Assistant")
        gr.Markdown("*Powered by PersonaSupervisor with RAG*")
        
        with gr.Row():
            with gr.Column(scale=3):
                chatbot = gr.Chatbot(
                    label="Conversation",
                    height=500,
                    value=chat_interface.get_initial_history()
                )
                
                with gr.Row():
                    msg = gr.Textbox(
                        label="Message",
                        placeholder="Type your message here...",
                        scale=4,
                        autofocus=True
                    )
                    submit = gr.Button("Send", variant="primary", scale=1)
                
                with gr.Row():
                    clear = gr.Button("🔄 New Session", variant="secondary")
                    
            with gr.Column(scale=1):
                gr.Markdown("### 📊 Session Info")
                info = gr.Markdown(chat_interface._get_info())
                refresh = gr.Button("🔃 Refresh Info")
                
                gr.Markdown("---")
                gr.Markdown("### ⚙️ Settings")
                gr.Markdown(f"**Persona:** {chat_interface.state.persona_id}")
                gr.Markdown(f"**Retriever:** Active")
        
        # Event handlers
        submit.click(
            chat_interface.chat,
            inputs=[msg, chatbot],
            outputs=[msg, chatbot]
        ).then(
            chat_interface._get_info,
            outputs=[info]
        )
        
        msg.submit(
            chat_interface.chat,
            inputs=[msg, chatbot],
            outputs=[msg, chatbot]
        ).then(
            chat_interface._get_info,
            outputs=[info]
        )
        
        clear.click(
            chat_interface.reset_chat,
            outputs=[chatbot, info]
        )
        
        refresh.click(
            chat_interface._get_info,
            outputs=[info]
        )
    
    return app


if __name__ == "__main__":
    try:
        app = create_app()
        app.launch(
            server_name="0.0.0.0",
            server_port=7860,
            theme=gr.themes.Soft(),
            share=False
        )
    except Exception as e:
        print(f"Error starting Gradio app: {e}")
        raise