import os
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from dotenv import load_dotenv
from openai import OpenAI
from pinecone import Pinecone

# Load environment variables
load_dotenv()

# Initialize FastAPI app
app = FastAPI()

# Configure CORS (so your frontend can talk to this API)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"]
)



# Initialize OpenAI client
llm = OpenAI()


pc = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))
dense_index = pc.Index("llm-project")

# Store conversations - one per user
conversations = {}

# Request body structure 
class ChatMessage(BaseModel):
    message: str
    conversation_id: str = "default"

# Helper Functions

def rag(user_input):
    """Search Pinecone for relevant documentation"""
    results = dense_index.search(
        namespace="all-gross",
        query={
            "top_k": 3,
            "inputs": {"text": user_input}
        }
    )
    # ... return documentation string
    documentation = ""
    for hit in results['result']['hits']:
        fields = hit.get('fields')
        chunk_text = fields.get('chunk_text')
        documentation += chunk_text

    return documentation

def system_prompt():
    """Return the system prompt for the chatbot"""
    return {
        "role": "developer", 
        "content": """You ara an AI event planner who is knowledgeable about various events in Dallas. One such event is a Live Jazz Night in the music category; its a live performance featuring talented local jazz musicians."""
        }

def user_prompt(user_input, documentation):
    """"Return the user prompt with RAG context"""
    return {
        "role": "user", 
        "content": f"""Here are the excerpts from the official text:{documentation}. Use whatever info from the above text excerpts (and no other info) to answer the following query: {user_input}."""}

# Root endpoint
# API Endpoints
@app.get("/")
def index():
    return {
        "message": "Event Planner Chatbot API (with RAG)",
        "endpoints": {
            "POST /chat": "Send a message (uses RAG)",
            "GET /conversations/{id}": "Get conversation history",
            "DELETE /conversations/{id}": "Clear conversation"
        }
    }

# RAG Chat endpoint
@app.post("/chat")
def create(chat_message: ChatMessage):
    conversation_id = chat_message.conversation_id
    user_message = chat_message.message

    # Initialize conversation if new
    if conversation_id not in conversations:
        conversations[conversation_id] = [
            system_prompt(),
            {"role": "assistant", "content": "What would you like to do today?" }
        ]

    # Get relevant documentation via RAG
    documentation = rag(user_message)

    # Add user message with RAG content
    conversations[conversation_id].append(user_prompt(user_message, documentation))
    
    # Get response from LLM
    response = llm.responses.create(
        model="gpt-4.1-mini",
        temperature=0.5,
        input=conversations[conversation_id]
    )

    assistant_message = response.output_text

    # Add response to history
    conversations[conversation_id].append({
        "role": "assistant",
        "content": assistant_message
    })

    return {
        "message": assistant_message,
        "conversation_id": conversation_id
    }


# Get conversation history
@app.get("/conversations/{conversation_id}")
def show(conversation_id: str):
    if conversation_id not in conversations:
        return {"error": "Conversation not found"}
    
    return {
        "conversation_id": conversation_id,
        "history": conversations[conversation_id]
    }

# Clear conversation
@app.delete("/conversations/{conversation_id}")
def destroy(conversation_id: str):
    if conversation_id in conversations:
        del conversations[conversation_id]
        return {"message": "Conversation deleted"}
    
    return {"error": "Conversation not found"}


