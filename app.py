"""
AI Study Bot - FastAPI Application
A complete AI-powered Study Assistant chatbot with MongoDB memory
"""

import os
from datetime import datetime
from typing import List, Dict
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException, status
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from pymongo import MongoClient
from pymongo.errors import ConnectionFailure, ServerSelectionTimeoutError
from langchain_groq import ChatGroq
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage

# Load environment variables from .env file
load_dotenv()

# ==============================
# CONFIGURATION & VALIDATION
# ==============================

AI_API_KEY = os.getenv("AI_API_KEY")
MONGODB_URI = os.getenv("MONGODB_URI")
MODEL_NAME = os.getenv("MODEL_NAME", "llama3-8b-8192")

if not AI_API_KEY:
    raise ValueError("AI_API_KEY is missing in .env file")
if not MONGODB_URI:
    raise ValueError("MONGODB_URI is missing in .env file")

# ==============================
# DATABASE CONNECTION
# ==============================

try:
    # Connect to MongoDB Atlas
    mongo_client = MongoClient(MONGODB_URI, serverSelectionTimeoutMS=5000)
    # Test the connection
    mongo_client.admin.command('ping')
    db = mongo_client["studybot"]
    chats_collection = db["chats"]
    print("✅ Successfully connected to MongoDB!")
except (ConnectionFailure, ServerSelectionTimeoutError) as e:
    print(f"❌ Failed to connect to MongoDB: {e}")
    raise

# ==============================
# LLM SETUP
# ==============================

# System prompt for study assistant behavior
SYSTEM_PROMPT = """You are a helpful AI Study Assistant. Your purpose is to assist students with academic and educational questions.

Guidelines:
- Answer questions related to academics, homework, studying, learning, education, and general knowledge
- Provide clear, accurate, and helpful explanations
- If asked non-academic questions, politely redirect the user to educational topics
- Be encouraging and supportive
- Break down complex topics into simpler explanations when needed"""

try:
    llm = ChatGroq(api_key=AI_API_KEY, model_name=MODEL_NAME, temperature=0.7)
    print(f"✅ LLM initialized with model: {MODEL_NAME}")
except Exception as e:
    print(f"❌ Failed to initialize LLM: {e}")
    raise

# ==============================
# FASTAPI APPLICATION
# ==============================

app = FastAPI(
    title="AI Study Bot",
    description="An AI-powered Study Assistant with conversation memory",
    version="1.0.0"
)

# Add CORS middleware for frontend compatibility
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ==============================
# PYDANTIC MODELS
# ==============================

class ChatRequest(BaseModel):
    """Request model for chat endpoint"""
    user_id: str = Field(..., description="Unique identifier for the user")
    message: str = Field(..., description="User's message/question")

class ChatResponse(BaseModel):
    """Response model for chat endpoint"""
    response: str = Field(..., description="AI assistant's response")

# ==============================
# HELPER FUNCTIONS
# ==============================

def get_conversation_history(user_id: str, limit: int = 10) -> List[Dict]:
    """
    Fetch the last N messages for a specific user from MongoDB
    
    Args:
        user_id: The user's unique identifier
        limit: Number of recent messages to fetch (default: 10)
    
    Returns:
        List of message dictionaries with role and message
    """
    try:
        messages = chats_collection.find(
            {"user_id": user_id}
        ).sort("timestamp", -1).limit(limit)
        
        # Reverse to get chronological order (oldest first)
        messages_list = list(messages)[::-1]
        return messages_list
    except Exception as e:
        print(f"Error fetching conversation history: {e}")
        return []

def save_message(user_id: str, role: str, message: str):
    """
    Save a message to MongoDB
    
    Args:
        user_id: The user's unique identifier
        role: Either "user" or "assistant"
        message: The message content
    """
    try:
        chats_collection.insert_one({
            "user_id": user_id,
            "role": role,
            "message": message,
            "timestamp": datetime.utcnow()
        })
    except Exception as e:
        print(f"Error saving message: {e}")
        raise

def build_langchain_messages(history: List[Dict], current_message: str) -> List:
    """
    Convert MongoDB history to LangChain message format
    
    Args:
        history: List of previous messages from database
        current_message: The current user message
    
    Returns:
        List of LangChain message objects
    """
    messages = [SystemMessage(content=SYSTEM_PROMPT)]
    
    # Add conversation history
    for msg in history:
        if msg["role"] == "user":
            messages.append(HumanMessage(content=msg["message"]))
        elif msg["role"] == "assistant":
            messages.append(AIMessage(content=msg["message"]))
    
    # Add current user message
    messages.append(HumanMessage(content=current_message))
    
    return messages

# ==============================
# API ENDPOINTS
# ==============================

@app.get("/")
async def root():
    """Health check endpoint"""
    return {
        "status": "online",
        "service": "AI Study Bot",
        "version": "1.0.0",
        "message": "Send POST requests to /chat to interact with the bot"
    }

@app.post("/chat", response_model=ChatResponse)
async def chat(request: ChatRequest):
    """
    Main chat endpoint - handles user messages and returns AI responses
    
    Args:
        request: ChatRequest containing user_id and message
    
    Returns:
        ChatResponse with the AI's response
    """
    try:
        # Validate input
        if not request.user_id or not request.message:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="user_id and message are required"
            )
        
        # Get conversation history for context
        history = get_conversation_history(request.user_id, limit=10)
        
        # Build message list for LLM with context
        messages = build_langchain_messages(history, request.message)
        
        # Get response from LLM
        try:
            response = llm.invoke(messages)
            ai_response = response.content
        except Exception as e:
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=f"LLM API error: {str(e)}"
            )
        
        # Save user message to database
        save_message(request.user_id, "user", request.message)
        
        # Save assistant response to database
        save_message(request.user_id, "assistant", ai_response)
        
        return ChatResponse(response=ai_response)
    
    except HTTPException as he:
        raise he
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Internal server error: {str(e)}"
        )

@app.get("/history/{user_id}")
async def get_history(user_id: str, limit: int = 20):
    """
    Get conversation history for a specific user
    
    Args:
        user_id: The user's unique identifier
        limit: Number of messages to retrieve (default: 20)
    
    Returns:
        List of messages with timestamps
    """
    try:
        history = get_conversation_history(user_id, limit)
        return {
            "user_id": user_id,
            "message_count": len(history),
            "messages": [
                {
                    "role": msg["role"],
                    "message": msg["message"],
                    "timestamp": msg["timestamp"].isoformat()
                }
                for msg in history
            ]
        }
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error fetching history: {str(e)}"
        )

@app.delete("/history/{user_id}")
async def clear_history(user_id: str):
    """
    Clear conversation history for a specific user
    
    Args:
        user_id: The user's unique identifier
    
    Returns:
        Confirmation message with count of deleted messages
    """
    try:
        result = chats_collection.delete_many({"user_id": user_id})
        return {
            "status": "success",
            "message": f"Cleared {result.deleted_count} messages for user {user_id}"
        }
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error clearing history: {str(e)}"
        )

# ==============================
# APPLICATION STARTUP
# ==============================

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app:app", host="0.0.0.0", port=8000, reload=True)