# code/router/route_v1.py
"""
FastAPI Router
API endpoints for Thai Regulatory AI
"""

import datetime
import logging
import uuid
from typing import Optional

from fastapi import APIRouter, HTTPException, status
from pydantic import BaseModel, Field

from adapter.response.response_custom import HandleSuccess
from model.conversation_state import ConversationState
from model.state_manager import StateManager
from model.persona_supervisor import PersonaSupervisor

import conf

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

api_v1 = APIRouter()


class ChatRequest(BaseModel):
    message: str = Field(..., description="User message to chatbot")
    session_id: Optional[str] = Field(None, description="Session ID for conversation continuity")


logger.info("Initializing services...")

try:
    if conf.USE_ZILLIZ:
        from service.vector_store import VectorStoreManager
        vs_manager = VectorStoreManager()
        retriever = vs_manager.connect_to_existing()
        logger.info("Using Milvus/Zilliz retriever")
    else:
        from service.local_vector_store import get_retriever
        retriever = get_retriever(fail_if_empty=False)
        logger.info("Using local Chroma retriever")

    supervisor = PersonaSupervisor(retriever=retriever)

    # StateManager now uses a stable default dir (code/data/states) unless overridden by conf.STATE_DIR
    state_manager = StateManager()

    logger.info("Services initialized successfully")

except Exception:
    logger.error("Failed to initialize services", exc_info=True)
    supervisor = None
    state_manager = None
    raise


@api_v1.get("/healthcheck")
async def health_check():
    return {
        "status": "ok",
        "timestamp": datetime.datetime.now().isoformat(),
        "service": "Thai Regulatory AI - น้องโคโค่",
        "version": "1.0.0",
        "supervisor_initialized": supervisor is not None,
        "state_manager_initialized": state_manager is not None,
        "use_zilliz": conf.USE_ZILLIZ,
        "collection_name": conf.COLLECTION_NAME,
    }


@api_v1.post("/chat")
async def chat(request: ChatRequest):
    if supervisor is None or state_manager is None:
        logger.error("Services not initialized")
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Services not initialized. Check server logs.",
        )

    if not request.message or not request.message.strip():
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Message cannot be empty",
        )

    session_id = request.session_id or f"s_{uuid.uuid4().hex[:8]}"

    try:
        logger.info(f"[{session_id}] User: {request.message[:120]}")

        saved = state_manager.load(session_id)
        state = saved if saved else ConversationState(session_id=session_id, persona_id="practical", context={})

        state, bot_reply = supervisor.handle(state, request.message)

        state_manager.save(session_id, state)

        logger.info(f"[{session_id}] Bot: {bot_reply[:120]}")

        return HandleSuccess(
            message="Chat completed",
            response=bot_reply,
            session_id=session_id,
        )

    except Exception as e:
        logger.error(f"[{session_id}] Chat failed", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Chat failed: {str(e)}",
        )
