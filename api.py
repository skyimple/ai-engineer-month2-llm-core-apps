"""
FastAPI production-grade multi-turn chat API.
Wraps tool_agent.py with HTTP endpoints, security, and streaming support.
"""

import json
import sys
from datetime import datetime
from typing import AsyncGenerator

from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect, Response
from fastapi.responses import JSONResponse, StreamingResponse
from pydantic import BaseModel

from tool_agent import (
    ConversationMemory,
    CostTracker,
    TOOLS,
    execute_tool,
    create_chat_completion,
    process_streaming_response,
    DEEPSEEK_PRICING,
)
from session import session_manager
from security import (
    injection_shield,
    rate_limiter,
    BudgetEnforcer,
    SYSTEM_PROMPT_API,
    SESSION_TOKEN_BUDGET,
    GLOBAL_TOKEN_BUDGET,
    SESSION_MAX_TURNS,
)


# =============================================================================
# Pydantic Models
# =============================================================================

def wrap_user_input(text: str) -> str:
    """
    Wrap user input in XML tags to visually/semantically isolate it
    from system instructions. This helps prevent injection attacks
    where user input might try to contain instruction-like content.
    """
    return f"<user_input>\n{text}\n</user_input>"


class ChatMessage(BaseModel):
    message: str


class ChatResponse(BaseModel):
    session_id: str
    message: str
    tokens_used: int | None = None
    cost: float | None = None


class SessionInfo(BaseModel):
    session_id: str
    created_at: str
    turn_count: int
    total_tokens: int
    prompt_tokens: int
    completion_tokens: int
    total_cost: float
    message_count: int


class ErrorResponse(BaseModel):
    error: str
    detail: str | None = None


# =============================================================================
# FastAPI App
# =============================================================================

app = FastAPI(
    title="Tool Chat API",
    description="Production-grade multi-turn chat API with tool calling support",
    version="1.0.0",
)

# Budget enforcer with session manager callbacks
budget_enforcer = BudgetEnforcer(
    get_session_tokens=session_manager.get_session_tokens,
    get_global_tokens=session_manager.get_global_tokens
)


# =============================================================================
# Exception Classes
# =============================================================================

class RateLimitExceeded(HTTPException):
    def __init__(self, detail: str):
        super().__init__(status_code=429, detail=detail)


class BudgetExceeded(HTTPException):
    def __init__(self, detail: str):
        super().__init__(status_code=402, detail=detail)


class InjectionDetected(HTTPException):
    def __init__(self, pattern: str):
        super().__init__(
            status_code=400,
            detail=f"Injection attempt detected: pattern '{pattern}'"
        )


class SessionNotFound(HTTPException):
    def __init__(self, session_id: str):
        super().__init__(status_code=404, detail=f"Session '{session_id}' not found")


# =============================================================================
# SSE Event Helpers
# =============================================================================

def sse_event(event_type: str, data: str | dict) -> str:
    """Create an SSE-formatted event."""
    if isinstance(data, dict):
        data = json.dumps(data)
    return f"event: {event_type}\ndata: {data}\n\n"


async def sse_stream(events: AsyncGenerator[tuple[str, str | dict], None]) -> AsyncGenerator[bytes, None]:
    """Convert async generator of events to SSE bytes."""
    async for event_type, data in events:
        yield sse_event(event_type, data).encode("utf-8")


# =============================================================================
# Streaming Chat Implementation
# =============================================================================

async def stream_chat(
    session_id: str,
    user_message: str
) -> AsyncGenerator[tuple[str, str | dict], None]:
    """
    Async generator yielding SSE-formatted chat events.

    Yields:
        Tuple of (event_type, data)
    """
    memory = session_manager.get_session(session_id)
    if not memory:
        yield "error", {"error": "Session not found"}
        return

    # Add user message to memory (wrapped in XML tags for injection protection)
    memory.add_user(wrap_user_input(user_message))

    # Track tool call state
    tool_calls_executed = 0
    max_tool_calls = 10  # Prevent infinite tool call loops

    while tool_calls_executed < max_tool_calls:
        # Create streaming completion
        try:
            stream = create_chat_completion(
                messages=memory.messages,
                tools=TOOLS,
                tool_choice="auto"
            )

            # Process streaming response
            full_content, tool_calls, usage = process_streaming_response(stream)

            # Update cost tracking
            if usage:
                memory.update_usage({
                    "prompt_tokens": usage.prompt_tokens,
                    "completion_tokens": usage.completion_tokens
                })
                total_tokens = usage.prompt_tokens + usage.completion_tokens
                yield "usage", {
                    "prompt_tokens": usage.prompt_tokens,
                    "completion_tokens": usage.completion_tokens,
                    "total_tokens": total_tokens
                }

            # Build assistant message with content and/or tool_calls
            if tool_calls:
                assistant_tool_calls = [
                    {
                        "id": tc["id"],
                        "type": "function",
                        "function": {
                            "name": tc["name"],
                            "arguments": json.dumps(tc["arguments"])
                        }
                    }
                    for tc in tool_calls
                ]
                memory.messages.append({
                    "role": "assistant",
                    "tool_calls": assistant_tool_calls
                })
            elif full_content:
                memory.add_assistant(full_content)

            # Handle tool calls
            if tool_calls:
                for tc in tool_calls:
                    tool_calls_executed += 1
                    yield "tool_call", {
                        "id": tc["id"],
                        "name": tc["name"],
                        "arguments": tc["arguments"]
                    }

                    # Execute tool
                    result = execute_tool(tc["name"], tc["arguments"])
                    yield "tool_result", {
                        "tool_call_id": tc["id"],
                        "result": result
                    }

                    # Add tool result to memory
                    memory.add_tool_result(tc["id"], result)
            else:
                # No more tool calls, we're done
                break

        except Exception as e:
            yield "error", {"error": str(e)}
            # Remove failed user message
            if memory.messages and memory.messages[-1]["role"] == "user":
                memory.messages.pop()
            return

    # Send final message event with the last assistant response
    if memory.messages:
        last_msg = memory.messages[-1]
        if last_msg["role"] == "assistant":
            content = last_msg.get("content", "")
            if content:
                yield "message", content

    yield "done", {"turn_count": session_manager.get_turn_count(session_id)}


# =============================================================================
# API Endpoints
# =============================================================================

@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {
        "status": "healthy",
        "timestamp": datetime.utcnow().isoformat(),
        "sessions": len(session_manager.sessions),
        "global_tokens": session_manager.get_global_tokens()
    }


@app.get("/sessions")
async def list_sessions():
    """List all sessions (admin/debug)."""
    return {
        "sessions": session_manager.get_all_sessions(),
        "global_tokens": session_manager.get_global_tokens()
    }


@app.get("/sessions/{session_id}")
async def get_session(session_id: str):
    """Get session information."""
    info = session_manager.get_session_info(session_id)
    if not info:
        raise SessionNotFound(session_id)
    return SessionInfo(**info)


@app.delete("/sessions/{session_id}")
async def delete_session(session_id: str):
    """Delete/clear a session."""
    if not session_manager.delete_session(session_id):
        raise SessionNotFound(session_id)
    return {"status": "deleted", "session_id": session_id}


@app.post("/chat")
async def chat(session_id: str, message: ChatMessage) -> StreamingResponse:
    """
    Send a message and receive a streaming SSE response.

    Use session_id to maintain conversation context across requests.
    """
    # Rate limiting
    allowed, error = rate_limiter.check(session_id)
    if not allowed:
        raise RateLimitExceeded(error)

    # Injection check
    is_blocked, pattern = injection_shield.check(message.message)
    if is_blocked:
        raise InjectionDetected(pattern)

    # Create session if doesn't exist
    session_manager.create_session(session_id)

    # Check budget
    budget = budget_enforcer.check(session_id)
    if not budget.allowed:
        raise BudgetExceeded(budget.reason)

    # Check max turns
    turn_count = session_manager.get_turn_count(session_id)
    if turn_count >= SESSION_MAX_TURNS:
        raise BudgetExceeded(f"Maximum turns ({SESSION_MAX_TURNS}) exceeded for session")

    # Increment turn counter
    session_manager.increment_turn(session_id)

    # Stream response
    async def event_generator():
        async for event_type, data in stream_chat(session_id, message.message):
            if event_type == "error":
                yield sse_event("error", data)
                break
            elif event_type == "done":
                yield sse_event("done", data)
            else:
                yield sse_event(event_type, data)

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no"
        }
    )


@app.websocket("/chat/stream/{session_id}")
async def websocket_chat(websocket: WebSocket, session_id: str):
    """
    WebSocket endpoint for streaming chat.

    Send JSON: {"message": "your message"}
    Receive JSON events via WebSocket messages
    """
    await websocket.accept()

    try:
        while True:
            # Receive message
            data = await websocket.receive_json()
            message = data.get("message", "")

            if not message:
                await websocket.send_json({"error": "Empty message"})
                continue

            # Injection check
            is_blocked, pattern = injection_shield.check(message)
            if is_blocked:
                await websocket.send_json({
                    "type": "error",
                    "error": f"Injection attempt detected: {pattern}"
                })
                continue

            # Rate limiting
            allowed, error = rate_limiter.check(session_id)
            if not allowed:
                await websocket.send_json({"type": "error", "error": error})
                continue

            # Create session if needed
            session_manager.create_session(session_id)

            # Check budget
            budget = budget_enforcer.check(session_id)
            if not budget.allowed:
                await websocket.send_json({
                    "type": "error",
                    "error": budget.reason
                })
                continue

            # Check max turns
            turn_count = session_manager.get_turn_count(session_id)
            if turn_count >= SESSION_MAX_TURNS:
                await websocket.send_json({
                    "type": "error",
                    "error": f"Maximum turns ({SESSION_MAX_TURNS}) exceeded"
                })
                continue

            session_manager.increment_turn(session_id)

            # Process chat (wrapped in XML tags for injection protection)
            memory = session_manager.get_session(session_id)
            memory.add_user(wrap_user_input(message))

            # Send thinking indicator
            await websocket.send_json({"type": "thinking"})

            # Process and send tool calls + responses
            tool_calls_executed = 0
            max_tool_calls = 10

            while tool_calls_executed < max_tool_calls:
                try:
                    stream = create_chat_completion(
                        messages=memory.messages,
                        tools=TOOLS,
                        tool_choice="auto"
                    )

                    full_content, tool_calls, usage = process_streaming_response(stream)

                    if usage:
                        memory.update_usage({
                            "prompt_tokens": usage.prompt_tokens,
                            "completion_tokens": usage.completion_tokens
                        })
                        await websocket.send_json({
                            "type": "usage",
                            "prompt_tokens": usage.prompt_tokens,
                            "completion_tokens": usage.completion_tokens,
                            "total_tokens": usage.prompt_tokens + usage.completion_tokens
                        })

                    # Add assistant message
                    if tool_calls:
                        assistant_tool_calls = [
                            {
                                "id": tc["id"],
                                "type": "function",
                                "function": {
                                    "name": tc["name"],
                                    "arguments": json.dumps(tc["arguments"])
                                }
                            }
                            for tc in tool_calls
                        ]
                        memory.messages.append({
                            "role": "assistant",
                            "tool_calls": assistant_tool_calls
                        })
                    elif full_content:
                        memory.add_assistant(full_content)
                        await websocket.send_json({
                            "type": "message",
                            "content": full_content
                        })

                    # Handle tool calls
                    if tool_calls:
                        for tc in tool_calls:
                            tool_calls_executed += 1
                            await websocket.send_json({
                                "type": "tool_call",
                                "id": tc["id"],
                                "name": tc["name"],
                                "arguments": tc["arguments"]
                            })

                            result = execute_tool(tc["name"], tc["arguments"])
                            await websocket.send_json({
                                "type": "tool_result",
                                "tool_call_id": tc["id"],
                                "result": result
                            })

                            memory.add_tool_result(tc["id"], result)
                    else:
                        break

                except Exception as e:
                    await websocket.send_json({
                        "type": "error",
                        "error": str(e)
                    })
                    if memory.messages and memory.messages[-1]["role"] == "user":
                        memory.messages.pop()
                    break

            await websocket.send_json({"type": "done"})

    except WebSocketDisconnect:
        pass
    except Exception as e:
        try:
            await websocket.send_json({"type": "error", "error": str(e)})
        except:
            pass


# =============================================================================
# Global Exception Handler
# =============================================================================

@app.exception_handler(Exception)
async def global_exception_handler(request, exc):
    """Global exception handler for unhandled errors."""
    return JSONResponse(
        status_code=500,
        content={"error": "Internal server error", "detail": str(exc)}
    )


# =============================================================================
# Startup Event
# =============================================================================

@app.on_event("startup")
async def startup_event():
    """Run on application startup."""
    print("=" * 60)
    print("Tool Chat API Server")
    print("=" * 60)
    print("Docs: http://127.0.0.1:8000/docs")
    print("WebSocket: ws://127.0.0.1:8000/chat/stream/{session_id}")
    print("=" * 60)


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000, reload=True)
