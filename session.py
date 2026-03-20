"""
Session management for the FastAPI Chat API.
Provides in-memory storage for ConversationMemory per session.
"""

from datetime import datetime
from typing import Optional

from tool_agent import ConversationMemory, CostTracker
from security import SYSTEM_PROMPT_API


class SessionManager:
    """In-memory session storage with per-session ConversationMemory."""

    def __init__(self):
        # session_id -> {
        #     "memory": ConversationMemory,
        #     "cost_tracker": CostTracker,
        #     "created_at": datetime,
        #     "turn_count": int
        # }
        self.sessions: dict[str, dict] = {}

    def create_session(self, session_id: str) -> ConversationMemory:
        """
        Create a new session or return existing one.

        Args:
            session_id: Unique session identifier

        Returns:
            ConversationMemory for the session
        """
        if session_id in self.sessions:
            return self.sessions[session_id]["memory"]

        memory = ConversationMemory(system_prompt=SYSTEM_PROMPT_API)
        self.sessions[session_id] = {
            "memory": memory,
            "cost_tracker": memory.cost_tracker,
            "created_at": datetime.utcnow(),
            "turn_count": 0
        }
        return memory

    def get_session(self, session_id: str) -> Optional[ConversationMemory]:
        """
        Get existing session memory.

        Args:
            session_id: Session identifier

        Returns:
            ConversationMemory if session exists, None otherwise
        """
        if session_id not in self.sessions:
            return None
        return self.sessions[session_id]["memory"]

    def session_exists(self, session_id: str) -> bool:
        """Check if a session exists."""
        return session_id in self.sessions

    def delete_session(self, session_id: str) -> bool:
        """
        Delete a session.

        Args:
            session_id: Session to delete

        Returns:
            True if deleted, False if not found
        """
        if session_id in self.sessions:
            del self.sessions[session_id]
            return True
        return False

    def get_all_sessions(self) -> dict:
        """
        Get summary of all sessions (for admin/debug).

        Returns:
            Dict mapping session_id to session info
        """
        result = {}
        for sid, data in self.sessions.items():
            result[sid] = {
                "created_at": data["created_at"].isoformat(),
                "turn_count": data["turn_count"],
                "total_tokens": (
                    data["cost_tracker"].total_prompt_tokens +
                    data["cost_tracker"].total_completion_tokens
                ),
                "total_cost": data["cost_tracker"].total_cost
            }
        return result

    def increment_turn(self, session_id: str) -> int:
        """
        Increment turn count for a session.

        Args:
            session_id: Session identifier

        Returns:
            New turn count
        """
        if session_id in self.sessions:
            self.sessions[session_id]["turn_count"] += 1
            return self.sessions[session_id]["turn_count"]
        return 0

    def get_turn_count(self, session_id: str) -> int:
        """Get current turn count for a session."""
        if session_id not in self.sessions:
            return 0
        return self.sessions[session_id]["turn_count"]

    def get_session_tokens(self, session_id: str) -> int:
        """Get total tokens used by a session."""
        if session_id not in self.sessions:
            return 0
        tracker = self.sessions[session_id]["cost_tracker"]
        return tracker.total_prompt_tokens + tracker.total_completion_tokens

    def get_global_tokens(self) -> int:
        """Get total tokens across all sessions."""
        return sum(
            tracker.total_prompt_tokens + tracker.total_completion_tokens
            for data in self.sessions.values()
            for tracker in [data["cost_tracker"]]
        )

    def get_session_info(self, session_id: str) -> Optional[dict]:
        """Get detailed session information."""
        if session_id not in self.sessions:
            return None
        data = self.sessions[session_id]
        tracker = data["cost_tracker"]
        return {
            "session_id": session_id,
            "created_at": data["created_at"].isoformat(),
            "turn_count": data["turn_count"],
            "total_tokens": (
                tracker.total_prompt_tokens + tracker.total_completion_tokens
            ),
            "prompt_tokens": tracker.total_prompt_tokens,
            "completion_tokens": tracker.total_completion_tokens,
            "total_cost": tracker.total_cost,
            "message_count": len(data["memory"].messages)
        }


# Global session manager instance
session_manager = SessionManager()
