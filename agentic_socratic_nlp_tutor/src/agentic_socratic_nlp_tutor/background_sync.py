"""
Background Sync for Knowledge Graph

Periodically runs the extract_topics_from_rag.py script to sync concepts
from RAG to the knowledge graph.

This runs as a background task and doesn't block the main application.
"""

import asyncio
import logging
import os
import subprocess
import sys
from pathlib import Path
from typing import Optional
from datetime import datetime, timedelta

logger = logging.getLogger(__name__)

logger = logging.getLogger(__name__)


class BackgroundSync:
    """
    Background sync manager for knowledge graph concepts.
    
    Periodically runs the extract_topics_from_rag.py script to discover
    and add new concepts from RAG to the knowledge graph.
    """
    
    def __init__(
        self,
        sync_interval_hours: int = 24,
        enabled: bool = True,
        script_path: Optional[str] = None
    ):
        """
        Initialize background sync.
        
        Args:
            sync_interval_hours: Hours between sync runs (default: 24)
            enabled: Whether sync is enabled (default: True)
            script_path: Path to extract_topics_from_rag.py (auto-detected if None)
        """
        self.sync_interval_hours = sync_interval_hours
        self.enabled = enabled and os.getenv("BACKGROUND_SYNC_ENABLED", "true").lower() == "true"
        self.script_path = script_path or self._find_extract_script()
        self.last_sync: Optional[datetime] = None
        self.sync_task: Optional[asyncio.Task] = None
        self.running = False
        
        if not self.script_path:
            logger.warning("⚠️ [BackgroundSync] extract_topics_from_rag.py not found - background sync disabled")
            self.enabled = False
    
    def _find_extract_script(self) -> Optional[str]:
        """Find the knowledge graph sync script.
        
        Prefers backend wrapper script to avoid environment conflicts.
        Falls back to original script if wrapper doesn't exist.
        """
        # Try multiple possible locations
        current_file = Path(__file__).resolve()
        project_root = current_file.parent.parent.parent.parent
        
        # Prefer backend wrapper script (runs in backend environment)
        possible_paths = [
            project_root / "backend" / "scripts" / "sync_knowledge_graph.py",  # Backend wrapper (preferred)
            project_root / "scripts" / "extract_topics_from_rag.py",  # Original script (fallback)
            project_root / "backend" / "extract_topics_from_rag.py",
            project_root / "extract_topics_from_rag.py",
        ]
        
        for path in possible_paths:
            if path.exists():
                logger.info(f"📁 [BackgroundSync] Using script: {path}")
                return str(path)
        
        return None
    
    async def start(self):
        """Start the background sync task."""
        if not self.enabled:
            logger.info("📚 [BackgroundSync] Background sync is disabled")
            return
        
        if self.running:
            logger.warning("⚠️ [BackgroundSync] Sync already running")
            return
        
        self.running = True
        logger.info(f"🔄 [BackgroundSync] Starting background sync (interval: {self.sync_interval_hours}h)")
        
        # Run initial sync after a short delay
        await asyncio.sleep(60)  # Wait 1 minute before first sync
        
        # Start the sync loop
        self.sync_task = asyncio.create_task(self._sync_loop())
    
    async def stop(self):
        """Stop the background sync task."""
        self.running = False
        if self.sync_task:
            self.sync_task.cancel()
            try:
                await self.sync_task
            except asyncio.CancelledError:
                pass
        logger.info("🛑 [BackgroundSync] Background sync stopped")
    
    async def _sync_loop(self):
        """Main sync loop - runs periodically."""
        while self.running:
            try:
                await self._run_sync()
                
                # Wait for next sync interval
                await asyncio.sleep(self.sync_interval_hours * 3600)
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"❌ [BackgroundSync] Error in sync loop: {e}")
                # Wait 1 hour before retrying on error
                await asyncio.sleep(3600)
    
    async def _run_sync(self):
        """Run a single sync operation."""
        if not self.script_path:
            return
        
        logger.info("🔄 [BackgroundSync] Starting knowledge graph sync...")
        start_time = datetime.now()
        
        try:
            # Get script directory for working directory
            script_path_obj = Path(self.script_path)
            
            # Determine working directory based on script location
            if "backend" in script_path_obj.parts and "scripts" in script_path_obj.parts:
                # Backend wrapper script: use backend/scripts as working directory
                # The wrapper will handle project root and .env loading
                script_dir = script_path_obj.parent  # backend/scripts
            elif "scripts" in script_path_obj.parts:
                # Original script: use project root
                script_dir = script_path_obj.parent.parent  # project root
            else:
                # Fallback
                current_file = Path(__file__).resolve()
                script_dir = current_file.parent.parent.parent.parent
            
            # Get environment variables to pass to subprocess
            env = os.environ.copy()
            
            # Run the script in a subprocess
            # The backend wrapper handles environment setup, so we just need to run it
            # with proper encoding for Windows compatibility
            
            result = await asyncio.to_thread(
                subprocess.run,
                [sys.executable, self.script_path, "--auto-confirm"],
                cwd=str(script_dir),  # Set working directory
                env=env,  # Pass environment variables
                capture_output=True,
                text=True,
                encoding='utf-8',  # Explicit UTF-8 encoding for Windows
                errors='replace',  # Replace invalid characters instead of failing
                timeout=3600  # 1 hour timeout
            )
            
            if result.returncode == 0:
                self.last_sync = datetime.now()
                duration = (self.last_sync - start_time).total_seconds()
                logger.info(f"✅ [BackgroundSync] Sync completed in {duration:.1f}s")
                
                # Log summary if available
                if result.stdout:
                    # Extract summary from output
                    lines = result.stdout.split('\n')
                    for line in lines[-10:]:  # Last 10 lines often contain summary
                        if "topics" in line.lower() or "concepts" in line.lower():
                            logger.info(f"📊 [BackgroundSync] {line.strip()}")
            else:
                logger.warning(f"⚠️ [BackgroundSync] Sync failed with return code {result.returncode}")
                if result.stderr:
                    # Show more of the error for debugging
                    error_lines = result.stderr.split('\n')
                    logger.warning(f"⚠️ [BackgroundSync] Error output:")
                    for line in error_lines[-20:]:  # Last 20 lines
                        if line.strip():
                            logger.warning(f"   {line}")
                if result.stdout:
                    # Also check stdout for errors
                    stdout_lines = result.stdout.split('\n')
                    for line in stdout_lines[-10:]:
                        if 'error' in line.lower() or 'exception' in line.lower() or 'traceback' in line.lower():
                            logger.warning(f"⚠️ [BackgroundSync] {line.strip()}")
        
        except subprocess.TimeoutExpired:
            logger.error("❌ [BackgroundSync] Sync timed out after 1 hour")
        except Exception as e:
            logger.error(f"❌ [BackgroundSync] Sync error: {e}")
            import traceback
            logger.error(f"❌ [BackgroundSync] Traceback: {traceback.format_exc()}")
    
    async def sync_now(self):
        """Manually trigger a sync immediately."""
        if not self.enabled:
            logger.warning("⚠️ [BackgroundSync] Background sync is disabled")
            return
        
        logger.info("🔄 [BackgroundSync] Manual sync triggered")
        await self._run_sync()
    
    def get_status(self) -> dict:
        """Get sync status."""
        return {
            "enabled": self.enabled,
            "running": self.running,
            "last_sync": self.last_sync.isoformat() if self.last_sync else None,
            "next_sync_in_hours": self.sync_interval_hours,
            "script_path": self.script_path
        }

