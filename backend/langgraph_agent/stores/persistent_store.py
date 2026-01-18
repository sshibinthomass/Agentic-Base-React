"""
Persistent store implementation for Long Memory using SQLite.
This ensures memory persists across application restarts.
"""
import os
import json
import aiosqlite
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
from langgraph.store.base import BaseStore
from langgraph.store.memory import InMemoryStore


class SQLitePersistentStore(InMemoryStore):
    """
    A persistent store that uses SQLite to save data to disk.
    Extends InMemoryStore but adds persistence layer.
    """
    
    def __init__(
        self,
        db_path: str = "langmem_store.db",
        index: Optional[Dict[str, Any]] = None,
    ):
        """
        Initialize the persistent store.
        
        Args:
            db_path: Path to SQLite database file
            index: Index configuration for embeddings (same as InMemoryStore)
        """
        super().__init__(index=index)
        self.db_path = db_path
        self._initialized = False
    
    async def setup(self):
        """Initialize the database and load existing data."""
        if self._initialized:
            return
        
        # Ensure directory exists
        db_file = Path(self.db_path)
        db_file.parent.mkdir(parents=True, exist_ok=True)
        
        # Create database and table
        async with aiosqlite.connect(self.db_path) as db:
            await db.execute("""
                CREATE TABLE IF NOT EXISTS store_data (
                    namespace TEXT NOT NULL,
                    key TEXT NOT NULL,
                    value TEXT NOT NULL,
                    PRIMARY KEY (namespace, key)
                )
            """)
            await db.execute("""
                CREATE TABLE IF NOT EXISTS index_data (
                    namespace TEXT NOT NULL,
                    key TEXT NOT NULL,
                    vector BLOB NOT NULL,
                    metadata TEXT,
                    PRIMARY KEY (namespace, key)
                )
            """)
            await db.commit()
            
            # Load existing data into memory
            loaded_count = 0
            async with db.execute("SELECT namespace, key, value FROM store_data") as cursor:
                async for row in cursor:
                    namespace_str, key, value_str = row
                    namespace = tuple(json.loads(namespace_str))
                    value = json.loads(value_str)
                    # Use parent class method to store in memory
                    await super().aput(namespace, key, value)
                    loaded_count += 1
            print(f"[STORE] Loaded {loaded_count} existing memories from database")
            
            # Load index data
            async with db.execute("SELECT namespace, key, vector, metadata FROM index_data") as cursor:
                async for row in cursor:
                    namespace_str, key, vector_bytes, metadata_str = row
                    namespace = tuple(json.loads(namespace_str))
                    vector = json.loads(vector_bytes) if vector_bytes else None
                    metadata = json.loads(metadata_str) if metadata_str else None
                    if vector:
                        # Store in index (this would need to be implemented based on InMemoryStore internals)
                        pass
        
        self._initialized = True
    
    async def aput(self, namespace: Tuple[str, ...], key: str, value: Any) -> None:
        """Put a value in the store and persist to SQLite."""
        print(f"[STORE] Storing value with namespace: {namespace}, key: {key}")
        # Store in memory (parent class)
        await super().aput(namespace, key, value)
        
        # Persist to SQLite
        namespace_str = json.dumps(list(namespace))
        value_str = json.dumps(value)
        
        async with aiosqlite.connect(self.db_path) as db:
            await db.execute(
                "INSERT OR REPLACE INTO store_data (namespace, key, value) VALUES (?, ?, ?)",
                (namespace_str, key, value_str)
            )
            await db.commit()
        print(f"[STORE] Value persisted to database successfully")
    
    async def aget(self, namespace: Tuple[str, ...], key: str) -> Optional[Any]:
        """Get a value from the store."""
        print(f"[STORE] Getting value with namespace: {namespace}, key: {key}")
        # Try to get from memory first (parent class)
        value = await super().aget(namespace, key)
        if value is not None:
            print(f"[STORE] Value found in memory")
            return value
        
        # If not in memory, try to load from SQLite (in case it wasn't loaded during setup)
        if self._initialized:
            namespace_str = json.dumps(list(namespace))
            async with aiosqlite.connect(self.db_path) as db:
                async with db.execute(
                    "SELECT value FROM store_data WHERE namespace = ? AND key = ?",
                    (namespace_str, key)
                ) as cursor:
                    row = await cursor.fetchone()
                    if row:
                        value = json.loads(row[0])
                        # Store in memory for future access
                        await super().aput(namespace, key, value)
                        print(f"[STORE] Value loaded from database and cached in memory")
                        return value
                    else:
                        print(f"[STORE] Value not found in database")
        else:
            print(f"[STORE] Store not initialized, cannot load from database")
        return None
    
    async def adelete(self, namespace: Tuple[str, ...], key: str) -> None:
        """Delete a value from the store and SQLite."""
        await super().adelete(namespace, key)
        
        namespace_str = json.dumps(list(namespace))
        async with aiosqlite.connect(self.db_path) as db:
            await db.execute(
                "DELETE FROM store_data WHERE namespace = ? AND key = ?",
                (namespace_str, key)
            )
            await db.commit()
    
    async def alist(self, namespace: Tuple[str, ...]) -> List[str]:
        """List all keys in a namespace."""
        namespace_str = json.dumps(list(namespace))
        keys = []
        
        # Try to get from parent class first (if it has the method)
        try:
            if hasattr(super(), 'alist'):
                keys = await super().alist(namespace)
        except (AttributeError, TypeError):
            pass
        
        # Also query database to ensure we get all keys
        if self._initialized:
            async with aiosqlite.connect(self.db_path) as db:
                async with db.execute(
                    "SELECT key FROM store_data WHERE namespace = ?",
                    (namespace_str,)
                ) as cursor:
                    async for row in cursor:
                        key = row[0]
                        if key not in keys:
                            keys.append(key)
        
        print(f"[STORE] Listing keys for namespace {namespace}: found {len(keys)} keys")
        return keys


def create_persistent_store(
    db_path: Optional[str] = None,
    embed_model: str = "openai:text-embedding-3-small",
) -> SQLitePersistentStore:
    """
    Create a persistent store for long memory.
    
    Args:
        db_path: Path to SQLite database file. If None, uses default location.
        embed_model: Embedding model to use for vector search.
    
    Returns:
        SQLitePersistentStore instance
    """
    if db_path is None:
        # Default to a data directory in the project
        data_dir = Path(__file__).parent.parent.parent / "data"
        data_dir.mkdir(exist_ok=True)
        db_path = str(data_dir / "langmem_store.db")
    
    store = SQLitePersistentStore(
        db_path=db_path,
        index={
            "dims": 1536,  # OpenAI text-embedding-3-small dimension
            "embed": embed_model,
        }
    )
    return store
