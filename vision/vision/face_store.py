"""
SQLite database wrapper for storing face embeddings and person information.
"""

import sqlite3
import numpy as np
from pathlib import Path
from datetime import datetime
from typing import Optional, List, Tuple, Dict


class FaceStore:
    """
    SQLite database for face recognition system.
    
    Schema:
        persons(id, name, created_at)
        embeddings(id, person_id, embedding_blob, image_path, created_at)
    """
    
    def __init__(self, db_path: str):
        """
        Initialize face store database.
        
        Args:
            db_path: Path to SQLite database file
        """
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        
        self.conn = sqlite3.connect(str(self.db_path), check_same_thread=False)
        self.conn.row_factory = sqlite3.Row  # Enable column access by name
        
        self._create_tables()
    
    def _create_tables(self):
        """Create database tables if they don't exist."""
        cursor = self.conn.cursor()
        
        # Persons table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS persons (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                name TEXT NOT NULL UNIQUE,
                created_at TEXT NOT NULL
            )
        """)
        
        # Embeddings table (multiple embeddings per person for robustness)
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS embeddings (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                person_id INTEGER NOT NULL,
                embedding_blob BLOB NOT NULL,
                image_path TEXT,
                created_at TEXT NOT NULL,
                FOREIGN KEY (person_id) REFERENCES persons(id) ON DELETE CASCADE
            )
        """)
        
        # Create index for faster lookups
        cursor.execute("""
            CREATE INDEX IF NOT EXISTS idx_person_id 
            ON embeddings(person_id)
        """)
        
        self.conn.commit()
    
    def add_person(self, name: str) -> int:
        """
        Add a new person to database.
        
        Args:
            name: Person's name (must be unique)
        
        Returns:
            Person ID
        
        Raises:
            sqlite3.IntegrityError: If name already exists
        """
        cursor = self.conn.cursor()
        now = datetime.now().isoformat()
        
        cursor.execute(
            "INSERT INTO persons (name, created_at) VALUES (?, ?)",
            (name, now)
        )
        self.conn.commit()
        
        return cursor.lastrowid
    
    def get_person_id(self, name: str) -> Optional[int]:
        """
        Get person ID by name.
        
        Args:
            name: Person's name
        
        Returns:
            Person ID or None if not found
        """
        cursor = self.conn.cursor()
        cursor.execute("SELECT id FROM persons WHERE name = ?", (name,))
        row = cursor.fetchone()
        return row[0] if row else None
    
    def get_or_create_person(self, name: str) -> int:
        """
        Get existing person ID or create new person.
        
        Args:
            name: Person's name
        
        Returns:
            Person ID
        """
        person_id = self.get_person_id(name)
        if person_id is None:
            person_id = self.add_person(name)
        return person_id
    
    def add_embedding(self, person_id: int, embedding: np.ndarray, 
                     image_path: Optional[str] = None) -> int:
        """
        Add face embedding for a person.
        
        Args:
            person_id: Person ID
            embedding: 512-D embedding vector (float32)
            image_path: Optional path to face image
        
        Returns:
            Embedding ID
        """
        if embedding.dtype != np.float32:
            embedding = embedding.astype(np.float32)
        
        if embedding.size != 512:
            raise ValueError(f"Expected 512-D embedding, got {embedding.size}-D")
        
        # Convert to bytes
        embedding_bytes = embedding.tobytes()
        
        cursor = self.conn.cursor()
        now = datetime.now().isoformat()
        
        cursor.execute(
            "INSERT INTO embeddings (person_id, embedding_blob, image_path, created_at) "
            "VALUES (?, ?, ?, ?)",
            (person_id, embedding_bytes, image_path, now)
        )
        self.conn.commit()
        
        return cursor.lastrowid
    
    def get_all_embeddings(self) -> List[Tuple[int, str, np.ndarray]]:
        """
        Get all embeddings from database.
        
        Returns:
            List of (person_id, name, embedding) tuples
        """
        cursor = self.conn.cursor()
        cursor.execute("""
            SELECT p.id, p.name, e.embedding_blob
            FROM embeddings e
            JOIN persons p ON e.person_id = p.id
        """)
        
        results = []
        for row in cursor.fetchall():
            person_id, name, embedding_bytes = row
            embedding = np.frombuffer(embedding_bytes, dtype=np.float32)
            
            if embedding.size != 512:
                print(f"Warning: Skipping invalid embedding (size={embedding.size})")
                continue
            
            results.append((person_id, name, embedding))
        
        return results
    
    def find_match(self, query_embedding: np.ndarray, 
                   threshold: float = 0.4) -> Optional[Tuple[str, float]]:
        """
        Find best matching person for query embedding.
        
        Args:
            query_embedding: 512-D query embedding (L2-normalized)
            threshold: Minimum cosine similarity threshold
        
        Returns:
            (name, similarity) tuple or None if no match above threshold
        """
        if query_embedding.size != 512:
            raise ValueError(f"Expected 512-D embedding, got {query_embedding.size}-D")
        
        all_embeddings = self.get_all_embeddings()
        
        if not all_embeddings:
            return None
        
        best_name = None
        best_score = threshold
        
        # Group embeddings by person and average
        person_embeddings = {}
        for person_id, name, emb in all_embeddings:
            if name not in person_embeddings:
                person_embeddings[name] = []
            person_embeddings[name].append(emb)
        
        # Find best match using average embedding per person
        for name, embs in person_embeddings.items():
            # Average embeddings for this person
            avg_emb = np.mean(embs, axis=0)
            # Re-normalize
            avg_emb /= np.linalg.norm(avg_emb)
            
            # Cosine similarity (both are L2-normalized)
            similarity = np.dot(query_embedding, avg_emb)
            
            if similarity > best_score:
                best_score = similarity
                best_name = name
        
        return (best_name, best_score) if best_name else None
    
    def get_person_stats(self) -> List[Dict]:
        """
        Get statistics about registered persons.
        
        Returns:
            List of dicts with person info and embedding counts
        """
        cursor = self.conn.cursor()
        cursor.execute("""
            SELECT p.id, p.name, p.created_at, COUNT(e.id) as embedding_count
            FROM persons p
            LEFT JOIN embeddings e ON p.id = e.person_id
            GROUP BY p.id
            ORDER BY p.name
        """)
        
        results = []
        for row in cursor.fetchall():
            results.append({
                'id': row[0],
                'name': row[1],
                'created_at': row[2],
                'embedding_count': row[3]
            })
        
        return results
    
    def delete_person(self, person_id: int):
        """
        Delete a person and all their embeddings.
        
        Args:
            person_id: Person ID to delete
        """
        cursor = self.conn.cursor()
        cursor.execute("DELETE FROM persons WHERE id = ?", (person_id,))
        self.conn.commit()
    
    def close(self):
        """Close database connection."""
        self.conn.close()
    
    def __enter__(self):
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()

