import sqlite3
import numpy as np
from typing import List, Tuple, Optional
from datetime import datetime
import os


class FaceStore:
    """
    SQLite-based storage for face embeddings and person data.
    """
    
    def __init__(self, db_path: str = "data/db/faces.db"):
        """
        Initialize face storage.
        
        Args:
            db_path: Path to SQLite database file
        """
        self.db_path = db_path
        
        # Ensure directory exists
        os.makedirs(os.path.dirname(db_path), exist_ok=True)
        
        # Initialize database
        self._init_db()
    
    def _init_db(self):
        """Create tables if they don't exist."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Create persons table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS persons (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                name TEXT UNIQUE NOT NULL,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)
        
        # Create face_embeddings table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS face_embeddings (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                person_id INTEGER NOT NULL,
                embedding BLOB NOT NULL,
                capture_timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (person_id) REFERENCES persons(id) ON DELETE CASCADE
            )
        """)
        
        conn.commit()
        conn.close()
    
    def register_person(self, name: str, embeddings: List[np.ndarray]) -> int:
        """
        Register a new person with their face embeddings.
        
        Args:
            name: Person's name (must be unique)
            embeddings: List of face embedding vectors
            
        Returns:
            Person ID
            
        Raises:
            ValueError: If name already exists
        """
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        try:
            # Insert person
            cursor.execute(
                "INSERT INTO persons (name, created_at) VALUES (?, ?)",
                (name, datetime.now())
            )
            person_id = cursor.lastrowid
            
            # Insert all embeddings
            for embedding in embeddings:
                embedding_bytes = embedding.tobytes()
                cursor.execute(
                    "INSERT INTO face_embeddings (person_id, embedding, capture_timestamp) VALUES (?, ?, ?)",
                    (person_id, embedding_bytes, datetime.now())
                )
            
            conn.commit()
            return person_id
            
        except sqlite3.IntegrityError:
            conn.rollback()
            raise ValueError(f"Person with name '{name}' already exists")
        finally:
            conn.close()
    
    def find_match(self, embedding: np.ndarray, threshold: float = 0.45) -> Optional[Tuple[str, float]]:
        """
        Find best matching person for given embedding.
        
        Args:
            embedding: Face embedding to match
            threshold: Minimum similarity threshold
            
        Returns:
            Tuple of (name, similarity) if match found, None otherwise
        """
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Get all persons and their average embeddings
        cursor.execute("""
            SELECT p.name, fe.embedding
            FROM persons p
            JOIN face_embeddings fe ON p.id = fe.person_id
        """)
        
        best_match = None
        best_similarity = threshold
        
        # Group embeddings by person and find best match
        person_embeddings = {}
        for row in cursor.fetchall():
            name, embedding_bytes = row
            stored_embedding = np.frombuffer(embedding_bytes, dtype=np.float32)
            
            if name not in person_embeddings:
                person_embeddings[name] = []
            person_embeddings[name].append(stored_embedding)
        
        conn.close()
        
        # Calculate average similarity for each person
        for name, embeddings_list in person_embeddings.items():
            # Calculate similarity with each stored embedding
            similarities = [np.dot(embedding, stored_emb) for stored_emb in embeddings_list]
            avg_similarity = np.mean(similarities)
            
            if avg_similarity > best_similarity:
                best_similarity = avg_similarity
                best_match = (name, avg_similarity)
        
        return best_match
    
    def get_person(self, name: str) -> Optional[dict]:
        """
        Get person information by name.
        
        Args:
            name: Person's name
            
        Returns:
            Dictionary with person info, or None if not found
        """
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute(
            "SELECT id, name, created_at FROM persons WHERE name = ?",
            (name,)
        )
        row = cursor.fetchone()
        
        if row is None:
            conn.close()
            return None
        
        person_id, name, created_at = row
        
        # Get number of embeddings
        cursor.execute(
            "SELECT COUNT(*) FROM face_embeddings WHERE person_id = ?",
            (person_id,)
        )
        embedding_count = cursor.fetchone()[0]
        
        conn.close()
        
        return {
            "id": person_id,
            "name": name,
            "created_at": created_at,
            "embedding_count": embedding_count
        }
    
    def list_persons(self) -> List[dict]:
        """
        List all registered persons.
        
        Returns:
            List of person dictionaries
        """
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute("""
            SELECT p.id, p.name, p.created_at, COUNT(fe.id) as embedding_count
            FROM persons p
            LEFT JOIN face_embeddings fe ON p.id = fe.person_id
            GROUP BY p.id, p.name, p.created_at
            ORDER BY p.created_at DESC
        """)
        
        persons = []
        for row in cursor.fetchall():
            persons.append({
                "id": row[0],
                "name": row[1],
                "created_at": row[2],
                "embedding_count": row[3]
            })
        
        conn.close()
        return persons
    
    def delete_person(self, name: str) -> bool:
        """
        Delete person and all their embeddings.
        
        Args:
            name: Person's name
            
        Returns:
            True if deleted, False if not found
        """
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute("DELETE FROM persons WHERE name = ?", (name,))
        deleted = cursor.rowcount > 0
        
        conn.commit()
        conn.close()
        
        return deleted
    
    def get_embedding_count(self) -> int:
        """Get total number of stored embeddings."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute("SELECT COUNT(*) FROM face_embeddings")
        count = cursor.fetchone()[0]
        
        conn.close()
        return count


# Example usage
if __name__ == "__main__":
    print("Testing FaceStore...")
    
    # Initialize store
    store = FaceStore()
    
    # List current persons
    print("\nCurrent persons:")
    persons = store.list_persons()
    if persons:
        for p in persons:
            print(f"  - {p['name']}: {p['embedding_count']} embeddings (added {p['created_at']})")
    else:
        print("  (none)")
    
    # Test with dummy embeddings
    print("\nTesting with dummy data...")
    try:
        # Create test embeddings
        test_embeddings = [np.random.randn(128).astype(np.float32) for _ in range(3)]
        # Normalize
        test_embeddings = [e / np.linalg.norm(e) for e in test_embeddings]
        
        # Register test person
        person_id = store.register_person("Test Person", test_embeddings)
        print(f"✓ Registered test person (ID: {person_id})")
        
        # Try to match
        match = store.find_match(test_embeddings[0], threshold=0.3)
        if match:
            print(f"✓ Found match: {match[0]} (similarity: {match[1]:.3f})")
        else:
            print("✗ No match found")
        
        # Get person info
        info = store.get_person("Test Person")
        print(f"✓ Person info: {info}")
        
        # Delete test person
        deleted = store.delete_person("Test Person")
        print(f"✓ Deleted test person: {deleted}")
        
    except Exception as e:
        print(f"✗ Error: {e}")
        import traceback
        traceback.print_exc()
    
    print("\nTest complete!")

