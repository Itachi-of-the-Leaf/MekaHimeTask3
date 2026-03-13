import os
import uuid
import logging
import chromadb
from typing import List, Tuple, Optional

logger = logging.getLogger(__name__)

class SpeakerDB:
    def __init__(self, db_path: str = "./speaker_chroma_db", collection_name: str = "speakers"):
        """
        Initializes the ChromaDB client to store and query speaker profiles.
        """
        logger.info(f"Initializing ChromaDB Persistent Client at {db_path}...")
        self.db_path = db_path
        self.client = chromadb.PersistentClient(path=db_path)
        # Using cosine distance for speaker embedding comparisons
        self.collection = self.client.get_or_create_collection(
            name=collection_name,
            metadata={"hnsw:space": "cosine"}
        )
        
    def enroll_speaker(self, name: str, embedding: List[float], speaker_id: Optional[str] = None) -> str:
        """
        Registers a new speaker in the database.
        
        Args:
            name (str): The name or label of the speaker.
            embedding (List[float]): The voiceprint embedding.
            speaker_id (Optional[str]): A unique ID for the speaker. If not provided, generates a UUID.
            
        Returns:
            str: The unique ID of the enrolled speaker.
        """
        if not speaker_id:
            speaker_id = str(uuid.uuid4())
            
        logger.info(f"Enrolling speaker '{name}' with ID '{speaker_id}'...")
        
        self.collection.upsert(
            embeddings=[embedding],
            documents=[name],
            metadatas=[{"name": name}],
            ids=[speaker_id]
        )
        return speaker_id
        
    def identify_speaker(self, embedding: List[float], n_results: int = 1, threshold: float = 0.5) -> List[Tuple[str, str, float]]:
        """
        Queries the database for the closest matching speaker profiles.
        
        Args:
            embedding (List[float]): The query voiceprint embedding.
            n_results (int): The number of closest speakers to retrieve.
            threshold (float): Maximum allowed cosine distance. Lower distance means closer match.
            
        Returns:
            List[Tuple[str, str, float]]: A list of tuples containing (speaker_id, name, distance).
        """
        logger.info("Querying for closest speaker matches...")
        
        results = self.collection.query(
            query_embeddings=[embedding],
            n_results=n_results
        )
        
        matches = []
        if results['ids']:
            ids = results['ids'][0]
            distances = results['distances'][0]
            metadatas = results['metadatas'][0]
            
            for i in range(len(ids)):
                dist = distances[i]
                if dist <= threshold:
                    matches.append((ids[i], metadatas[i]["name"], dist))
                    
        return matches
