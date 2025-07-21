import os
from typing import Optional, Dict, Any
from pymongo import MongoClient
from motor.motor_asyncio import AsyncIOMotorClient
import asyncio
from dotenv import load_dotenv
from bson.objectid import ObjectId

load_dotenv()

class MongoDBClient:
    """
    MongoDB client for fetching character data from the database
    """
    
    def __init__(self, mongodb_url: str = None):
        self.mongodb_url = mongodb_url or os.getenv("MONGODB_URL")
        if not self.mongodb_url:
            raise ValueError("MongoDB URL is required")
        
        self.sync_client = None
        self.async_client = None
        self.database = None
        self.async_database = None
        
    def _get_query_id(self, character_id: str):
        """Convert character_id to appropriate format for MongoDB query"""
        try:
            # Try to convert to ObjectId first (if it's a valid ObjectId string)
            if ObjectId.is_valid(character_id):
                return ObjectId(character_id)
            else:
                # If not a valid ObjectId, use as string
                return character_id
        except:
            # Fallback to string if ObjectId conversion fails
            return character_id
    
    def connect(self):
        """Establish synchronous connection to MongoDB"""
        try:
            self.sync_client = MongoClient(self.mongodb_url)
            # Extract database name from URL or use default
            self.database = self.sync_client.get_default_database()
            print(f"[MongoDB] Connected to database: {self.database.name}")
        except Exception as e:
            print(f"[MongoDB] Connection error: {e}")
            raise e
    
    async def connect_async(self):
        """Establish asynchronous connection to MongoDB"""
        try:
            self.async_client = AsyncIOMotorClient(self.mongodb_url)
            # Extract database name from URL or use default
            self.async_database = self.async_client.get_default_database()
            print(f"[MongoDB] Async connected to database: {self.async_database.name}")
        except Exception as e:
            print(f"[MongoDB] Async connection error: {e}")
            raise e
    
    def get_character_by_id(self, character_id: str) -> Optional[Dict[str, Any]]:
        """
        Fetch character data by ID (synchronous)
        
        Args:
            character_id: The ID of the character to fetch
            
        Returns:
            Dictionary containing character data or None if not found
        """
        try:
            if self.database is None:
                self.connect()
            
            characters_collection = self.database.characters
            character = characters_collection.find_one({"_id": self._get_query_id(character_id)})
            
            if character is not None:
                print(f"[MongoDB] Found character: {character.get('name', 'Unknown')}")
                return character
            else:
                print(f"[MongoDB] Character with ID {character_id} not found")
                return None
                
        except Exception as e:
            print(f"[MongoDB] Error fetching character: {e}")
            return None
    
    async def get_character_by_id_async(self, character_id: str) -> Optional[Dict[str, Any]]:
        """
        Fetch character data by ID (asynchronous)
        
        Args:
            character_id: The ID of the character to fetch
            
        Returns:
            Dictionary containing character data or None if not found
        """
        try:
            if self.async_database is None:
                await self.connect_async()
            
            characters_collection = self.async_database.characters
            character = await characters_collection.find_one({"_id": self._get_query_id(character_id)})
            
            if character is not None:
                print(f"[MongoDB] Found character: {character.get('name', 'Unknown')}")
                return character
            else:
                print(f"[MongoDB] Character with ID {character_id} not found")
                return None
                
        except Exception as e:
            print(f"[MongoDB] Error fetching character: {e}")
            return None
    
    def get_character_voice_and_prompt(self, character_id: str) -> tuple[Optional[str], Optional[str]]:
        """
        Fetch voice ID and prompt for a character (synchronous)
        
        Args:
            character_id: The ID of the character
            
        Returns:
            Tuple of (voice_id, prompt) or (None, None) if not found
        """
        character = self.get_character_by_id(character_id)
        if character:
            voice_id = character.get('voiceId') or character.get('voice_id')
            prompt = character.get('prompt')
            return voice_id, prompt
        return None, None
    
    async def get_character_voice_and_prompt_async(self, character_id: str) -> tuple[Optional[str], Optional[str]]:
        """
        Fetch voice ID and prompt for a character (asynchronous)
        
        Args:
            character_id: The ID of the character
            
        Returns:
            Tuple of (voice_id, prompt) or (None, None) if not found
        """
        character = await self.get_character_by_id_async(character_id)
        if character:
            voice_id = character.get('voiceId') or character.get('voice_id')
            prompt = character.get('prompt')
            return voice_id, prompt
        return None, None
    
    def close(self):
        """Close database connections"""
        if self.sync_client:
            self.sync_client.close()
        if self.async_client:
            self.async_client.close()

# Global instance
mongodb_client = MongoDBClient() 