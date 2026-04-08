import asyncio
import os
from dotenv import load_dotenv
from motor.motor_asyncio import AsyncIOMotorClient

# Load the .env file
load_dotenv('.env/.env')

async def main():
    try:
        url = os.getenv('MONGO_URI')
        if not url:
            print("ERROR: MONGO_URI is missing!")
            return
            
        print(f"Connecting to: {url.split('@')[-1]}")
        client = AsyncIOMotorClient(url, serverSelectionTimeoutMS=5000)
        db = client['agrithon']
        
        # Test insert
        result = await db['prices'].insert_one({"test": "data"})
        print(f"SUCCESS! Inserted test data with ID: {result.inserted_id}")
        
    except Exception as e:
        import traceback
        print("\n=== DETAILED ERROR ===")
        traceback.print_exc()

if __name__ == '__main__':
    asyncio.run(main())
