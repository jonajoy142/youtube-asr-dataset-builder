import asyncio
import logging
from src.main import main

logging.basicConfig(level=logging.INFO)

if __name__ == "__main__":
    asyncio.run(main())
