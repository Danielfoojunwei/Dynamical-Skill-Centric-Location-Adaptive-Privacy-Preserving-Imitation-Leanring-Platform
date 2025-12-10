import sys
import os
import asyncio
from typing import Dict, Any
from pydantic import BaseModel

# Add src to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

# Mock FastAPI dependencies if needed, but here we just import the function
from src.platform.api.main import create_hazard_type, get_hazard_types, HazardTypeCreate

import pytest

@pytest.mark.asyncio
async def test_direct_calls():
    print("Testing get_hazard_types...")
    hazards = await get_hazard_types()
    print(f"Got {len(hazards)} hazards")
    assert len(hazards) > 0
    
    print("Testing create_hazard_type...")
    new_hazard = HazardTypeCreate(
        type_key="DIRECT_TEST",
        display_name="Direct Test",
        description="Testing via direct call",
        default_severity=0.1,
        default_behaviour={"action": "WARN"}
    )
    
    try:
        result = await create_hazard_type(new_hazard)
        print(f"Result: {result}")
    except Exception as e:
        # It might raise HTTPException, which is fine, we just want to see if logic works
        # But create_hazard_type raises HTTPException if exists.
        print(f"Result: {e}")
    
    hazards_after = await get_hazard_types()
    assert any(h['type_key'] == "DIRECT_TEST" for h in hazards_after)
    print("Verification successful!")

if __name__ == "__main__":
    loop = asyncio.new_event_loop()
    loop.run_until_complete(test_direct_calls())
    loop.close()
