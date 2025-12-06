"""
Test script for Hazard Model & Registry
"""
import os
import sys
import json
from pathlib import Path

# Add src to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.core.environment_hazards import hazard_registry, HazardTypeDefinition, HazardCategory
from src.platform.safety_manager import safety_manager

def test_builtin_hazards():
    print("\n--- Testing Built-in Hazards ---")
    overhang = hazard_registry.get("OVERHANG")
    if overhang:
        print(f"PASS: Found OVERHANG: {overhang.description}")
    else:
        print("FAIL: OVERHANG not found")
        
    person = hazard_registry.get("PERSON")
    if person:
        print(f"PASS: Found PERSON: {person.default_behaviour}")
    else:
        print("FAIL: PERSON not found")

def test_custom_registration():
    print("\n--- Testing Custom Hazard Registration ---")
    
    custom_type = "FALLING_BOX"
    definition = HazardTypeDefinition(
        type_key=custom_type,
        category=HazardCategory.CUSTOM,
        display_name="Falling Box",
        description="Box detected falling from shelf",
        default_severity=0.9,
        default_behaviour={"action": "STOP", "clearance_m": 3.0}
    )
    
    hazard_registry.register_custom(definition)
    
    retrieved = hazard_registry.get(custom_type)
    if retrieved and retrieved.category == HazardCategory.CUSTOM:
        print(f"PASS: Registered and retrieved {custom_type}")
    else:
        print(f"FAIL: Could not retrieve {custom_type}")
        
    # Check persistence
    config_file = Path("config/custom_hazards.json")
    if config_file.exists():
        with open(config_file, 'r') as f:
            data = json.load(f)
            found = any(d['type_key'] == custom_type for d in data)
            if found:
                print("PASS: Custom hazard persisted to disk")
            else:
                print("FAIL: Custom hazard not found in JSON file")
    else:
        print("FAIL: config/custom_hazards.json not created")

def test_safety_evaluation():
    print("\n--- Testing Safety Evaluation ---")
    
    # Test Built-in
    action = safety_manager.evaluate_hazard("PERSON", 0.5)
    print(f"PERSON at 0.5m -> {action} (Expected: STOP)")
    
    action = safety_manager.evaluate_hazard("PERSON", 5.0)
    print(f"PERSON at 5.0m -> {action} (Expected: SAFE)")
    
    # Test Custom
    action = safety_manager.evaluate_hazard("FALLING_BOX", 1.0)
    print(f"FALLING_BOX at 1.0m -> {action} (Expected: STOP)")
    
    # Test Unknown
    action = safety_manager.evaluate_hazard("UNKNOWN_THING", 1.0)
    print(f"UNKNOWN_THING at 1.0m -> {action} (Expected: WARN)")

if __name__ == "__main__":
    test_builtin_hazards()
    test_custom_registration()
    test_safety_evaluation()
