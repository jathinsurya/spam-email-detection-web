import sys
try:
    print("Loading model...")
    from model import predict_message
    print("✅ Model loaded successfully")
    
    print("\nTesting predictions...")
    test_cases = [
        "Hello friend",
        "Win money now",
        "Click here to claim reward",
        "Meeting at 5 pm"
    ]
    
    for test in test_cases:
        try:
            result = predict_message(test)
            print(f"  ✅ '{test}' -> {result}")
        except Exception as e:
            print(f"  ❌ '{test}' -> ERROR: {e}")
            
except Exception as e:
    print(f"❌ Model load error: {e}")
    import traceback
    traceback.print_exc()
