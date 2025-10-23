"""
API Testing Script for Medical SOAP Summarization
==================================================

Test the deployed Flask API with various scenarios.

Usage:
    python test_api.py --url http://your-api-url.com
"""

import requests
import json
import time
import argparse
from typing import Dict, Any

# ============================================================================
# TEST DATA
# ============================================================================

TEST_DIALOGUES = [
    {
        "name": "Headache Case",
        "dialogue": """
Patient: Hello doctor, I've been experiencing severe headaches for the past week.
Doctor: Can you describe the type of pain?
Patient: It's a throbbing pain on both sides of my head.
Doctor: Any other symptoms like nausea or sensitivity to light?
Patient: Yes, I feel nauseous and bright lights make it worse.
Doctor: This appears to be migraines. I'll prescribe medication.
        """
    },
    {
        "name": "Cough Case",
        "dialogue": """
Patient: I've had a persistent cough for two weeks.
Doctor: Is it a dry cough or producing mucus?
Patient: Yellowish mucus, especially in the morning.
Doctor: Any fever or shortness of breath?
Patient: Had a low-grade fever last week, now gone.
Doctor: Let me check your lungs. You have some wheezing.
        """
    },
    {
        "name": "Fatigue Case",
        "dialogue": """
Patient: I've been feeling extremely tired for a month.
Doctor: Any other symptoms?
Patient: I feel dizzy when standing up quickly and lost 5 pounds.
Doctor: Changes in appetite?
Patient: Not really, but my skin looks paler.
Doctor: We should run blood tests for anemia or thyroid issues.
        """
    }
]

# ============================================================================
# TEST FUNCTIONS
# ============================================================================

def test_health_check(base_url: str) -> bool:
    """Test the health check endpoint"""
    print("\n" + "="*70)
    print("TEST 1: Health Check")
    print("="*70)
    
    try:
        response = requests.get(f"{base_url}/health")
        print(f"Status Code: {response.status_code}")
        print(f"Response: {json.dumps(response.json(), indent=2)}")
        
        assert response.status_code == 200, "Health check failed"
        assert response.json()['status'] == 'healthy', "API not healthy"
        
        print("✅ PASSED: Health check successful")
        return True
        
    except Exception as e:
        print(f"❌ FAILED: {e}")
        return False

def test_api_info(base_url: str) -> bool:
    """Test the API info endpoint"""
    print("\n" + "="*70)
    print("TEST 2: API Info")
    print("="*70)
    
    try:
        response = requests.get(base_url)
        print(f"Status Code: {response.status_code}")
        print(f"Response: {json.dumps(response.json(), indent=2)}")
        
        assert response.status_code == 200, "API info request failed"
        
        print("✅ PASSED: API info retrieved successfully")
        return True
        
    except Exception as e:
        print(f"❌ FAILED: {e}")
        return False

def test_single_generation(base_url: str, dialogue_data: Dict[str, str]) -> bool:
    """Test single SOAP note generation"""
    print("\n" + "="*70)
    print(f"TEST 3: Single Generation - {dialogue_data['name']}")
    print("="*70)
    
    try:
        start_time = time.time()
        
        payload = {
            "dialogue": dialogue_data['dialogue'],
            "max_length": 900,
            "min_length": 150,
            "num_beams": 4,
            "length_penalty": 1.5,
            "repetition_penalty": 1.2
        }
        
        response = requests.post(
            f"{base_url}/generate",
            json=payload,
            headers={"Content-Type": "application/json"}
        )
        
        elapsed_time = time.time() - start_time
        
        print(f"Status Code: {response.status_code}")
        print(f"Response Time: {elapsed_time:.2f} seconds")
        
        if response.status_code == 200:
            result = response.json()
            print(f"\n📝 Input (first 200 chars):\n{dialogue_data['dialogue'][:200]}...")
            print(f"\n🤖 Generated SOAP Note:\n{result['soap_note']}")
            print(f"\n📊 Metadata:")
            for key, value in result['metadata'].items():
                print(f"  {key}: {value}")
            
            print(f"\n✅ PASSED: Generation successful")
            return True
        else:
            print(f"Response: {response.text}")
            print(f"❌ FAILED: Status code {response.status_code}")
            return False
        
    except Exception as e:
        print(f"❌ FAILED: {e}")
        return False

def test_batch_generation(base_url: str) -> bool:
    """Test batch SOAP note generation"""
    print("\n" + "="*70)
    print("TEST 4: Batch Generation")
    print("="*70)
    
    try:
        start_time = time.time()
        
        dialogues = [d['dialogue'] for d in TEST_DIALOGUES]
        
        payload = {
            "dialogues": dialogues,
            "max_length": 900,
            "min_length": 150
        }
        
        response = requests.post(
            f"{base_url}/batch",
            json=payload,
            headers={"Content-Type": "application/json"}
        )
        
        elapsed_time = time.time() - start_time
        
        print(f"Status Code: {response.status_code}")
        print(f"Total Time: {elapsed_time:.2f} seconds")
        
        if response.status_code == 200:
            result = response.json()
            print(f"Total Processed: {result['total_processed']}")
            print(f"Server Processing Time: {result['total_time_seconds']} seconds")
            
            for i, res in enumerate(result['results']):
                if res['success']:
                    print(f"\n✅ Result {i+1}: SUCCESS")
                    print(f"   Output length: {res['metadata']['output_length']} chars")
                else:
                    print(f"\n❌ Result {i+1}: FAILED - {res.get('error', 'Unknown error')}")
            
            print(f"\n✅ PASSED: Batch generation successful")
            return True
        else:
            print(f"Response: {response.text}")
            print(f"❌ FAILED: Status code {response.status_code}")
            return False
        
    except Exception as e:
        print(f"❌ FAILED: {e}")
        return False

def test_error_handling(base_url: str) -> bool:
    """Test error handling with invalid inputs"""
    print("\n" + "="*70)
    print("TEST 5: Error Handling")
    print("="*70)
    
    test_cases = [
        {
            "name": "Empty dialogue",
            "payload": {"dialogue": ""},
            "expected_status": 400
        },
        {
            "name": "Missing dialogue field",
            "payload": {"text": "some text"},
            "expected_status": 400
        },
        {
            "name": "Invalid max_length",
            "payload": {"dialogue": "test", "max_length": 5000},
            "expected_status": 400
        },
        {
            "name": "Invalid length_penalty",
            "payload": {"dialogue": "test", "length_penalty": 10.0},
            "expected_status": 400
        }
    ]
    
    passed = 0
    failed = 0
    
    for test_case in test_cases:
        try:
            response = requests.post(
                f"{base_url}/generate",
                json=test_case['payload'],
                headers={"Content-Type": "application/json"}
            )
            
            if response.status_code == test_case['expected_status']:
                print(f"✅ {test_case['name']}: Correctly returned {response.status_code}")
                passed += 1
            else:
                print(f"❌ {test_case['name']}: Expected {test_case['expected_status']}, got {response.status_code}")
                failed += 1
                
        except Exception as e:
            print(f"❌ {test_case['name']}: Exception - {e}")
            failed += 1
    
    print(f"\nError Handling: {passed}/{len(test_cases)} tests passed")
    
    if failed == 0:
        print("✅ PASSED: All error handling tests successful")
        return True
    else:
        print(f"❌ FAILED: {failed} error handling tests failed")
        return False

def test_parameter_variations(base_url: str) -> bool:
    """Test with different parameter combinations"""
    print("\n" + "="*70)
    print("TEST 6: Parameter Variations")
    print("="*70)
    
    test_dialogue = TEST_DIALOGUES[0]['dialogue']
    
    parameter_sets = [
        {"name": "Short output", "max_length": 400, "min_length": 100},
        {"name": "Long output", "max_length": 900, "min_length": 200},
        {"name": "High quality", "num_beams": 8, "length_penalty": 2.0},
        {"name": "Fast generation", "num_beams": 2, "length_penalty": 1.0},
    ]
    
    passed = 0
    
    for params in parameter_sets:
        name = params.pop('name')
        payload = {"dialogue": test_dialogue, **params}
        
        try:
            response = requests.post(
                f"{base_url}/generate",
                json=payload,
                headers={"Content-Type": "application/json"}
            )
            
            if response.status_code == 200:
                result = response.json()
                output_len = result['metadata']['output_length']
                gen_time = result['metadata']['generation_time_seconds']
                print(f"✅ {name}: {output_len} chars, {gen_time:.2f}s")
                passed += 1
            else:
                print(f"❌ {name}: Status {response.status_code}")
                
        except Exception as e:
            print(f"❌ {name}: {e}")
    
    print(f"\nParameter Tests: {passed}/{len(parameter_sets)} passed")
    
    if passed == len(parameter_sets):
        print("✅ PASSED: All parameter variation tests successful")
        return True
    else:
        print(f"❌ FAILED: Some parameter tests failed")
        return False

# ============================================================================
# MAIN TEST RUNNER
# ============================================================================

def run_all_tests(base_url: str):
    """Run all API tests"""
    print("="*70)
    print("🧪 MEDICAL SOAP API TEST SUITE")
    print("="*70)
    print(f"Testing API at: {base_url}")
    print("="*70)
    
    results = []
    
    # Run tests
    results.append(("Health Check", test_health_check(base_url)))
    results.append(("API Info", test_api_info(base_url)))
    
    for dialogue_data in TEST_DIALOGUES:
        results.append((
            f"Single Generation - {dialogue_data['name']}",
            test_single_generation(base_url, dialogue_data)
        ))
    
    results.append(("Batch Generation", test_batch_generation(base_url)))
    results.append(("Error Handling", test_error_handling(base_url)))
    results.append(("Parameter Variations", test_parameter_variations(base_url)))
    
    # Print summary
    print("\n" + "="*70)
    print("📊 TEST SUMMARY")
    print("="*70)
    
    passed = sum(1 for _, result in results if result)
    total = len(results)
    
    for test_name, result in results:
        status = "✅ PASSED" if result else "❌ FAILED"
        print(f"{status}: {test_name}")
    
    print("="*70)
    print(f"Total: {passed}/{total} tests passed ({passed/total*100:.1f}%)")
    print("="*70)
    
    if passed == total:
        print("\n🎉 ALL TESTS PASSED!")
    else:
        print(f"\n⚠️  {total - passed} tests failed. Please review.")

# ============================================================================
# CLI
# ============================================================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Test Medical SOAP API")
    parser.add_argument(
        "--url",
        type=str,
        default="http://localhost:5000",
        help="Base URL of the API (default: http://localhost:5000)"
    )
    
    args = parser.parse_args()
    
    # Remove trailing slash
    base_url = args.url.rstrip('/')
    
    # Run tests
    run_all_tests(base_url)
