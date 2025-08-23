# concept-llama/test.py
import requests
import os

# --- Configuration ---
VLLM_URL = os.getenv("VLLM_URL", "https://upatpmb38ag5v3-8000.proxy.runpod.net/v1/completions")
MODEL_NAME = "meta-llama/Llama-3.1-8B-Instruct"

def format_prompt_for_llama3(prompt: str) -> str:
    """Wraps the user's prompt in the official Llama 3 chat template."""
    return (
        f"<|begin_of_text|><|start_header_id|>user<|end_header_id|>\n\n"
        f"{prompt}<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n"
    )

def test_k_values():
    """
    Tests the vLLM server to see the maximum supported k-value for logprobs.
    """
    print(f"--- Testing logprobs limit on: {VLLM_URL} ---")
    print(f"--- Model: {MODEL_NAME} ---")
    
    prompt = "The capital of France is"
    formatted_prompt = format_prompt_for_llama3(prompt)
    
    # Test a range of k values, starting from what we know works and fails
    k_values_to_test = [10, 20, 25, 50, 100, 200, 500]

    for k in k_values_to_test:
        print(f"\n--- Testing k={k} ---")
        
        json_payload = {
            "model": MODEL_NAME,
            "prompt": formatted_prompt,
            "max_tokens": 1,
            "logprobs": k,
            "temperature": 0.0,
        }

        try:
            response = requests.post(VLLM_URL, json=json_payload, timeout=30)
            
            if response.status_code == 200:
                data = response.json()
                # Check if the response structure is as expected
                if (data.get("choices") and 
                    data["choices"][0].get("logprobs") and 
                    data["choices"][0]["logprobs"].get("top_logprobs")):
                    
                    actual_logprobs_count = len(data["choices"][0]["logprobs"]["top_logprobs"][0])
                    print(f"✅ Success! Server returned {actual_logprobs_count} logprobs.")
                else:
                    print("❌ Failed - Response OK, but logprobs data is missing or malformed.")
                    print("Full response:", data)

            else:
                print(f"❌ Failed - Status Code: {response.status_code}")
                print(f"   Response: {response.text}")

        except requests.exceptions.RequestException as e:
            print(f"❌ An error occurred while calling the vLLM API: {e}")
            break # Stop testing if the server is unreachable

if __name__ == "__main__":
    test_k_values()
