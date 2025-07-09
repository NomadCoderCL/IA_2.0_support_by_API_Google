"""
Command-line interface for the NOVA chatbot.
This provides a fallback when GUI dependencies are not available.
"""
import sys
import json
from config import IA_NAME

# Try to import Google API, fall back gracefully if not available
try:
    from google_api import predict_google_api
    GOOGLE_API_AVAILABLE = True
except ImportError:
    print("Warning: Google API not available. Using echo responses.")
    GOOGLE_API_AVAILABLE = False
    
    def predict_google_api(input_text):
        return f"Echo: {input_text} (Google API not available)"


def save_interaction(user_input, response):
    """Save interaction to log file."""
    try:
        log_entry = {
            "user_input": user_input,
            "response": response
        }
        with open("interactions_log.json", "a") as log_file:
            log_file.write(json.dumps(log_entry) + "\n")
    except Exception as e:
        print(f"Error saving interaction: {e}")


def main():
    """Main command-line interface loop."""
    print(f"Welcome to {IA_NAME} - Command Line Interface")
    print("Type 'quit', 'exit', or press Ctrl+C to exit")
    print("=" * 50)
    
    if not GOOGLE_API_AVAILABLE:
        print("Note: Running in echo mode (Google API not available)")
        print()
    
    try:
        while True:
            try:
                user_input = input("You: ").strip()
                
                if user_input.lower() in ['quit', 'exit', 'q']:
                    print("Goodbye!")
                    break
                
                if not user_input:
                    continue
                
                # Get response
                response = predict_google_api(user_input)
                print(f"{IA_NAME}: {response}")
                print()
                
                # Save interaction
                save_interaction(user_input, response)
                
            except EOFError:
                print("\nGoodbye!")
                break
            except Exception as e:
                print(f"Error: {e}")
                
    except KeyboardInterrupt:
        print("\nGoodbye!")


if __name__ == "__main__":
    main()