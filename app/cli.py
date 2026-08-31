import os
import sys
from pathlib import Path

# Add project root directory to sys.path so we can import src
BASE_DIR = Path(__file__).resolve().parent.parent
if str(BASE_DIR) not in sys.path:
    sys.path.append(str(BASE_DIR))

from src.chatbot import Chatbot

def print_help():
    """Prints list of CLI commands."""
    print("\n" + "="*50)
    print("CAMPUSMATE CLI COMMANDS:")
    print("  /help    - Show this help menu")
    print("  /debug   - Toggle debug mode (visualize NLP pipeline details)")
    print("  /intents - List all supported intent categories")
    print("  /stats   - View model metadata and accuracy metrics")
    print("  /reset   - Reset dialog context memory")
    print("  /quit    - Exit the chatbot application")
    print("="*50 + "\n")

def run_cli():
    """Starts the interactive CLI loop."""
    print("="*60)
    print("CAMPUSMATE: UNIVERSITY FAQ & INTENT ASSISTANT (CLI)")
    print("Loading machine learning models, please wait...")
    print("="*60)

    try:
        # Initialize chatbot
        chatbot = Chatbot()
    except Exception as e:
        print(f"\nCRITICAL ERROR: Could not load model: {e}")
        print("Please ensure you have trained the model first:")
        print("  python -m src.train")
        sys.exit(1)

    print("\nBot: Hello! I am CampusMate, your University FAQ assistant. How can I help you today?")
    print("Tip: Type '/help' to see list of developer commands, or '/quit' to exit.")
    print("-" * 60)

    while True:
        try:
            user_input = input("You: ").strip()
        except (KeyboardInterrupt, EOFError):
            print("\nBot: Goodbye!")
            break

        if not user_input:
            continue

        # Handle commands starting with '/'
        if user_input.startswith("/"):
            cmd = user_input.lower()
            if cmd == "/quit":
                print("Bot: Goodbye!")
                break
            elif cmd == "/help":
                print_help()
            elif cmd == "/debug":
                chatbot.debug_mode = not chatbot.debug_mode
                status = "ENABLED" if chatbot.debug_mode else "DISABLED"
                print(f"Bot: Debug mode has been {status}.")
            elif cmd == "/intents":
                classes = chatbot.metadata.get("classes", []) if chatbot.metadata else list(chatbot.classifier.classes_)
                print("\nSupported FAQ Intents:")
                for tag in sorted(classes):
                    print(f"  - {tag}")
                print(f"Total: {len(classes)} intents")
                print("-" * 60 + "\n")
            elif cmd == "/reset":
                chatbot.reset_context()
                print("Bot: Chatbot dialog context memory has been cleared.")
            elif cmd == "/stats":
                if chatbot.metadata:
                    print("\nModel Statistics and Evaluation Metrics:")
                    print(f"  Model Name:        {chatbot.metadata.get('model_name')}")
                    print(f"  Training Timestamp: {chatbot.metadata.get('training_date')}")
                    print(f"  Total Samples:     {chatbot.metadata.get('total_samples')}")
                    print(f"  Training Samples:  {chatbot.metadata.get('train_samples_count')}")
                    print(f"  Testing Samples:   {chatbot.metadata.get('test_samples_count')}")
                    
                    metrics = chatbot.metadata.get("metrics", {})
                    print(f"  Test Accuracy:     {metrics.get('test_accuracy', 0.0)*100:.2f}%")
                    print(f"  Test Weighted F1:  {metrics.get('test_weighted_f1', 0.0)*100:.2f}%")
                else:
                    print("Bot: No model metadata available.")
                print("-" * 60 + "\n")
            else:
                print(f"Bot: Unknown command '{user_input}'. Type '/help' to view commands.")
            continue

        # Process user query through chatbot
        response_packet = chatbot.respond(user_input)
        print(f"Bot: {response_packet['response']}")

        # Render debug outputs if debug mode is active
        if chatbot.debug_mode:
            debug = response_packet["debug_info"]
            print("\n" + "-"*30 + " PIPELINE DEBUG " + "-"*30)
            print(f"  Raw Input text:      {debug['input']}")
            print(f"  Processed tokens:   {debug['processed_tokens']}")
            print(f"  Predicted Intent:   {debug['predicted_intent']}")
            print(f"  Top Confidence:     {debug['confidence']:.4f}")
            if debug.get("second_intent"):
                print(f"  Second Intent:      {debug['second_intent']}")
                print(f"  Second Confidence:  {debug['second_confidence']:.4f}")
            print(f"  Fallback Triggered: {debug['is_fallback']}")
            print(f"  Fallback Reason:    {debug['reason']}")
            print("-" * 76 + "\n")

if __name__ == "__main__":
    run_cli()
