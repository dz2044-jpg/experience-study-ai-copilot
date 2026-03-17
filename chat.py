"""Terminal entry point for the Experience Study AI Copilot."""

from agents.orchestrator import StudyOrchestrator


def main() -> None:
    orchestrator = StudyOrchestrator()

    print("=" * 50)
    print("=== Experience Study AI Copilot ===")
    print("=" * 50)
    print("\nType 'exit' or 'quit' to stop.\n")

    while True:
        user_input = input("\nYou: ").strip()

        if not user_input:
            continue
        if user_input.lower() in ("exit", "quit"):
            print("\nGoodbye.")
            break

        print("\nCopilot is thinking...")
        response = orchestrator.process_query(user_input)
        print(f"\nCopilot: {response}")


if __name__ == "__main__":
    main()
