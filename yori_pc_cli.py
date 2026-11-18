import sys
import time
from yori_pc_agent import YoriPcEngine

def main():
    print("=" * 65)
    print("⚡ YORI — Autonomous Executive Agent (PC Edition)")
    print("Pure Neural Maria-1B on CUDA (Zero Hardcoding / Zero Regex)")
    print("=" * 65)
    print("Loading model...")
    engine = YoriPcEngine()
    print("\n[READY] Type your goal, question, or command. Type 'exit' to quit.\n")

    while True:
        try:
            query = input("\nYori > ").strip()
            if not query:
                continue
            if query.lower() in ("exit", "quit", "q"):
                print("Exiting Yori.")
                break

            print("\n[🧠 Reasoning...]", flush=True)
            t0 = time.time()
            full_output = ""
            for step in engine.stream_turn(query, max_tokens=90):
                full_output = step["full_text"]

            elapsed = time.time() - t0
            llm_thought = full_output.split("</thought>")[0].replace("<thought>", "").strip()
            print(f"Thought: {llm_thought}")

            if "<action>" in full_output:
                action_str = full_output.split("<action>")[1].split("</action>")[0].strip()
                print(f"\n⚡ Action Emitted: {action_str}")
                res = engine.execute_pc_action(action_str)
                print(f"↳ Execution Result: {res['message']}")
            else:
                print(f"\nResponse: {llm_thought}")

            print(f"(Elapsed: {elapsed:.2f}s)")

        except KeyboardInterrupt:
            print("\nInterrupted.")
            break
        except Exception as e:
            print(f"Error: {e}")

if __name__ == "__main__":
    main()
