import threading

from src.handler_rl import HandlerRL
from src.utils.io_record import INPUT_QUEUE, OUTPUT_QUEUE, init_record


def console_io_loop():
    while True:
        question = OUTPUT_QUEUE.get()
        print(f"\nQUESTION: {question}", flush=True)
        try:
            user_input = input("Your answer: ")
        except (EOFError, KeyboardInterrupt):
            INPUT_QUEUE.put("stop")
            break
        INPUT_QUEUE.put(user_input)


def main():
    init_record()
    threading.Thread(target=console_io_loop, daemon=True).start()
    HandlerRL().run()


if __name__ == "__main__":
    main()
