"""
Example: Extract Â from Qwen3-8B and test it.

Usage:
    python examples/quickstart.py
    python examples/quickstart.py --model Qwen/Qwen3-1.7B --device cpu
"""

import argparse
from a_hat_optimizer import AHat


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="Qwen/Qwen3-8B")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--save-to", default="a_hat_output")
    args = parser.parse_args()

    print(f"Extracting Â from {args.model}...")
    a_hat = AHat.from_model(args.model, device=args.device)
    print(a_hat)
    print(f"Info: {a_hat.info()}")

    a_hat.save(args.save_to)
    print(f"Saved to {args.save_to}/")

    # Reload and verify
    a_hat_loaded = AHat.from_file(args.save_to)
    print(f"Reloaded: {a_hat_loaded}")


if __name__ == "__main__":
    main()
