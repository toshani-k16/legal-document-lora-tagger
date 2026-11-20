"""Command-line entry point: `python -m app.cli --help`."""
import argparse


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(prog="legal-document-lora-tagger", description="Public sector clerks classify documents faster. This tool uses LoRA to fine-tune GPT-2 to tag legal clauses in public records, improving efficiency and reducing manual review time.")
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("hello", help="check that everything is wired up")
    # TODO: add your own commands here as you work through the milestones.
    args = parser.parse_args(argv)
    if args.command == "hello":
        print("Legal Document LoRA Tagger is ready. Open README.md for your milestones.")


if __name__ == "__main__":
    main()
