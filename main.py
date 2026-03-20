import argparse
import json
import sys

from models import Invoice
from parser import parse_with_instructor, parse_with_normal_prompt
from invoices import INVOICES


def cmd_parse(args):
    """Parse a single invoice text."""
    invoice = parse_with_instructor(args.text)
    print(json.dumps(invoice.model_dump(mode="json"), indent=2, ensure_ascii=False))


def cmd_batch(args):
    """Batch parse all invoices and save to invoices.json."""
    results = []
    for i, invoice_text in enumerate(INVOICES, 1):
        try:
            invoice = parse_with_instructor(invoice_text)
            results.append({
                "invoice_index": i,
                "success": True,
                "data": invoice.model_dump(mode="json")
            })
            print(f"Invoice {i}: OK")
        except Exception as e:
            results.append({
                "invoice_index": i,
                "success": False,
                "error": str(e)
            })
            print(f"Invoice {i}: FAILED - {e}")

    with open("invoices.json", "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    success_count = sum(1 for r in results if r["success"])
    print(f"\nCompleted: {success_count}/{len(INVOICES)} successful")


def cmd_compare(args):
    """Compare Instructor vs normal prompt parsing."""
    print("Comparing Instructor vs Normal Prompt parsing...\n")

    for i, invoice_text in enumerate(INVOICES, 1):
        print(f"=== Invoice {i} ===")
        try:
            instructor_result = parse_with_instructor(invoice_text)
            print(f"Instructor: OK - {instructor_result.invoice_number}, ${instructor_result.total_amount}")
        except Exception as e:
            print(f"Instructor: FAILED - {e}")

        try:
            normal_result = parse_with_normal_prompt(invoice_text)
            print(f"Normal Prompt: OK - {normal_result.invoice_number}, ${normal_result.total_amount}")
        except Exception as e:
            print(f"Normal Prompt: FAILED - {e}")
        print()


def main():
    parser = argparse.ArgumentParser(description="Invoice Parser CLI")
    subparsers = parser.add_subparsers(dest="command", required=True)

    # parse command
    subparsers.add_parser("parse", help="Parse a single invoice").add_argument("text", help="Invoice text")

    # batch command
    subparsers.add_parser("batch", help="Batch parse all invoices")

    # compare command
    subparsers.add_parser("compare", help="Compare parsing methods")

    args = parser.parse_args()

    if args.command == "parse":
        cmd_parse(args)
    elif args.command == "batch":
        cmd_batch(args)
    elif args.command == "compare":
        cmd_compare(args)


if __name__ == "__main__":
    main()
