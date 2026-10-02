"""Command-line options shared by the Phase 2 download steps."""
import argparse
import datetime as dt

import config


def parse_dates(description, product_choices=None):
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument("--start", default=config.START_DATE,
                        help=f"first date YYYY-MM-DD (default {config.START_DATE})")
    parser.add_argument("--end", default=config.END_DATE,
                        help="last date YYYY-MM-DD (default: yesterday)")
    if product_choices:
        parser.add_argument("--products", nargs="+", default=product_choices,
                            choices=product_choices, help="which products to run")
    args = parser.parse_args()
    if args.end is None:
        args.end = (dt.date.today() - dt.timedelta(days=1)).isoformat()
    return args
