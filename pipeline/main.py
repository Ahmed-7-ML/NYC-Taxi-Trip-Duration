"""
Entry point for the Prefect-Orchestrated Trip Duration Pipeline
    Usage:
        python main.py --> Full Run with defaults
        python main.py --sample-size 5000 --no-tune
        python main.py --sample-size 20000 --promote
        python main.py --experiment-name my_experiment

What happened when you run this:
    1. Imports and Calls @flow from `flow.py`
    2. Prefect creates a local `FlowRun` (stored in ~/.prefect)
    3. Each @task inside the flow gets its own `TaskRun` with state tracking
    4. Prefect UI Running (Prefect Server Starting) -> http://127.0.0.1:4200

[argparse]:https://docs.python.org/3/library/argparse.html
    - Built-in Python Module used to Parse Command-Line Options, Arguments and Sub-Commands.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

import argparse
from flow import trip_duration_pipeline

def parse_args():
    # It is a container for argument specifications and has options that apply to the parser as whole
    parser = argparse.ArgumentParser(
        prog='Trip Duration Pipeline - Ahmed Akram Amer'و
        description='Trip Duration Pipeline — Prefect Orchestrated',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        # Text at the bottom of help
        epilog="""
    Examples:
    # Quick test run (small data, no tuning)
    python main.py --sample-size 50000 --no-tune

    # Full run with tuning
    python main.py --sample-size 200000 --tune

    # Full run with auto-promotion to production
    python main.py --sample-size 200000 --tune --promote

    # Custom experiment name in MLflow
    python main.py --experiment-name trip_duration_v2

    # Start Prefect UI first, then run
    prefect server start &
    python main.py --sample-size 100000
            """
    )

    # It attaches individual argument specifications to the parser. It supports positional arguments, options that accept values, and on/off flags
    parser.add_argument(
        '--sample-size', type=int, default=20000,
        help='Number of training samples (default: 20000)'
    )

    # action='store_true' -> On Flag
    parser.add_argument(
        '--tune', action='store_true', default=True,
        help='Enable hyperparameter tuning (default: True)'
    )

    # action='store_false' -> Off Flag
    parser.add_argument(
        '--no-tune', action='store_false', dest='tune',
        help='Disable hyperparameter tuning'
    )

    parser.add_argument(
        '--promote', action='store_true', default=False,
        help='Auto-promote best model to Production in MLflow (default: False)'
    )

    parser.add_argument(
        '--experiment-name', type=str, default=None,
        help='Override MLflow experiment name'
    )
    # parser.print_help()
    # It runs the parser and places the extracted data in a argparse.Namespace object
    return parser.parse_args()

if __name__ == '__main__':
    args = parse_args()
    
    print("\n" + "=" * 60)
    print("TRIP DURATION PIPELINE — PREFECT ORCHESTRATED")
    print("=" * 60)
    print(f"  sample_size : {args.sample_size:,}")
    print(f"  tune        : {args.tune}")
    print(f"  promote     : {args.promote}")
    if args.experiment_name:
        print(f"  experiment  : {args.experiment_name}")
    print("=" * 60 + "\n")
    
    result = trip_duration_pipeline(
        sample_size=args.sample_size,
        tune=args.tune,
        promote_to_prod=args.promote,
        experiment_name=args.experiment_name
    )
    
    print("\n" + "=" * 60)
    print("RESULT")
    print("=" * 60)
    for k, v in result.items():
        print(f"  {k}: {v}")
    print("=" * 60)
