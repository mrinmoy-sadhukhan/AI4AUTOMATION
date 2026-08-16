#!/usr/bin/env python3
"""Inspect accepted/rejected NL2PLC algorithm records in the SQLite database."""

from __future__ import annotations

import argparse
import json
import sqlite3
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--database",
        type=Path,
        default=Path(
            "data/gold_algorithm_database/gold_algorithms.sqlite3"
        ),
    )
    parser.add_argument("--id", dest="example_id")
    parser.add_argument("--limit", type=int, default=10)
    args = parser.parse_args()

    if not args.database.is_file():
        raise FileNotFoundError(args.database)
    connection = sqlite3.connect(args.database)
    connection.row_factory = sqlite3.Row
    try:
        if args.example_id:
            source = connection.execute(
                "SELECT * FROM source_examples WHERE example_id = ?",
                (args.example_id,),
            ).fetchone()
            if source is None:
                raise ValueError(f"Unknown example id: {args.example_id}")
            accepted = connection.execute(
                """
                SELECT a.*, r.model, r.validation_json
                FROM accepted_algorithms AS a
                JOIN teacher_runs AS r USING (run_id)
                WHERE a.example_id = ?
                """,
                (args.example_id,),
            ).fetchone()
            runs = connection.execute(
                """
                SELECT run_id, attempt_number, model, status, error,
                       validation_json
                FROM teacher_runs
                WHERE example_id = ?
                ORDER BY run_id
                """,
                (args.example_id,),
            ).fetchall()
            result = {
                "id": source["example_id"],
                "source_row": source["source_row"],
                "split": source["split"],
                "instruction": source["instruction"],
                "addresses": json.loads(source["addresses_json"]),
                "gold_st": source["gold_st"],
                "accepted_algorithm": (
                    json.loads(accepted["algorithm_json"])
                    if accepted
                    else None
                ),
                "runs": [
                    {
                        **dict(run),
                        "validation_json": json.loads(
                            run["validation_json"]
                        ),
                    }
                    for run in runs
                ],
            }
            print(json.dumps(result, ensure_ascii=False, indent=2))
            return

        summary = connection.execute(
            """
            SELECT
                s.example_id,
                s.source_row,
                s.split,
                s.logic_family,
                CASE WHEN a.example_id IS NULL
                     THEN 'unaccepted' ELSE 'accepted' END AS status,
                COUNT(r.run_id) AS attempts
            FROM source_examples AS s
            LEFT JOIN accepted_algorithms AS a USING (example_id)
            LEFT JOIN teacher_runs AS r USING (example_id)
            GROUP BY s.example_id
            ORDER BY s.example_id
            LIMIT ?
            """,
            (args.limit,),
        ).fetchall()
        for row in summary:
            print(
                f"{row['example_id']:16} "
                f"row={str(row['source_row']):4} "
                f"split={row['split']:10} "
                f"family={row['logic_family']:16} "
                f"status={row['status']:10} "
                f"attempts={row['attempts']}"
            )
    finally:
        connection.close()


if __name__ == "__main__":
    main()
