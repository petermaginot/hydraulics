"""
interpolate_profile.py
----------------------
Standalone utility to insert n equally-spaced interpolated rows between
every pair of existing rows in a pipeline survey CSV file.

All numeric columns are linearly interpolated.  Non-numeric columns
(e.g. a text ID column) are left blank in the inserted rows; the
original rows keep their original values.

Usage (command line):
    python interpolate_profile.py <input_csv> <n> [output_csv]

    input_csv  -- path to the source CSV file
    n          -- number of points to insert between each existing pair
    output_csv -- optional output path; defaults to
                  <input_stem>_interp<n>.<ext>

Usage (as a module):
    from interpolate_profile import interpolate_profile_csv
    interpolate_profile_csv("survey.csv", n=9)
    interpolate_profile_csv("survey.csv", n=4, output_path="survey_fine.csv")
"""

import csv
import os
import sys


def interpolate_profile_csv(input_path, n, output_path=None):
    """Insert n interpolated rows between every consecutive pair of rows in a
    pipeline survey CSV file and write the result to a new CSV file.

    Columns whose values cannot be converted to float are treated as
    non-numeric: original rows keep their original string value, and
    inserted rows receive an empty string for that column.

    Linear interpolation is used for all numeric columns.  If a column is
    numeric in one row but non-numeric in an adjacent row the column is
    treated as non-numeric for that interval and left blank in inserted rows.

    Args:
        input_path  : str -- path to the input CSV file.
        n           : int -- number of points to insert between each pair of
                      existing rows.  Must be >= 1.
        output_path : str or None -- path for the output file.  If None, the
                      output is written alongside the input file with the
                      suffix "_interp<n>" appended to the stem.

    Returns:
        str -- the path of the output file that was written.

    Raises:
        ValueError  : if n < 1 or the input file has fewer than 2 data rows.
        FileNotFoundError : if input_path does not exist.
    """
    if n < 1:
        raise ValueError(f"n must be >= 1 (received n={n}).")

    if not os.path.isfile(input_path):
        raise FileNotFoundError(f"Input file not found: '{input_path}'")

    # ------------------------------------------------------------------
    # Build default output path if not supplied.
    # ------------------------------------------------------------------
    if output_path is None:
        stem, ext = os.path.splitext(input_path)
        output_path = f"{stem}_interp{n}{ext}"

    # ------------------------------------------------------------------
    # Read the input CSV.
    # ------------------------------------------------------------------
    with open(input_path, newline="") as f:
        reader = csv.reader(f)
        header = next(reader)
        data_rows = list(reader)

    if len(data_rows) < 2:
        raise ValueError(
            f"Input file '{input_path}' must contain at least 2 data rows "
            f"(found {len(data_rows)})."
        )

    n_cols = len(header)

    # ------------------------------------------------------------------
    # Helper: attempt to parse a cell as float; return None on failure.
    # ------------------------------------------------------------------
    def _to_float(cell):
        try:
            return float(cell)
        except (ValueError, TypeError):
            return None

    # ------------------------------------------------------------------
    # Write output CSV.
    # ------------------------------------------------------------------
    output_rows = []

    for row_idx in range(len(data_rows)):
        row_a = data_rows[row_idx]

        # Pad short rows to n_cols so indexing is always safe.
        while len(row_a) < n_cols:
            row_a.append("")

        # Always emit the original row unchanged.
        output_rows.append(row_a)

        # If this is the last original row, nothing to interpolate after it.
        if row_idx == len(data_rows) - 1:
            break

        row_b = data_rows[row_idx + 1]
        while len(row_b) < n_cols:
            row_b.append("")

        # Parse both rows once; cache as floats or None.
        vals_a = [_to_float(row_a[c]) for c in range(n_cols)]
        vals_b = [_to_float(row_b[c]) for c in range(n_cols)]

        # Insert n interpolated rows between row_a and row_b.
        # Fractional positions: 1/(n+1), 2/(n+1), ..., n/(n+1)
        for k in range(1, n + 1):
            t = k / (n + 1)          # interpolation parameter in (0, 1)
            interp_row = []
            for c in range(n_cols):
                a = vals_a[c]
                b = vals_b[c]
                if a is not None and b is not None:
                    # Both endpoints are numeric -- interpolate.
                    interp_val = a + t * (b - a)
                    # Preserve the formatting style of the original value:
                    # if either endpoint looks like an integer string, write
                    # an integer; otherwise use enough decimal places to
                    # avoid rounding the least-significant digit present.
                    n_dec = max(_count_decimals(row_a[c]),
                                _count_decimals(row_b[c]))
                    if n_dec == 0:
                        interp_row.append(str(int(round(interp_val))))
                    else:
                        interp_row.append(f"{interp_val:.{n_dec}f}")
                else:
                    # Non-numeric column -- leave blank in inserted rows.
                    interp_row.append("")
            output_rows.append(interp_row)

    # ------------------------------------------------------------------
    # Write to output file.
    # ------------------------------------------------------------------
    with open(output_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(header)
        writer.writerows(output_rows)

    n_original  = len(data_rows)
    n_inserted  = (n_original - 1) * n
    n_total     = n_original + n_inserted
    print(
        f"  interpolate_profile_csv: {n_original} original rows + "
        f"{n_inserted} inserted rows = {n_total} total rows written to "
        f"'{output_path}'."
    )
    return output_path


def _count_decimals(cell_str):
    """Return the number of decimal places in a numeric string.

    Returns 0 for integers, non-numeric strings, or empty strings.

    Examples:
        '3.068'  -> 3
        '42'     -> 0
        '1e-3'   -> 0   (scientific notation: treated as 0 for simplicity)
        ''       -> 0
    """
    cell_str = cell_str.strip()
    if "." in cell_str and "e" not in cell_str.lower():
        return len(cell_str.split(".")[1])
    return 0


# ---------------------------------------------------------------------------
# Command-line entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    # if len(sys.argv) < 3:
    #     print(
    #         "Usage: python interpolate_profile.py <input_csv> <n> [output_csv]"
    #     )
    #     sys.exit(1)

    # _input   = sys.argv[1]
    # _n       = int(sys.argv[2])
    # _output  = sys.argv[3] if len(sys.argv) >= 4 else None

    interpolate_profile_csv("Example_Well_Survey2.csv", 9, "Example_Well_Survey2_interp.csv")
