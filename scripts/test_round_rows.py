"""ROUND_ROWS is a round's row bound: 16 (the pinned kit's widest row kernel) up to 48 (the engine's max_rows).

Above 16 rows the engine runs a gate the kit cannot launch whole as 16-row pieces (0017), and `TF_DS_PLAIN_ROWS`
keeps the cuBLAS linear shapes at 16 so a wider round cannot move a token's bits through cuBLAS's pick.
"""
from pathlib import Path
import subprocess
import unittest

ROOT = Path(__file__).resolve().parents[1]


class RoundRowsTest(unittest.TestCase):
    def test_supported_and_missing_kernel_sizes(self):
        line = next(line for line in (ROOT / "start.sh").read_text().splitlines()
                    if line.startswith('[[ -z "$ROUND_ROWS" ]] || {'))
        for value in ("", "16", "17", "32", "48", "0", "15", "49", "garbage"):
            with self.subTest(value=value):
                result = subprocess.run(["bash", "-c", 'die() { exit 1; }; ROUND_ROWS=$1; ' + line, "test", value],
                                        capture_output=True, text=True)
                self.assertEqual(result.returncode == 0, value in ("", "16", "17", "32", "48"), result.stderr)


if __name__ == "__main__":
    unittest.main()
