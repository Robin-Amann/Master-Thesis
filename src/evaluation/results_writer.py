from pathlib import Path
from datetime import datetime
import json

class ResultsWriter :
    def __init__(self, directory: str | Path, metadata: dict | None, filename: str, prefix: str = "results"):
        self.directory = Path(directory)
        self.directory.mkdir(parents=True, exist_ok=True)

        self.prefix = prefix
        self.file_index = 0
        self.file = None

        self._open_new_file(metadata, filename)


    def _open_new_file(self, filename: str | None = None) -> None:
        """Create and open a new results file."""
        if filename == None :
            filename = f"{self.prefix}_{self.file_index:04d}.json"
        self.file_index += 1

        self.file = (self.directory / filename).open(mode="w", encoding="utf-8")


    def write(self, results: dict, end: str = "\n") -> None:
        """Append results to the current file."""
        if self.file is None or self.file.closed:
            raise RuntimeError("ResultsWriter is closed.")

        print(json.dumps(results), end=end, file=self.file)
        self.file.flush()


    def close(self, next_filename: str | None = None) -> None:
        """Close the current file and start a new one."""
        if self.file is not None and not self.file.closed:
            self.file.close()

        self._open_new_file(next_filename)


    def finish(self) -> None:
        """Close the writer without creating another file."""
        if self.file is not None and not self.file.closed:
            self.file.close()


    def __enter__(self):
        return self


    def __exit__(self, exc_type, exc_value, traceback):
        self.finish()