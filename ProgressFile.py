from pathlib import Path
from tqdm.auto import tqdm
from typing import Callable, IO, List, Literal, Tuple
import bisect
import os


class ProgressFile:
    """
    A file-like object that wraps a real file and reports read progress using a tqdm
    progress bar.
    """

    def __init__(
            self,
            file: str | Path | int,
            mode: Literal[
                'r', 'rb', 'rt', 'r+', 'r+b', 'r+t', 'w+', 'w+b', 'w+t', 'a+', 'a+b', 'a+t'
            ] = 'r',
            buffering: int = -1,
            encoding: str | None = None,
            errors: str = None,
            newline: Literal['', r'\n', r'\r', r'\r\n'] | None = None,
            closefd: bool = True,
            opener: Callable[[str | Path, int], int] | None = None,
            desc: str | None = None,
            total: int | None = None,
            unit_scale: int | bool | None = True,
            **tqdm_kwargs
    ):
        """Initialize an instance of the ProgressFile class.

        :param file: Path to the file to be opened or integer file descriptor.
        :param mode: File mode, typically 'rb' for reading in binary. This class allows
               for creating an updatable file but doesn't support writing.
        :param buffering: 0: none (binary only), 1: line (text only), >1: size in bytes
               of chunk buffer.
        :param encoding: defaults to whatever locale.getencoding() returns.
        :param errors: text mode only. Valid values are names passed into
               codecs.register_error().
        :param newline: see open() docs for what the different values mean.
        :param closefd: Set to falls if the file passed in is a file like object that
               you don't want closed.
        :param opener: If not
        :param desc: Optional description for the tqdm progress bar.
        :param total: Total size of the file in bytes (inferred if not provided).
        :param unit_scale: If 1 or True, the number of iterations will be reduced/scaled
               automatically and a metric prefix following the International System of
               Units standard will be added (kilo, mega, etc.) [default: True]. If any
               other non-zero number, will scale total and n. The default action here is
               different from tqdm because files tend to be large enough that byte values
               become harder to interpret.
        :param tqdm_kwargs: Additional keyword arguments to pass to tqdm. These can be
               (at the time of this document creation):  "leave", "file", "ncols",
               "mininterval", "maxinterval", "miniters", "ascii", "disable", "unit",
               "dynamic_ncols", "smoothing", "bar_format", "initial", "position",
               "postfix", "unit_devisor", "write_bytes", "lock_arg", "nrows", "colour",
               and "delay".
        :return: Tqdm-displaying file-like object with context manager support suitable
                 for reading a file while displaying read progress.
        """
        if not mode in [
            'r', 'rb', 'rt', 'r+', 'r+b', 'r+t', 'w+', 'w+b', 'w+t', 'a+', 'a+b', 'a+t']:
            raise ValueError("must open for reading.")
        self._file: IO = open(
            file, mode, buffering=buffering, encoding=encoding, errors=errors,
            newline=newline, closefd=closefd, opener=opener)
        self._filename: str | None = (
            str(Path(file).absolute()) if isinstance(file, (str, Path)) else None)
        if not self._file.readable() and self._file.seekable():
            self._file.close()
            f = self._filename if self._filename else "file"
            raise RuntimeError(f'Opened {f} in mode {mode} is not readable and seekable')
        self._total: int = total or os.path.getsize(file)
        unit_scale = True if unit_scale is None else unit_scale
        self._pbar: tqdm = tqdm(
            total=self._total, desc=desc or file, unit_scale=unit_scale, **tqdm_kwargs)
        self._read_regions: List[Tuple[int, int]] = []

    def read(self, size: int = -1) -> bytes:
        """Read bytes from the file and update progress bar if new data is read.

        :param size: Number of bytes to read (-1 to read all).
        :return: Bytes read.
        """
        start = self._file.tell()
        data = self._file.read(size)
        end = self._file.tell()
        self._update_progress(start, end)
        return data

    def readline(self, size: int = -1) -> bytes:
        """Read a line from the file and update progress.

        :param size: Maximum number of bytes to read.
        :return: Line read as bytes.
        """
        start = self._file.tell()
        line = self._file.readline(size)
        end = self._file.tell()
        self._update_progress(start, end)
        return line

    def readlines(self, hint: int = -1) -> List[bytes]:
        """Read multiple lines and update progress.

        :param hint: Optional hint for total number of bytes to read.
        :return: List of lines read as bytes.
        """
        start = self._file.tell()
        lines = self._file.readlines(hint)
        end = self._file.tell()
        self._update_progress(start, end)
        return lines

    def seek(self, offset: int, whence: int = os.SEEK_SET) -> int:
        """Move file pointer to a specific position.

        :param offset: Offset in bytes.
        :param whence: Where to seek from (start, current, end).
        :return: New position.
        """
        result = self._file.seek(offset, whence)
        return result

    def tell(self) -> int:
        """Gets the current byte position in the file.
        :return: Current byte position in file.
        """
        return self._file.tell()

    def close(self) -> None:
        """Close the file and the progress bar.
        """
        self._pbar.close()
        self._file.close()

    def _update_progress(self, start: int, end: int) -> None:
        """Internal method to update progress bar based on new read range.

        :param start: Start byte position.
        :param end: End byte position.
        """
        new_bytes = self._add_read_region(start, end)
        if new_bytes > 0:
            self._pbar.update(new_bytes)

    def _add_read_region(self, start: int, end: int) -> int:
        """Track read region and return the number of *new* bytes read.

        :param start: Start byte index.
        :param end: End byte index.
        :return: Number of previously unread bytes.
        """
        if start >= end:
            return 0

        regions = self._read_regions
        new_bytes = end - start

        # Binary search for insertion points
        i = bisect.bisect_left(regions, (start, start))
        # noinspection PyTypeChecker
        j = bisect.bisect_right(regions, (end, float('inf')))

        # Merge overlapping regions
        overlap = regions[i:j]
        for r_start, r_end in overlap:
            overlap_start = max(start, r_start)
            overlap_end = min(end, r_end)
            if overlap_start < overlap_end:
                new_bytes -= (overlap_end - overlap_start)
            start = min(start, r_start)
            end = max(end, r_end)

        # Replace the overlapping regions with the new merged one
        self._read_regions = regions[:i] + [(start, end)] + regions[j:]
        return max(0, new_bytes)

    def __enter__(self) -> "ProgressFile":
        """
        :return: Self (context manager support).
        """
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        self.close()

    def fileno(self) -> int:
        """Gets the underlying file descriptor.
        :return: Underlying file descriptor.
        """
        return self._file.fileno()

    def readable(self) -> bool:
        """Gets a value indicating whether the file is readable.
        :return: Whether the file is readable.
        """
        return True

    def seekable(self) -> bool:
        """Gets a value indicating whether the file supports seeking.
        :return: Whether the file supports seeking.
        """
        return True
