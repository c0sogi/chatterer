import os
from contextlib import contextmanager, suppress
from io import BytesIO
from typing import Iterator, Optional

from ..common_types.io import BytesReadable, PathOrReadable, StringReadable


@contextmanager
def read_bytes_stream(
    path_or_file: PathOrReadable,
    assume_pathlike_bytes_as_path: bool = False,
    assume_pathlike_string_as_path: bool = True,
) -> Iterator[Optional[BytesReadable]]:
    """
    Context manager for opening a file or using an existing stream.

    Handles different types of input (file paths, byte streams, string streams)
    and yields a BytesReadable object that can be used to read binary data.

    Args:
        path_or_file: File path or readable object.
        assume_pathlike_bytes_as_path: If True, assume bytes-like objects are file paths. Else, treat as data itself.
        assume_pathlike_string_as_path: If True, assume string-like objects are file paths. Else, treat as data itself.

    Yields:
        Optional[BytesReadable]: A readable binary stream or None if opening fails.
    """
    stream: Optional[BytesReadable] = None
    should_close: bool = True  # Whether the stream should be closed after use
    try:
        with suppress(BaseException):
            if isinstance(path_or_file, BytesReadable):
                # Assume the input is already a bytes stream
                # NOTE: Delivers itself, so shouldn't be closed.
                stream = path_or_file
                should_close = False
            elif isinstance(path_or_file, StringReadable):
                # Convert the string stream to bytes stream
                stream = BytesIO(path_or_file.read().encode("utf-8"))
            elif isinstance(path_or_file, bytes):
                # Convert the bytes-like object to bytes stream
                if assume_pathlike_bytes_as_path and os.path.exists(path_or_file):
                    stream = open(path_or_file, "rb")
                else:
                    stream = BytesIO(path_or_file)
            elif isinstance(path_or_file, str):
                # Convert the file path to bytes stream
                if assume_pathlike_string_as_path and os.path.exists(path_or_file):
                    stream = open(path_or_file, "rb")
                else:
                    stream = BytesIO(path_or_file.encode("utf-8"))
            else:
                # Assume the input is a file descriptor or path
                stream = open(path_or_file, "rb")
        yield stream
    finally:
        if stream is not None and should_close:
            stream.close()
