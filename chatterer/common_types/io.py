import os
from io import BufferedReader, BufferedWriter, BytesIO, StringIO, TextIOWrapper
from typing import TypeAlias

# Type aliases for callback functions and file descriptors
FileDescriptorOrPath: TypeAlias = int | str | bytes | os.PathLike[str] | os.PathLike[bytes]

# Type aliases for different types of IO objects
BytesReadable: TypeAlias = BytesIO | BufferedReader
BytesWritable: TypeAlias = BytesIO | BufferedWriter
StringReadable: TypeAlias = StringIO | TextIOWrapper
StringWritable: TypeAlias = StringIO | TextIOWrapper

# Combined type aliases for readable and writable objects
Readable: TypeAlias = BytesReadable | StringReadable
Writable: TypeAlias = BytesWritable | StringWritable

# Type alias for path or readable object
PathOrReadable: TypeAlias = FileDescriptorOrPath | Readable
