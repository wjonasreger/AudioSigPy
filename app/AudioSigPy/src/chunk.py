import struct
from typing import Union
import io
from . import utils as Utils


class IFFChunk:
    def __init__(
        self, 
        file, 
        align_to_word: bool=True, 
        use_big_endian: bool=True, 
        exclude_header: bool=False
    ):
        """
        Initialize a new IFFChunk instance.
        
        Theory:
            A file is a collection of data stored on a computer that can be read and processed by software. 
            The IFF (Interchange File Format) is a standard file format used for storing data in chunks, 
            which are smaller, manageable segments of the file. Each chunk in an IFF file has a specific 
            format consisting of a 4-byte chunk ID and a 4-byte chunk size, followed by the actual data. 
            When reading an IFF file, the software reads these chunks sequentially. It first reads the 
            chunk ID to identify the type of data, then the chunk size to determine the length of the data, 
            and finally reads the specified amount of data bytes. These bytes are then interpreted according 
            to the chunk type, converting them from their binary form into human-readable information such 
            as text, images, or audio.

        Arguments:
            file (file-like object): The file-like object to read the chunk from.
            align_to_word (bool, optional): Whether to align to word (2-byte) boundaries. Default is True.
            use_big_endian (bool, optional): Whether to use big-endian byte order. Default is True.
            exclude_header (bool, optional): Whether to exclude the header from the chunk size. Default is False.
            
        Raises:
            TypeError: If file is not a file-like object, or if align_to_word, use_big_endian, or exclude_header are not booleans.
            EOFError: If the chunk ID or chunk size cannot be read.
        """
        
        # Input validation
        if not hasattr(file, 'read'):
            raise TypeError(f"[ERROR] Chunk initialisation: File must be a file-like object. Got {type(file).__name__}.")
        if not isinstance(align_to_word, bool):
            raise TypeError(f"[ERROR] Chunk initialisation: Word alignment must be a boolean. Got {type(align_to_word).__name__}.")
        if not isinstance(use_big_endian, bool):
            raise TypeError(f"[ERROR] Chunk initialisation: Use big endian must be a boolean. Got {type(use_big_endian).__name__}.")
        if not isinstance(exclude_header, bool):
            raise TypeError(f"[ERROR] Chunk initialisation: Exclude header must be a boolean. Got {type(exclude_header).__name__}")
        
        # Initialize internal state and settings based on provided parameters
        self.set_file(file=file)
        self.set_alignment(alignment=align_to_word)
        self.set_byte_order(byte_order='>' if use_big_endian else '<')
        self.set_close_status(close_status=False)
        self.set_bytes_read(bytes_read=0)

        # Read the chunk ID (4 bytes)
        self.set_chunk_id(chunk_id=self.get_file().read(4))
        if len(self.get_chunk_id()) < 4:
            raise EOFError("[ERROR] Chunk initialisation: Failed to read chunk ID. Ensure the file has at least 4 bytes available for the chunk ID.")

        # Read the chunk size (4 bytes)
        try:
            self.set_chunk_size(
                chunk_size=struct.unpack_from(self.get_byte_order() + 'L', self.get_file().read(4))[0]
            )
        except struct.error:
            raise EOFError("[ERROR] Chunk initialisation: Failed to read chunk size. Ensure the file has at least 4 bytes available for the chunk size.")

        # Adjust chunk size if excluding the header
        if exclude_header:
            self.set_chunk_size(chunk_size=self.get_chunk_size() - 8) # subtract header size

        # Determine if the file is seekable
        try:
            self.set_start_offset(start_offset=self.get_file().tell())
        except (AttributeError, OSError):
            self.set_seek_status(seek_status=False)
        else:
            self.set_seek_status(seek_status=True)


    # public methods

    def close(self) -> None:
        """
        Close the chunk and skip to the end.

        This method ensures that the chunk is properly closed by setting the
        `is_closed` attribute to True after attempting to skip to the end of the chunk.
        """
        
        if not self.get_close_status():
            try:
                self.skip()
            finally:
                self.set_close_status(close_status=True)


    def seek(self, position: int, reference_point: int = 0) -> None:
        """
        Seek to the specified position in the chunk.

        Arguments:
            position (int): The position to seek to.
            reference_point (int, optional): The reference point for position. Default is 0 (start of chunk).
            
        Raises:
            ValueError: If the chunk is closed.
            OSError: If the file is not seekable.
            RuntimeError: If the position is out of bounds.
        """
        
        # Check if the chunk is closed
        if self.get_close_status():
            raise ValueError(f"[ERROR] Seek: I/O operation on closed file (Invalid state: file is closed). The file must be open before performing operations.")

        # Check if the file is seekable
        if not self.get_seek_status():
            raise OSError("[ERROR] Seek: file is not seekable.")

        # Adjust position based on the reference point
        if reference_point == 1:
            # Reference point is the current position
            position = position + self.get_bytes_read()
        elif reference_point == 2:
            # Reference point is the end of the chunk
            position = position + self.get_chunk_size()

        # Check if the position is within the bounds of the chunk
        if position < 0 or position > self.get_chunk_size():
            raise RuntimeError(f"[ERROR] Seek: Seek position out of bounds [0, {self.get_chunk_size()}]. Got {position}.")

        # Seek to the new position in the file
        self.get_file().seek(self.get_start_offset() + position, 0)
        # Update the bytes read to the new position
        self.set_bytes_read(bytes_read=position)


    def tell(self) -> int:
        """
        Return the current position in the chunk.

        Returns:
            int: The current position.
            
        Raises:
            ValueError: If the file is closed.
        """
        
        if self.get_close_status():
            raise ValueError("[ERROR] Tell: I/O operation on closed file (Invalid state: file is closed). The file must be open before performing operations.")
        
        bytes_read = self.get_bytes_read()
        
        return bytes_read
    
    
    def skip(self) -> None:
        """
        Skip the rest of the chunk.

        Raises:
            ValueError: If the chunk is closed.
            EOFError: If the end of file is reached unexpectedly.
        """
        
        # Check if the file is closed and raise ValueError if it is
        if self.get_close_status():
            raise ValueError("[ERROR] Skip: I/O operation on closed file (Invalid state: file is closed). The file must be open before performing operations.")

        # Check if the seek status is enabled
        if self.get_seek_status():
            try:
                # Calculate remaining bytes to skip
                remaining_bytes = self.get_chunk_size() - self.get_bytes_read()
                # Adjust for alignment if needed
                if self.get_alignment() and (self.get_chunk_size() & 1):
                    remaining_bytes += 1
                # Skip the remaining bytes in the file
                self.get_file().seek(remaining_bytes, 1)
                # Update bytes read
                self.set_bytes_read(bytes_read=self.get_bytes_read() + remaining_bytes)
                return
            except OSError as e:
                # Raise an OSError if there is an issue during seeking
                raise OSError(f"[ERROR] Skip: {e}.")

        # Read and discard data until the end of the chunk is reached
        while self.get_bytes_read() < self.get_chunk_size():
            remaining_bytes = self.get_chunk_size() - self.get_bytes_read()
            chunk_size = min(8192, remaining_bytes)
            # Read a chunk of data
            data = self.read(size=chunk_size)
            # If no data is read, raise EOFError
            if not data:
                raise EOFError("[ERROR] Skip: Unexpected end of file.")


    def read(self, size: int = -1) -> bytes:
        """
        Read at most size bytes from the chunk.

        Arguments:
            size (int, optional): The number of bytes to read. If negative, read until the end of the chunk.
            
        Returns:
            bytes: The data read from the chunk.
            
        Raises:
            ValueError: If the file is closed.
        """
        
        # Check if the file is closed and raise an error if it is
        if self.get_close_status():
            raise ValueError("[ERROR] Read: I/O operation on closed file (Invalid state: file is closed). The file must be open before performing operations.")

        # Check if all bytes have already been read from the chunk
        if self.get_bytes_read() >= self.get_chunk_size():
            data = b''
            return data

        # Calculate the remaining bytes that can be read from the chunk
        remaining_bytes = self.get_chunk_size() - self.get_bytes_read()

        # Adjust the size to read if it is negative or greater than remaining bytes
        if size < 0 or size > remaining_bytes:
            size = remaining_bytes

        # Read the specified number of bytes from the file
        data = self.get_file().read(size)

        # Update the count of bytes read
        self.set_bytes_read(bytes_read=self.get_bytes_read() + len(data))

        # Check if reading is done and if alignment adjustment is needed
        is_done = self.get_bytes_read() == self.get_chunk_size()
        if is_done and self.get_alignment() and (self.get_chunk_size() & 1):
            temp = self.get_file().read(1)
            self.set_bytes_read(bytes_read=self.get_bytes_read() + len(temp))

        return data
    
    
    # setter methods
    
    def set_chunk_id(self, chunk_id: bytes) -> None:
        """
        Set the chunk ID.
        
        Arguments:
            chunk_id (bytes): ID of the chunk
            
        Raises:
            TypeError: If the attribute is not valid type.
        """
        
        if not isinstance(chunk_id, bytes):
            raise TypeError(f"[ERROR] Set chunk id: Chunk ID must be byte type. Got {type(chunk_id).__name__}.")
        
        self.__chunk_id = chunk_id


    def set_chunk_size(self, chunk_size: int) -> None:
        """
        Set the chunk size.

        Arguments:
            chunk_size (int): Size of the chunk
            
        Raises:
            TypeError: If the attribute is not valid type.
            ValueError: If the attribute is not valid value.
        """
        
        if not isinstance(chunk_size, int):
            raise TypeError(f"[ERROR] Set chunk size: Chunk size must be int type. Got {type(chunk_size).__name__}.")
        if chunk_size < 0:
            raise ValueError(f"[ERROR] Set chunk size: Chunk size must be a non-negative integer. Got {chunk_size}.")
        
        self.__chunk_size = chunk_size


    def set_close_status(self, close_status: bool) -> None:
        """
        Set the close status.

        Arguments:
            close_status (bool): Close status of the file
            
        Raises:
            TypeError: If the attribute is not valid type.
        """
        
        if not isinstance(close_status, bool):
            raise TypeError(f"[ERROR] Set close status: Close status must be bool type. Got {type(close_status).__name__}.")
        
        self.__is_closed = close_status


    def set_seek_status(self, seek_status: bool) -> None:
        """
        Set the seek status.

        Arguments:
            seek_status (bool): Seek status of the file
            
        Raises:
            TypeError: If the attribute is not valid type.
        """
        
        if not isinstance(seek_status, bool):
            raise TypeError(f"[ERROR] Set seek status: Seek status must be bool type. Got {type(seek_status).__name__}.")
        
        self.__is_seekable = seek_status


    def set_file(self, file: Union[io.BufferedReader, io.BufferedWriter]) -> None:
        """
        Set the file.

        Arguments:
            file (IO): A file-like object
            
        Raises:
            TypeError: If the attribute is not valid type.
        """
        
        if not isinstance(file, (io.BufferedReader, io.BufferedWriter)):
            try:
                file = file.get_file()
                if not isinstance(file, (io.BufferedReader, io.BufferedWriter)):
                    raise TypeError(f"[ERROR] Set file: File must be IO Buffered Reader/Writer type. Got {type(file).__name__}.")
            except:
                raise TypeError(f"[ERROR] Set file: File must be IO Buffered Reader/Writer type. Got {type(file).__name__}.")
        
        self.__file = file


    def set_bytes_read(self, bytes_read: int) -> None:
        """
        Set the number of bytes read.

        Arguments:
            bytes_read (int): Number of bytes read from the file
            
        Raises:
            TypeError: If the attribute is not valid type.
            ValueError: If the attribute is not valid value.
        """
        
        if not isinstance(bytes_read, int):
            raise TypeError(f"[ERROR] Set bytes read: Bytes read must be int type. Got {type(bytes_read).__name__}.")
        if bytes_read < 0:
            raise ValueError(f"[ERROR] Set bytes read: Bytes read must be a non-negative integer. Got {bytes_read}.")
        
        self.__bytes_read = bytes_read


    def set_start_offset(self, start_offset: int) -> None:
        """
        Set the start offset.

        Arguments:
            start_offset (int): Start offset in the file
            
        Raises:
            TypeError: If the attribute is not valid type.
            ValueError: If the attribute is not valid value.
        """
        
        if not isinstance(start_offset, int):
            raise TypeError(f"[ERROR] Set start offset: Start offset must be an int type. Got {type(start_offset).__name__}.")
        if start_offset < 0:
            raise ValueError(f"[ERROR] Set start offset: Start offset must be a non-negative integer. Got {start_offset}.")
        
        self.__start_offset = start_offset


    def set_alignment(self, alignment: bool) -> None:
        """
        Set the alignment.

        Arguments:
            alignment (bool): Alignment to word boundary
            
        Raises:
            TypeError: If the attribute is not valid type.
        """
        
        if not isinstance(alignment, bool):
            raise TypeError(f"[ERROR] Set alignment: Alignment must be a bool type. Got {type(alignment).__name__}.")
        
        self.__align_to_word = alignment


    def set_byte_order(self, byte_order: str) -> None:
        """
        Set the byte order.

        Arguments:
            byte_order (str): Byte order (e.g., '>', '<')
            
        Raises:
            TypeError: If the attribute is not valid type.
            ValueError: If the attribute is not valid value.
        """
        
        if not isinstance(byte_order, str):
            raise TypeError(f"[ERROR] Set byte order: Byte order must be a str type. Got {type(byte_order).__name__}.")
        if byte_order not in ["<", ">"]:
            raise ValueError(f"[ERROR] Set byte order: Byte order must be one of '<' or '>'. Got {byte_order}.")
        
        self.__byte_order = byte_order


    # getter methods
    
    def get_chunk_id(self) -> bytes:
        """
        Return the ID (name) of the current chunk.

        Returns:
            bytes: The chunk ID.
            
        Raises:
            ValueError: If the attribute is not set.
        """
        
        if self.__chunk_id is None:
            raise ValueError("[ERROR] Get chunk id: Chunk ID not set.")
        
        return self.__chunk_id


    def get_chunk_size(self) -> int:
        """
        Return the size of the current chunk.

        Returns:
            int: The chunk size.
            
        Raises:
            ValueError: If the attribute is not set.
        """
        
        if self.__chunk_size is None:
            raise ValueError("[ERROR] Get chunk size: Chunk size not set.")
        
        return self.__chunk_size


    def get_close_status(self) -> bool:
        """
        Returns the closed status of the file.
        
        Returns:
            bool: True if the file is closed, False otherwise.
            
        Raises:
            ValueError: If the attribute is not set.
        """
        
        if self.__is_closed is None:
            raise ValueError("[ERROR] Get close status: Close status not set.")
        
        return self.__is_closed


    def get_seek_status(self) -> bool:
        """
        Returns the seekable status of the file.
        
        Returns:
            bool: True if the file is seekable, False otherwise.
            
        Raises:
            ValueError: If the attribute is not set.
        """
        
        if self.__is_seekable is None:
            raise ValueError("[ERROR] Get seek status: Seek status not set.")
        
        return self.__is_seekable


    def get_file(self) -> Union[io.BufferedReader, io.BufferedWriter]:
        """
        Returns the file-like object.
        
        Returns:
            IO: The file-like object.
            
        Raises:
            ValueError: If the attribute is not set.
        """
        
        if self.__file is None:
            raise ValueError("[ERROR] Get file: File not set.")
        
        return self.__file


    def get_bytes_read(self) -> int:
        """
        Returns the number of bytes read.
        
        Returns:
            int: The number of bytes read from the file.
            
        Raises:
            ValueError: If the attribute is not set.
        """
        
        if self.__bytes_read is None:
            raise ValueError("[ERROR] Get bytes read: Bytes read not set.")
        
        return self.__bytes_read


    def get_start_offset(self) -> int:
        """
        Returns the start offset.
        
        Returns:
            int: The start offset in the file.
            
        Raises:
            ValueError: If the attribute is not set.
        """
        
        if self.__start_offset is None:
            raise ValueError("[ERROR] Get start offset: Start offset not set.")
        
        return self.__start_offset


    def get_alignment(self) -> bool:
        """
        Returns the alignment status.
        
        Returns:
            bool: True if aligned to word, False otherwise.
            
        Raises:
            ValueError: If the attribute is not set.
        """
        
        if self.__align_to_word is None:
            raise ValueError("[ERROR] Get alignment: Alignment not set.")
        
        return self.__align_to_word


    def get_byte_order(self) -> str:
        """
        Returns the byte order.
        
        Returns:
            str: The byte order (e.g., '>', '<').
            
        Raises:
            ValueError: If the attribute is not set.
        """
        
        if self.__byte_order is None:
            raise ValueError("[ERROR] Get byte order: Byte order not set.")
        
        return self.__byte_order

    
    # representation methods
    
    def get_parameters(self) -> dict:
        """
        Return the parameters of the IFFChunk object.

        Returns:
            dict: Dictionary of parameters and their values.
            
        Raises:
            ValueError: If not all parameters are set.
        """
        
        params = {
            "chunk_id": self.get_chunk_id(),
            "chunk_size": self.get_chunk_size(),
            "is_closed": self.get_close_status(),
            "is_seekable": self.get_seek_status(),
            "bytes_read": self.get_bytes_read(),
            "start_offset": self.get_start_offset(),
            "align_to_word": self.get_alignment(),
            "byte_order": self.get_byte_order()
        }
        
        if not all(params):
            raise ValueError("[ERROR] Get parameters: Not all parameters are set.")
        
        return params
        
        
    def __repr__(self) -> str:
        """
        Return a string representation of the IFFChunk object for debugging.

        Returns:
            str: String representation of the object.
        """
        
        developer_repr = Utils.base_repr(self)
        
        return developer_repr


    def __str__(self) -> str:
        """
        Return a string representation of the IFFChunk object for the end user.

        Returns:
            str: User-friendly string representation of the object.
        """
        
        user_repr = Utils.base_str(self)
        
        return user_repr