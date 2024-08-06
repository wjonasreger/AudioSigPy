import struct
import sys
import io
from typing import Union, Optional
import audioop
import numpy as np
from .chunk import IFFChunk as Chunk
from . import utils as Utils


class Wave():
    def __init__(self, filename: str):
        """
        Initialize the wave object with a filename.

        Arguments:
            filename (str): The name of the file.

        Raises:
            TypeError: If filename is not a string.
            ValueError: If filename is not a non-empty string.
        """
        
        if not isinstance(filename, str):
            raise TypeError(f"[ERROR] Wave initialisation: Filename must be a string. Got {type(filename).__name__}.")
        
        if filename == "":
            raise ValueError("[ERROR] Wave initialisation: Filename must be a non-empty string.")
        
        self.set_filename(filename=filename)
        
        
    # public methods

    def open(self, mode: str) -> None:
        """
        Opens the file with the specified mode.

        Arguments:
            mode (str): Mode to open the file in. Must be 'rb' or 'wb'.

        Raises:
            ValueError: If mode is not 'rb' or 'wb'.
        """
        
        self.set_mode(mode=mode)
        
        # Touch the file according the read/write mode to initialize parameters for chunking
        if self.get_mode() == "rb":
            self.__touch_file(read_mode=True)
        elif self.get_mode() == "wb":
            self.__touch_file(read_mode=False)
        else:
            raise ValueError(
                f"[ERROR] Open wave: Mode must be 'rb' or 'wb'. Got {mode}."
            )


    def close(self) -> None:
        """
        Close the file safely.

        Summary:
            Ensures the file is properly closed by flushing and writing headers if needed,
            then sets the file attribute to None and closes the file copy if it exists.
        """
        
        try:
            # Check if there is an open file and if it's in write mode
            if self.get_file() and self.get_mode() == "write":
                # Write an empty header
                self.__write_header(length=0)
                # Patch the header
                self.__patch_header()
                # Flush any remaining data to the file
                self.get_file().flush()
        finally:
            # Set the main file attribute to None
            self.set_file(file=None)
            # Get the file copy
            temp_file = self.get_file_copy()
            if temp_file:
                # Clear the file copy attribute
                self.set_file_copy(file=None)
                # Close the file copy
                temp_file.close()
                
     
    def read_samples(self, sample_count: int) -> bytes:
        """
        Read a specified number of samples from the audio chunk.

        Arguments:
            sample_count (int): The number of samples to read.

        Returns:
            (bytes): The raw audio data read from the chunk.
            
        Raises:
            TypeError: If sample count is not an integer.
            ValueError: If sample count is negative.
        """
        
        if not isinstance(sample_count, int):
            raise TypeError(f"[ERROR] Read samples: Sample count must be int. Got {type(sample_count).__name__}.")
        
        if sample_count < 0:
            raise ValueError(f"[ERROR] Read samples: Sample count must be non-negative. Got {sample_count}.")
        
        # Check if seek status is True and handle seek position
        if self.get_seek_status():
            # Reset chunk seek to the beginning
            self.get_chunk().seek(position=0, reference_point=0)
            
            # Calculate the seek position based on current position and sample size
            position = self.get_position() * self.get_sample_size()
            if position:
                self.get_chunk().seek(position=position, reference_point=0)
                
            # Update seek status to False after handling
            self.set_seek_status(seek_status=False)
            
        # Return empty bytes if sample_count is 0
        if sample_count == 0:
            return b''
        
        # Read the required number of bytes from the audio chunk
        data = self.get_chunk().read(sample_count * self.get_sample_size())
        
        # Handle byteswap for systems with big-endian byte order if sample width is greater than 1
        if self.get_sample_width() != 1 and sys.byteorder == 'big':
            data = audioop.byteswap(data, self.get_sample_width())
            
        # Calculate the number of samples read and update the position accordingly
        delta = len(data) // (self.get_channel_count() * self.get_sample_width())
        self.set_position(
            position=self.get_position() + delta
        )
        
        # Return the read audio data
        return data

      
    def write_samples(self, data: Union[bytes, bytearray, memoryview]) -> None:
        """
        Write audio samples to the file.

        Arguments:
            data (Union[bytes, bytearray, memoryview]): Audio data to write.
            
        Raises:
            Exception: If data fails to cast to bytes.
        """
        
        # Ensure data is in bytes format for consistency
        if not isinstance(data, (bytes, bytearray)):
            try:
                data = memoryview(data).cast('B')
            except Exception as e:
                raise Exception(f"[ERROR] Write samples: Failed to cast data to bytes. More info: {e}.")
            
        # Write header with the length of the data
        self.__write_header(length=len(data))
        
        # Calculate the number of samples from the data length
        sample_count = len(data) // (self.get_sample_width() * self.get_channel_count())
        
        # If system byte order is big-endian and sample width is not 1, swap bytes for compatibility
        if self.get_sample_width() != 1 and sys.byteorder == "big":
            data = audioop.byteswap(data, self.get_sample_width())
            
        # Write the audio data to the file
        self.get_file().write(data)
        
        # Update the position and samples written counters
        self.set_position(position=self.get_position() + len(data), check=False)
        self.set_samples_written(samples_written=self.get_samples_written() + sample_count)
        
        # Update the header to reflect changes
        self.__patch_header()
            
    
    def decode(self, raw_bytes: bytes) -> np.ndarray:
        """
        Decodes raw bytes into an array of floating point amplitude values.
        
        Decodes raw byte data into a 2D array of floating point amplitude values. It first calculates 
        the total number of samples and determines the sample width in bits. For 24-bit samples, it 
        unpacks the bytes into integers with an additional byte for handling, while for other widths, 
        it unpacks the bytes directly. It then dequantises the data by adjusting the integer values 
        based on the sample width, converting these integers into floating point values by normalizing 
        them with a calculated factor. Finally, it reshapes the array to match the sample count and 
        channel count, returning the resulting 2D array.

        Arguments:
            raw_bytes (bytes): Raw bytes from read samples method.

        Returns:
            np.ndarray: A 2D array with shape (sample_count, channel_count) containing 
            the decoded floating point amplitude values.
            
        Raises:
            Exception: If raw_bytes fails to cast to bytes.
        """
        
        # Ensure data is in bytes format for consistency
        if not isinstance(raw_bytes, (bytes, bytearray)):
            try:
                raw_bytes = memoryview(raw_bytes).cast('B')
            except Exception as e:
                raise Exception(f"[ERROR] Decode: Failed to cast raw_bytes to bytes. More info: {e}.")
        
        # Calculate the total number of samples
        total_samples = self.get_sample_count() * self.get_channel_count()
        sample_width = self.get_sample_width() * 8
        
        # Convert bytes to integers based on sample width
        if sample_width == 24:
            # For 24-bit samples, extract each byte and convert to integers
            format_ = self.__make_format(bits=sample_width, total_samples=1)
            byte_shift = int(sample_width / 8)
        
            raw_integers = []
            for i in range(0, len(raw_bytes), byte_shift):
                byte_temp = raw_bytes[i:i+byte_shift]
                # Append the unpacked integer value to the list
                unpacked_value = struct.unpack(format_, b"\x00" + byte_temp)[0]
                raw_integers.append(unpacked_value)
                
            # Calculate the total bits after dequantisation
            bits_dequant = sample_width + 8
            
        else:
            # For other sample widths, directly unpack the bytes to integers
            format_ = self.__make_format(bits=sample_width, total_samples=total_samples)
            raw_integers = struct.unpack(format_, raw_bytes)
            # Total bits after dequantisation is equal to sample width
            bits_dequant = sample_width
            
        # Dequantise the data based on the sample width
        if sample_width > 8:
            shift = 0
        else:
            shift = int(2 ** sample_width / 2)
            
        # Convert integer data to floating point amplitude values array
        array = np.array([float(int_quant - shift) / pow(2, bits_dequant - 1) for int_quant in raw_integers])
        # Reshape the array array to match sample count and channel count
        array.shape = (self.get_sample_count(), self.get_channel_count())
        
        return array
    
    
    def encode(self, array: np.ndarray) -> bytes:
        """
        Encode the given array into raw bytes.
        
        Encodes a given 2D array of floating point amplitude values into a raw byte stream. It 
        begins by calculating the total number of samples and determining the sample width in 
        bits. The array is reshaped into a 1-dimensional array, and the floating point values 
        are quantized based on the sample width, adjusting for integer conversion. The quantized 
        values are converted into integers and then into bytes, with specific handling for 
        24-bit samples to ensure proper byte alignment. Finally, the byte stream is constructed 
        and returned as the encoded result.

        Arguments:
            array (np.ndarray): Input array to be encoded.

        Returns:
            bytes: Encoded byte stream.
            
        Raises:
            Exception: If array fails to cast to numpy array.
        """
        
        # Ensure data is in numpy.ndarray format for consistency
        if not isinstance(array, np.ndarray):
            try:
                array = np.array(array)
            except Exception as e:
                raise Exception(f"[ERROR] Encode: Failed to cast array to numpy array. More info: {e}.")
        
        # Calculate total number of samples
        total_samples = self.get_sample_count() * self.get_channel_count()
        
        sample_width = self.get_sample_width() * 8
            
        # Reshape the array into a 1-dimensional array
        array.shape = (1, total_samples)
        
        # Quantise the array with the given sample width
        if sample_width > 8:
            shift = 0
        else:
            shift = int(2 ** sample_width / 2)
            
        # Convert the scaled floating-point values to raw integers
        raw_integers = [int(sp * (pow(2, sample_width - 1)) - 1) + shift for sp in np.nditer(array)]
        
        # Convert integers to bytes based on sample width
        if sample_width == 24:
            format_ = self.__make_format(bits=sample_width, total_samples=1)
            raw_bytes = struct.pack(format_, raw_integers.pop(0))[:3]
            for n in raw_integers:
                raw_bytes = raw_bytes + struct.pack("i", n)[:3]
        else:
            format_ = self.__make_format(bits=sample_width, total_samples=total_samples)
            raw_bytes = struct.pack(format_, *raw_integers)
            
        return raw_bytes

        
    # private methods
    
    def __file_params(self, file: Union[io.BufferedReader, io.BufferedWriter], read_mode: bool) -> None:
        """
        Summary:
            Configure file parameters for reading or writing a wave file.
        
        Arguments:
            file (Chunk): The file to be read or written.
            read_mode (bool): Flag to indicate read mode. Must be of type bool.
        
        Raises:
            TypeError: If read_mode is not a boolean.
            ValueError: If the file does not start with 'RIFF' ID or is not of wave type.
                        If data chunk appears before format chunk or either format or data chunk is missing.
        """
        
        # Validate that the read argument is of type bool
        if not isinstance(read_mode, bool):
            raise TypeError(f"[ERROR] File parameters: Read/Write mode must be set with valid type. Got {type(read_mode).__name__}.")
        
        # Set default writing parameters
        self.set_header_status(header_status=False)
        self.set_data_length(data_length=0)
        self.set_position(position=0, check=False)
        self.set_samples_written(samples_written=0)
        
        # Set default wave parameters
        self.set_sample_count(sample_count=0, check=False)
        self.set_sample_rate(sample_rate=0, check=False)
        self.set_sample_width(sample_width=0, check=False)
        self.set_channel_count(channel_count=0, check=False)
        
        if read_mode:
            # Set initial reading parameters
            self.set_format_status(format_status=False)
            self.set_chunk(chunk=None)
            
            # Initialize the file for reading and verify its format
            self.set_file(file=Chunk(file=file, use_big_endian=False))
            if self.get_file().get_chunk_id() != b'RIFF':
                raise ValueError(f"[ERROR] File parameters: File does not start with RIFF ID. Got {self.get_file().get_chunk_id()}.")
            if self.get_file().read(4) != b'WAVE':
                raise ValueError(f"[ERROR] File parameters: File is not wave type. Got {self.get_file().read(4)}.")
            
            while True:
                # Enable seeking status to read chunks
                self.set_seek_status(seek_status=True)
                
                # Attempt to read a chunk from the file
                try:
                    chunk = Chunk(file=self.get_file(), use_big_endian=False)
                except EOFError:
                    break
                
                # Process the chunk based on its ID
                chunk_id = chunk.get_chunk_id()
                if chunk_id == b'fmt ':
                    # Read format chunk and update format status
                    self.__read_format(chunk=chunk)
                    self.set_format_status(format_status=True)
                elif chunk_id == b'data':
                    # Ensure format chunk is processed before data chunk
                    if not self.get_format_status():
                        raise ValueError("[ERROR] File parameters: Data chunk prior to format chunk.")
                    self.set_chunk(chunk=chunk)
                    sample_count = self.get_chunk().get_chunk_size() // self.get_sample_size()
                    self.set_sample_count(sample_count=sample_count)
                    self.set_seek_status(seek_status=False)
                    break
                
                # Skip to the next chunk
                chunk.skip()
                
            # Verify that both format and data chunks have been processed
            if not self.get_format_status() or not self.get_chunk():
                raise ValueError(f"[ERROR] File parameters: {'Format' if not self.get_format_status() else 'Data'} chunk is missing.")
                
        else:
            # Initialize the file for writing
            self.set_file(file=file)
            
            
    def __touch_file(self, read_mode: bool=True) -> None:
        """
        Manage file access for reading or writing.

        Arguments:
            read_mode (bool, optional): The mode boolean flag ('rb' for read, 'wb' for write).
        """
        
        # Initialize file copy to None
        self.set_file_copy(file=None)
        
        # Check if the filename is a valid string
        if isinstance(self.get_filename(), str):
            # Determine the mode based on the read flag
            mode = "rb" if read_mode else "wb"
            # Open the file in the determined mode
            file = open(self.get_filename(), mode)
            # Set the file copy with the opened file
            self.set_file_copy(file=file)
        
        try:
            # Perform file operations based on parameters
            self.__file_params(file=file, read_mode=read_mode)
        except Exception:
            # Close the file if an exception occurs and a file copy exists
            if self.get_file_copy():
                file.close()
            # Rethrow the exception
            raise
        
    
    def __make_format(self, bits: int, total_samples: int) -> str:
        """
        Create packing format for number-to-byte conversion and vice versa.

        Arguments:
            bits (int): Number of bits per sample (8, 16, 24, or 32)
            total_samples (int): Total number of samples

        Returns:
            str: Packing format string

        Raises:
            ValueError: If bits is not 8, 16, 24, or 32
        """
        
        # make packing format for number-to-byte conversion and vise versa
        if bits == 8: 
            type_ = "B" # 1-byte unsigned char
        elif bits == 16:
            type_ = "h" # 2-byte signed shorts
        elif bits == 24 or bits == 32:
            type_ = "i" # 4-byte signed integer    
        # elif bits == 32:
        #     type_ = "f"  # 4-byte IEEE Float format
        else:
            raise ValueError(f"[ERROR] Make format: Bits value must be 8, 16, 24, or 32. Got {bits}.")

        format_ = "{}{}".format(total_samples, type_)
        
        return format_
    
    
    def __read_format(self, chunk: Chunk) -> None:
        """
        Summary:
            Read and set the format from the given chunk.

        Arguments:
            chunk (Chunk): The input chunk to read the format from.

        Raises:
            EOFError: If there is an error reading the chunk data or sample width.
            ValueError: If an unknown format tag is encountered.
        """
        
        try:
            # Unpack the initial 14 bytes of the chunk into format_tag, channel_count, sample_rate, etc.
            temp = struct.unpack_from("<HHLLH", chunk.read(14))
            format_tag = temp[0]
            
            # Set the channel count and sample rate using unpacked values
            self.set_channel_count(channel_count=temp[1])
            self.set_sample_rate(sample_rate=temp[2])
            
        except struct.error:
            # Raise an error if the initial unpacking fails
            raise EOFError("[ERROR] Read format: Failed to read chunk data.")
        
        if format_tag in [0x0001, 0x0003]:
            try:
                # Unpack the next 2 bytes to get the sample width
                sample_width = struct.unpack_from("<H", chunk.read(2))[0]
            except struct.error:
                # Raise an error if unpacking the sample width fails
                raise EOFError("[ERROR] Read format: Failed to read sample width.")
            
            # Set the sample width, converting bits to bytes
            self.set_sample_width(sample_width=(sample_width + 7) // 8)
            
        else:
            # Raise an error if the format tag is unknown
            raise ValueError(f"[ERROR] Read format: Unsupported format. Supported formats are PCM (1) and IEEE Float (3). Got {format_tag}.")
        
        # Calculate and set the sample size based on channel count and sample width
        self.set_sample_size(sample_size=self.get_channel_count() * self.get_sample_width())
        
        # Set compression type and name
        self.set_compression_type(compression_type="NONE")
        self.set_compression_name(compression_name="not compressed")
        
        
    def __patch_header(self) -> None:
        """
        Summary:
            Updates the header of the file with the current position and length.
        """
        
        # Check if the current data length and position are different and the header status is true
        if self.get_data_length() != self.get_position() and self.get_header_status():
            # Store the current file position to return to it later
            position = self.get_file().tell()
            
            # Move to the position where the former length is stored and update it
            self.get_file().seek(self.get_former_length(), 0)
            self.get_file().write(struct.pack(
                "<L", 36 + self.get_position()
            ))
            
            # Move to the position where the new length is stored and update it
            self.get_file().seek(self.get_new_length(), 0)
            self.get_file().write(struct.pack(
                "<L", self.get_position()
            ))
            
            # Return to the original position in the file
            self.get_file().seek(position, 0)
            
            # Update the internal data length to the current position
            self.set_data_length(data_length=self.get_position())
            
            
    def __write_header(self, length: int) -> None:
        """
        Write the WAV file header if it has not been written yet.
        
        Arguments:
            length: The length of the data.
        """
        
        if not self.get_header_status():
            # Check if necessary attributes are set
            self.get_channel_count()
            self.get_sample_width()
            self.get_sample_rate()
        
            # Write the 'RIFF' header chunk
            self.get_file().write(b'RIFF')
            
            # If sample count is not set, calculate and set it
            if not self.get_sample_count():
                sample_count = length // (self.get_channel_count() * self.get_sample_width())
                self.set_sample_count(sample_count=sample_count)
                
            # Calculate and set the data length
            data_length = self.get_sample_count() * self.get_channel_count() * self.get_sample_width()
            self.set_data_length(data_length=data_length)
            
            try:
                # Record the current position in the file (for future reference)
                self.set_former_length(former_length=self.get_file().tell())
            except (AttributeError, OSError):
                # Handle cases where getting the position fails
                self.set_former_length(former_length=None)
            
            # Write the 'WAVE' format and format chunk
            self.get_file().write(struct.pack(
                "<L4s4sLHHLLHH4s",
                36 + self.get_data_length(),    # Total size of the chunk
                b"WAVE",                        # File type
                b"fmt ",                        # Format chunk marker
                16,                             # Length of the format data
                0x0001,                         # Type of format (1 is PCM, 3 is IEEE Float format)
                self.get_channel_count(),       # Number of channels
                self.get_sample_rate(),         # Sample rate
                self.get_channel_count() * \
                    self.get_sample_rate() * \
                    self.get_sample_width(),    # Byte rate
                self.get_channel_count() * \
                    self.get_sample_width(),    # Block align
                self.get_sample_width() * 8,    # Bits per sample
                b"data"                         # Data chunk header
            ))
            
            # If we recorded the file position earlier, update it with the new position
            if self.get_former_length() is not None:
                self.set_new_length(new_length=self.get_file().tell())
            
            # Write the data length to the file
            self.get_file().write(struct.pack(
                "<L",
                self.get_data_length()
            ))
            
            # Mark the header as written
            self.set_header_status(header_status=True)
    
    
    # setter methods
    
    def set_channel_count(self, channel_count: int, check: bool=True) -> None:
        """
        Summary:
            Sets the channel count for the instance.
        
        Arguments:
            channel_count (int): The number of channels to set.
            check (bool, optional): Flag to indicate if validation is required.
        
        Raises:
            TypeError: If the input type for channel_count or check is not as expected.
            ValueError: If samples have already been written or if the channel count is invalid.
        """
        
        if not isinstance(channel_count, int):
            raise TypeError(
                f"[ERROR] Set channel count: Channel count must be an int. Got {type(channel_count).__name__})."
            )
        
        if not isinstance(check, bool):
            raise TypeError(
                f"[ERROR] Set channel count: check must be a bool. Got {type(check).__name__})."
            )
        
        if self.get_samples_written():
            raise ValueError("[ERROR] Set channel count: Cannot change parameters after starting to write.")
        
        if check and channel_count < 1:
            raise ValueError(f"[ERROR] Set channel count: Channel count must be positive. Got {channel_count}.")
            
        self.__channel_count = channel_count
        
        
    def set_sample_width(self, sample_width: int, check: bool=True) -> None:
        """
        Set the sample width.

        Arguments:
            sample_width (int): The desired sample width.
            check (bool, optional): Flag to check the validity of sample width. Defaults to True.

        Raises:
            TypeError: If sample_width or check is not of the correct type.
            ValueError: If the sample width is invalid or if samples have already been written.
        """
        
        if not isinstance(sample_width, int):
            raise TypeError(f"[ERROR] Set sample width: Sample width must be an integer. Got {type(sample_width).__name__}).")

        if not isinstance(check, bool):
            raise TypeError(f"[ERROR] Set sample width: Check must be a boolean. Got {type(check).__name__}).")
        
        if self.get_samples_written():
            raise ValueError("[ERROR] Set sample width: Cannot change parameters after starting to write.")
        
        if check and (sample_width < 1 or sample_width > 4):
            raise ValueError(f"[ERROR] Set sample width: Sample width must be between 1 and 4. Got {sample_width}")
            
        self.__sample_width = sample_width
        
        
    def set_sample_rate(self, sample_rate: int, check: bool=True) -> None:
        """
        Set the sample rate for the audio processing.

        Arguments:
            sample_rate (int, optional): The new sample rate to be set.
            check (bool, optional): Flag to enable or disable checking the validity of the sample rate.

        Raises:
            TypeError: If sample_rate is not an int.
            ValueError: If the sample rate is invalid or if samples have already been written.
        """
        
        if not isinstance(sample_rate, int):
            raise TypeError(f"[ERROR] Set sample rate: Sample rate must be an int. Got {type(sample_rate).__name__}.")

        if self.get_samples_written():
            raise ValueError("[ERROR] Set sample rate: Cannot change parameters after starting to write.")
        
        if check and sample_rate < 1:
            raise ValueError(f"[ERROR] Set sample rate: Sample rate must be positive. Got {sample_rate}.")
            
        self.__sample_rate = sample_rate
        
        
    def set_sample_count(self, sample_count: int, check: bool=True) -> None:
        """
        Set the sample count for the audio processing.

        Arguments:
            sample_count (int): The new sample count to be set.
            check (bool, optional): Flag to enable or disable checking the validity of the sample count.

        Raises:
            TypeError: If sample_count is not an integer or check is not a boolean.
            ValueError: If the sample count is invalid or if samples have already been written.
        """

        if not isinstance(sample_count, int):
            raise TypeError(f"[ERROR] Set sample count: Sample count must be an integer. Got {type(sample_count).__name__}).")
        if not isinstance(check, bool):
            raise TypeError(f"[ERROR] Set sample count: Check must be a boolean. Got {type(check).__name__}).")
        
        if self.get_samples_written():
            raise ValueError("[ERROR] Set sample count: Cannot change parameters after starting to write.")
        
        if check and sample_count < 1:
            raise ValueError(f"[ERROR] Set sample count: Sample count must be positive. Got {sample_count}")
            
        self.__sample_count = sample_count
        
        
    def set_compression_type(self, compression_type: str) -> None:
        """
        Sets the compression type for the object.

        Arguments:
            compression_type (str): The type of compression to set. Only 'NONE' is valid.

        Raises:
            TypeError: If compression_type is not a string.
            ValueError: If samples have already been written or if the compression type is invalid.
        """
        
        if not isinstance(compression_type, str):
            raise TypeError(f"[ERROR] Set compression type: Compression type must be a string. Got {type(compression_type).__name__}.")
    
        if self.get_samples_written():
            raise ValueError("[ERROR] Set compression type: Cannot change parameters after starting to write.")
        
        if compression_type not in ["NONE"]:
            raise ValueError(f"[ERROR] Set compression type: Compression type is unsupported. Only 'NONE' is supported. Got {compression_type}.")
        
        self.__compression_type = compression_type
        
        
    def set_compression_name(self, compression_name: str) -> None:
        """
        Set the compression name.

        Arguments:
            compression_name (str): The name of the compression algorithm.

        Raises:
            TypeError: If compression_name is not a string.
            ValueError: If attempting to set the compression name after samples have been written.
        """
        
        if not isinstance(compression_name, str):
            raise TypeError(f"[ERROR] Set compression name: Compression name must be a string. Got {type(compression_name).__name__}).")

        if self.get_samples_written():
            raise ValueError("[ERROR] Set compression name: Cannot change parameters after starting to write.")
        
        self.__compression_name = compression_name
        
        
    def set_filename(self, filename: str) -> None:
        """
        Summary:
            Sets the filename attribute.

        Arguments:
            filename (str): The name to set as the filename.

        Raises:
            TypeError: If the filename is not a string.
            ValueError: If the filename is empty.
        """
        
        if not isinstance(filename, str):
            raise TypeError(f"[ERROR] Set filename: Filename must be a string. Got {type(filename).__name__}.")
        
        if not filename:
            raise ValueError("[ERROR] Set filename: Filename cannot be empty.")
    
        self.__filename = filename
        
        
    def set_file(self, file: Optional[Union[Chunk, io.BufferedReader, io.BufferedWriter]]) -> None:
        """
        Summary:
            Sets the file attribute with a file-like or Chunk object

        Arguments:
            file (Chunk or File-like): The file-like object to save.
            
        Raises:
            TypeError: If the file is not an instance of Chunk, io.BufferedReader, or io.BufferedWriter.
        """
        
        if file is not None and not isinstance(file, (Chunk, io.BufferedReader, io.BufferedWriter)):
            raise TypeError(f"[ERROR] Set file: File must be file-like. Got {type(file).__name__}).")
        
        self.__file = file
        
        
    def set_file_copy(self, file: Optional[Union[Chunk, io.BufferedReader, io.BufferedWriter]]) -> None:
        """
        Set the file copy.

        Arguments:
            file (Union[io.BufferedReader, io.BufferedWriter]): The file to be set.

        Raises:
            TypeError: If the file is not an instance of io.BufferedReader or io.BufferedWriter.
        """
        
        if file is not None and not isinstance(file, (Chunk, io.BufferedReader, io.BufferedWriter)):
            raise TypeError(f"[ERROR] Set file copy: File copy must be file-like. Got {type(file).__name__}).")
            
        self.__file_copy = file
        
        
    def set_sample_size(self, sample_size: int) -> None:
        """
        Set the sample size for the instance.

        Arguments:
            sample_size (int): The size of the sample to be set.

        Raises:
            TypeError: If sample_size is not an integer.
            ValueError: If sample_size is not positive.
        """
        
        if not isinstance(sample_size, int):
            raise TypeError(f"[ERROR] Set sample size: Sample size must be an integer. Got {type(sample_size).__name__}).")
        
        if sample_size <= 0:
            raise ValueError(f"[ERROR] Set sample size: Sample size must be positive. Got {sample_size}).")
    
        self.__sample_size = sample_size
        
        
    def set_position(self, position: int, check: bool=True) -> None:
        """
        Sets the sound position to the specified value.
        
        Arguments:
            position (int): The new position to set.
            check (bool, optional): Whether to check the validity of the position. Default is True.
        
        Raises:
            TypeError: If `position` is not an integer or `check` is not a boolean.
            ValueError: If `position` is out of the valid range (0 to sample count).
        """
        
        if not isinstance(position, int):
            raise TypeError(f"[ERROR] Set sound position: Position must be an integer. Got {type(position).__name__}.")
        if not isinstance(check, bool):
            raise TypeError(f"[ERROR] Set sound position: Check must be a boolean. Got {type(check).__name__}.")
        
        if check and (position < 0 or position > self.get_sample_count()):
            raise ValueError(f"[ERROR] Set sound position: Position out of bounds [0, {self.get_sample_count()}]. Got {position}.")
        
        self.__sound_position = position
        self.set_seek_status(seek_status=True)
        
        
    def set_format_status(self, format_status: bool) -> None:
        """
        Set the format status.

        Arguments:
            format_status (bool): The format status to be set.

        Raises:
            TypeError: If format_status is not of type bool.
        """
        
        if not isinstance(format_status, bool):
            raise TypeError(f"[ERROR] Set format status: Format status must be a bool. Got {type(format_status).__name__}).")

        self.__format_status = format_status
        
        
    def set_chunk(self, chunk: Optional[Chunk]) -> None:
        """
        Sets the chunk attribute.

        Arguments:
            chunk (Chunk): The chunk to set.

        Raises:
            TypeError: If the input is not of type Chunk.
        """
        
        if chunk is not None and not isinstance(chunk, Chunk):
            raise TypeError(f"[ERROR] Set chunk: Chunk must be a Chunk. Got {type(chunk).__name__}.")

        self.__chunk = chunk
        
        
    def set_seek_status(self, seek_status: bool) -> None:
        """
        Summary:
            Sets the seek status for the object.

        Arguments:
            seek_status (bool): The new seek status to set.

        Raises:
            TypeError: If the provided seek_status is not a boolean.
        """
        
        if not isinstance(seek_status, bool):
            raise TypeError(f"[ERROR] Set seek status: Seek status must be a boolean. Got {type(seek_status).__name__}).")
    
        self.__seek_status = seek_status
        
        
    def set_header_status(self, header_status: bool) -> None:
        """
        Set the status of the header.

        Arguments:
            header_status (bool): The new status for the header.

        Raises:
            TypeError: If header_status is not a boolean.
        """
        
        if not isinstance(header_status, bool):
            raise TypeError(f"[ERROR] Set header status: Header status must be a bollean. Got {type(header_status).__name__}).")
        
        self.__header_status = header_status
        
        
    def set_data_length(self, data_length: int) -> None:
        """
        Set the length of the data.

        Arguments:
            data_length (int): The length of the data to set. Must be a non-negative integer.

        Raises:
            TypeError: If data_length is not an integer.
            ValueError: If data_length is a negative integer.
        """
        
        if not isinstance(data_length, int):
            raise TypeError(f"[ERROR] Set data length: Data length must be an int. Got {type(data_length).__name__}.")
        
        if data_length < 0:
            raise ValueError(f"[ERROR] Set data length: Data length cannot be negative. Got {data_length}).")

        self.__data_length = data_length
        
        
    def set_samples_written(self, samples_written: int) -> None:
        """
        Set the number of samples written.

        Arguments:
            samples_written (int): The number of samples to set.

        Raises:
            TypeError: If samples_written is not an integer.
            ValueError: If samples_written is negative.
        """
        
        if not isinstance(samples_written, int):
            raise TypeError(f"[ERROR] Set samples written: Samples written must be an int. Got {type(samples_written).__name__}.")

        if samples_written < 0:
            raise ValueError(f"[ERROR] Set samples written: Samples written cannot be negative. Got {samples_written}).")

        self.__samples_written = samples_written
        
        
    def set_mode(self, mode: str) -> None:
        """
        Sets the read/write mode for the object.

        Arguments:
            mode (str): The read/write mode. Valid options are 'read', 'r', 'rb', 'write', 'w', and 'wb'.

        Raises:
            TypeError: If mode is not a string.
            ValueError: If mode is not a valid value.
        """
        
        if not isinstance(mode, str):
            raise TypeError(f"[ERROR] Set mode: Read/write mode must be a string. Got {type(mode).__name__}.")
        
        mode = mode.lower()
        if mode in ["read", "r", "rb"]:
            mode = "rb"
        elif mode in ["write", "w", "wb"]:
            mode = "wb"
        else:
            raise ValueError(f"[ERROR] Set mode: Read/Write mode must be 'read' or 'write'. Got {mode}.")
        
        self.__mode = mode
        
        
    def set_former_length(self, former_length: int) -> None:
        """
        Sets the former length value.

        Arguments:
            former_length (int): The length to set.

        Raises:
            TypeError: If former_length is not an integer.
            ValueError: If former_length is not a positive value.
        """
        
        if not isinstance(former_length, int):
            raise TypeError(f"[ERROR] Set former length: Former length must be an integer. Got {type(former_length).__name__}.")
        
        if former_length <= 0:
            raise ValueError(f"[ERROR] Set former length: Former length must be a positive integer. Got {former_length}.")
    
        self.__former_length = former_length
        
        
    def set_new_length(self, new_length: int) -> None:
        """
        Sets the new length for the object.

        Arguments:
            new_length (int): The new length value to be set.

        Raises:
            TypeError: If `new_length` is not a int.
            ValueError: If `new_length` is not a positive value.
        """
        
        if not isinstance(new_length, int):
            raise TypeError(f"[ERROR] Set new length: New length must be int. Got {type(new_length).__name__}.")
        
        if new_length <= 0:
            raise ValueError(f"[ERROR] Set new length: New length must be a positive value. Got {new_length}.")
    
        self.__new_length = new_length
        
        
    # getter methods
    
    def get_channel_count(self) -> int:
        """
        Gets the number of channels.

        Returns:
            int: The number of channels.

        Raises:
            ValueError: If the number of channels is not set.
        """
        
        if self.__channel_count is None:
            raise ValueError("[ERROR] Get channel count: Channel count not set.")
        
        return self.__channel_count 
    
    
    def get_sample_width(self) -> int:
        """
        Gets the sample width.

        Returns:
            int: The sample width.

        Raises:
            ValueError: If the sample width is not set.
        """
        
        if self.__sample_width is None:
            raise ValueError("[ERROR] Get sample width: Sample width not set.")
        
        return self.__sample_width
    
    
    def get_sample_rate(self) -> Union[int, float]:
        """
        Gets the sample rate.

        Returns:
            int: The sample rate.

        Raises:
            ValueError: If the sample rate is not set.
        """
        
        if self.__sample_rate is None:
            raise ValueError("[ERROR] Get sample rate: Sample rate not set.")
        
        return self.__sample_rate
    
    
    def get_sample_count(self) -> int:
        """
        Gets the sample count.

        Returns:
            int: The sample count.

        Raises:
            ValueError: If the sample count is not set.
        """
        
        if self.__sample_count is None:
            raise ValueError("[ERROR] Get sample count: Sample count not set.")
        
        return self.__sample_count


    def get_compression_type(self) -> str:
        """
        Gets the compression type.

        Returns:
            str: The compression type.

        Raises:
            ValueError: If the compression type is not set.
        """
        
        if not self.__compression_type:
            raise ValueError("[ERROR] Get compression type: Compression type not set.")
        
        return self.__compression_type


    def get_compression_name(self) -> str:
        """
        Gets the compression name.

        Returns:
            str: The compression name.

        Raises:
            ValueError: If the compression name is not set.
        """
        
        if not self.__compression_name:
            raise ValueError("[ERROR] Get compression name: Compression name not set.")
        
        return self.__compression_name


    def get_filename(self) -> str:
        """
        Gets the filename.

        Returns:
            str: The filename.

        Raises:
            ValueError: If the filename is not set.
        """
        
        if not self.__filename:
            raise ValueError("[ERROR] Get filename: Filename not set.")
        
        return self.__filename


    def get_file(self) -> object:
        """
        Gets the file object.

        Returns:
            object: The file object.

        Raises:
            ValueError: If the file is not set.
        """
        
        if not self.__file:
            raise ValueError("[ERROR] Get file: File not set.")
        
        return self.__file
    
    
    def get_file_copy(self) -> object:
        """
        Gets a copy of the file object.

        Returns:
            object: The file copy.

        Raises:
            ValueError: If the file copy is not set.
        """
        
        if not self.__file_copy:
            raise ValueError("[ERROR] Get file copy: File copy not set.")
        
        return self.__file_copy


    def get_sample_size(self) -> int:
        """
        Gets the sample size.

        Returns:
            int: The sample size.

        Raises:
            ValueError: If the sample size is not set.
        """
        
        if self.__sample_size is None:
            raise ValueError("[ERROR] Get sample size: Sample size not set.")
        
        return self.__sample_size


    def get_position(self) -> int:
        """
        Gets the sound position.

        Returns:
            int: The sound position.

        Raises:
            ValueError: If the position is not set.
        """
        
        if self.__sound_position is None:
            raise ValueError("[ERROR] Get position: Sound position not set.")
        
        return self.__sound_position


    def get_format_status(self) -> bool:
        """
        Gets the format status.

        Returns:
            bool: The format status.

        Raises:
            ValueError: If the format status is not set.
        """
        
        if self.__format_status is None:
            raise ValueError("[ERROR] Get format status: Format status not set.")
        
        return self.__format_status


    def get_chunk(self) -> object:
        """
        Gets the chunk object.

        Returns:
            object: The chunk object.

        Raises:
            ValueError: If the chunk is not set.
        """
        
        if not self.__chunk:
            raise ValueError("[ERROR] Get chunk: Chunk not set.")
        
        return self.__chunk
    
    
    def get_seek_status(self) -> bool:
        """
        Gets the seek status.

        Returns:
            bool: The seek status.

        Raises:
            ValueError: If the seek status is not set.
        """
        
        if self.__seek_status is None:
            raise ValueError("[ERROR] Get seek status: Seek status not set.")
        
        return self.__seek_status


    def get_header_status(self) -> bool:
        """
        Gets the header status.

        Returns:
            bool: The header status.

        Raises:
            ValueError: If the header status is not set.
        """
        
        if self.__header_status is None:
            raise ValueError("[ERROR] Get header status: Header status not set.")
        
        return self.__header_status


    def get_data_length(self) -> int:
        """
        Gets the data length.

        Returns:
            int: The data length.

        Raises:
            ValueError: If the data length is not set.
        """
        
        if self.__data_length is None:
            raise ValueError("[ERROR] Get data length: Data length not set.")
        
        return self.__data_length


    def get_samples_written(self) -> int:
        """
        Gets the number of samples written.

        Returns:
            int: The number of samples written.

        Raises:
            ValueError: If the samples written is not set.
        """
        
        if self.__samples_written is None:
            raise ValueError("[ERROR] Get samples written: Samples written not set.")
        
        return self.__samples_written


    def get_mode(self) -> str:
        """
        Gets the mode.

        Returns:
            str: The mode.

        Raises:
            ValueError: If the mode is not set.
        """
        
        if not self.__mode:
            raise ValueError("[ERROR] Get mode: Read/write mode not set.")
        
        return self.__mode
        
        
    def get_former_length(self) -> int:
        """
        Gets the former length.

        Returns:
            int: The former length.

        Raises:
            ValueError: If the former length is not set.
        """
        
        if self.__former_length is None:
            raise ValueError("[ERROR] Get former length: Former length not set.")
        
        return self.__former_length


    def get_new_length(self) -> int:
        """
        Gets the new length.

        Returns:
            int: The new length.

        Raises:
            ValueError: If the new length is not set.
        """
        
        if self.__new_length is None:
            raise ValueError("[ERROR] Get new length: New length not set.")
        
        return self.__new_length
    

    # representation methods
    
    def get_parameters(self) -> dict:
        """
        Return the parameters of the Wave object.

        Returns:
            dict: Dictionary of parameters and their values.
            
        Raises:
            ValueError: If not all parameters are set.
        """
        
        parameters = {
            "channel count": self.get_channel_count(),
            "sample width": self.get_sample_width(),
            "sample rate": self.get_sample_rate(),
            "sample count": self.get_sample_count(),
            "compression type": self.get_compression_type(),
            "compression name": self.get_compression_name()
        }
        
        if not all(parameters):
            raise ValueError("[ERROR] Get parameters: Not all parameters are set.")
        
        return parameters
        
        
    def __repr__(self) -> str:
        """
        Return a string representation of the Wave object for debugging.

        Returns:
            str: String representation of the object.
        """
        
        developer_repr = Utils.base_repr(self)
        
        return developer_repr


    def __str__(self) -> str:
        """
        Return a string representation of the Wave object for the end user.

        Returns:
            str: User-friendly string representation of the object.
        """
        
        user_repr = Utils.base_str(self)
        
        return user_repr