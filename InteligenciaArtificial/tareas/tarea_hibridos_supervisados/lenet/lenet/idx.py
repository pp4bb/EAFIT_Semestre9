"""Module for parsing IDX files.
"""

import numpy as np


def parse_dtype(data_type: int) -> np.dtype:
    """Maps data type from IDX file to numpy dtype.

    Parameters
    ----------
    data_type : int
        Data type from IDX file.
    
    Returns
    -------
    numpy.dtype
        Numpy dtype.
    """

    match data_type:
        case 0x08:
            return np.dtype(np.uint8)
        case 0x09:
            return np.dtype(np.int8)
        case 0x0B:
            return np.dtype(np.int16)
        case 0x0C:
            return np.dtype(np.int32)
        case 0x0D:
            return np.dtype(np.float32)
        case 0x0E:
            return np.dtype(np.float64)
        case _:
            raise ValueError(f'Unknown data type {data_type}')

def parse_idx(filename: str) -> np.ndarray:
    """Parse IDX file into numpy array.
    
    Parameters
    ----------
    filename : str
        Path to IDX file.
    
    Returns
    -------
    numpy.ndarray
        Numpy array with data from IDX file."""
    with open(filename, 'rb') as f:
        zeros = int.from_bytes(f.read(2), 'big')

        assert zeros == 0, 'Zeros should be 0'

        data_type = int.from_bytes(f.read(1), 'big')
        data_type = parse_dtype(data_type)
        # Set the byte order to big endian
        data_type = data_type.newbyteorder('>')

        dimensions = int.from_bytes(f.read(1), 'big')
        shape = [
            int.from_bytes(f.read(4), 'big')
            for _ in range(dimensions)
        ]

        data = np.frombuffer(f.read(), dtype=data_type)

        # Reshape the data
        data = data.reshape(shape)

        return data


if __name__ == '__main__':
    data = parse_idx('t10k-images-idx3-ubyte')
    import matplotlib.pyplot as plt
    plt.imshow(data[0], cmap='gray')
    plt.show()