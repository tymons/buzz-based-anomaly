import math

from typing import Union, Tuple, List


def convolutional_to_mlp(input_size: Union[int, Tuple], depth: int, kernel: int, padding: int, max_pool: int) \
        -> Tuple[int, List[Union[Tuple, int]]]:
    """
    Function for calculating end of convolutional output size
    :param input_size: input size
    :param depth: depth of layers
    :param kernel: kernel size
    :param padding: padding
    :param max_pool: maxpool
    :return: output size, temporals
    """
    if not isinstance(input_size, tuple):
        input_size = (input_size, )

    output_dims = []
    output_dims_temporal = []
    for dimension in input_size:
        temporal_values = [dimension]
        for _ in range(depth):
            dimension = int(((dimension - kernel + 2 * padding) + 1))
            dimension = dimension // max_pool
            temporal_values.append(dimension)
        output_dims.append(dimension)
        output_dims_temporal.append(temporal_values)

    connector_size = math.prod(output_dims)
    return connector_size, output_dims_temporal[0] if len(output_dims) == 1 else list(zip(*output_dims_temporal))
