from typing import List, Tuple

import torch
from torch import Tensor
import torch.nn as nn
from torch.nn.parameter import Parameter
import torch.optim
import numpy as np
import pandas as pd

from config_types import Config
from models.Transformers import FTTransformer


def init_ft_transformer(
    args: Config,
    adata_filtered_var: pd.DataFrame,
    use_cls: bool = False,
    use_predictor: bool = False,
) -> Tuple[FTTransformer, np.ndarray, np.ndarray, np.ndarray]:
    num_indices = adata_filtered_var["n_categorical_options"] == 1
    bin_indices = adata_filtered_var["n_categorical_options"] == 2
    cat_indices = adata_filtered_var["n_categorical_options"] > 2

    n_num_features = num_indices.sum()
    n_bin_features = bin_indices.sum()
    cat_cardinalities = adata_filtered_var[cat_indices][
        "n_categorical_options"
    ].tolist()

    model = FTTransformer(
        args=args,
        n_num_features=n_num_features,
        n_bin_features=n_bin_features,
        cat_cardinalities=cat_cardinalities,
        use_cls=use_cls,
        use_predictor=use_predictor,
    )

    return (
        model,
        num_indices.to_numpy(),
        bin_indices.to_numpy(),
        cat_indices.to_numpy(),
    )


class NumericalFeaturePredictions(nn.Module):
    """Prediction (reconstruction) of numerical features

    **Shape**

    - Input: `(*, d_embedding)`
    - Output: `(*, n_features, 1)`

    **Examples**

    >>> batch_size = 2
    >>> d_embedding = 4
    >>> x = torch.randn(batch_size, d_embedding)
    >>> n_features = 3
    >>> m = FeaturePredictions(n_features, d_embedding)
    >>> m(x).shape
    torch.Size([2, 3, 1])
    """

    def __init__(self, n_features: int, d_embedding: int) -> None:
        """
        Args:
            n_features: the number of features.
            d_embedding: the embedding size.
            bias: if `True`, for each feature, a trainable vector is added to the
                prediction. For each feature, a separate non-shared bias vector is allocated.
        """
        super().__init__()
        if d_embedding <= 0:
            raise ValueError(f"d_embedding must be positive, however: {d_embedding=}")

        self.weight = Parameter(torch.empty(n_features, d_embedding, 1))
        self.bias = Parameter(torch.empty(n_features, 1))
        self.reset_parameters()

    def reset_parameters(self) -> None:
        d_rsqrt = self.weight.shape[1] ** -0.5
        nn.init.uniform_(self.weight, -d_rsqrt, d_rsqrt)
        nn.init.uniform_(self.bias, -d_rsqrt, d_rsqrt)

    def forward(self, x: Tensor) -> Tensor:
        """Do the forward pass."""
        if x.ndim < 2:
            raise ValueError(
                f"The input must have at least two dimensions, however: {x.ndim=}"
            )
        # x is of shape (batch, n_features, d_embedding)
        # Perform row-wise multiplication and sum across d_embedding
        result = torch.einsum("bfd,fdo -> bfo", x, self.weight) + self.bias
        # The output is of shape (batch, n_features, 1) after adding biases
        return result


class CategoricalFeaturePredictions(nn.Module):
    """Prediction (reconstruction) of categorical features

    **Shape**

    - Input: `(*, d_embedding)`
    - Output: `(*, n_features, n_options)`
    """

    def __init__(self, cardinalities: List[int], d_embedding: int) -> None:
        """
        Args:
            cardinalities: the number of options for each feature.
            d_embedding: the embedding size.
        """
        super().__init__()
        if not cardinalities:
            raise ValueError("cardinalities must not be empty")
        if any(x <= 0 for x in cardinalities):
            i, value = next((i, x) for i, x in enumerate(cardinalities) if x <= 0)
            raise ValueError(
                "cardinalities must contain only positive values,"
                f" however: cardinalities[{i}]={value}"
            )
        if d_embedding <= 0:
            raise ValueError(f"d_embedding must be positive, however: {d_embedding=}")

        # self.weights = nn.ParameterList([Parameter(torch.empty(d_embedding, x)) for x in cardinalities])
        # self.biases = nn.ParameterList([Parameter(torch.empty(x)) for x in cardinalities])

        self.n_features = len(cardinalities)
        self.max_cardinality = max(cardinalities)
        self.cardinalities = cardinalities

        # Create a single weight matrix and bias vector
        self.weight = nn.Parameter(
            torch.empty(self.n_features, d_embedding, self.max_cardinality)
        )
        self.bias = nn.Parameter(torch.empty(self.n_features, self.max_cardinality))

        self.reset_parameters()

    # def reset_parameters(self) -> None:
    #     d_rsqrt = self.weights[0].size(0) ** -0.5
    #     for w in self.weights:
    #         nn.init.uniform_(w, -d_rsqrt, d_rsqrt)
    #     for b in self.biases:
    #         nn.init.uniform_(b, -d_rsqrt, d_rsqrt)

    def reset_parameters(self) -> None:
        d_rsqrt = self.weight.size(1) ** -0.5
        nn.init.uniform_(self.weight, -d_rsqrt, d_rsqrt)
        nn.init.uniform_(self.bias, -d_rsqrt, d_rsqrt)

    def forward(self, x: Tensor) -> Tensor:
        """Do the forward pass.
        Assume: x is of shape (batch, n_features, d_embedding)
        """
        # out = [
        #     torch.matmul(x[:, i, :], weight) + bias for i, (weight, bias) in enumerate(zip(self.weights, self.biases))
        # ]
        # return out

        # Perform matrix multiplication for all features at once
        out = torch.matmul(x.unsqueeze(2), self.weight.unsqueeze(0)).squeeze(2)

        # Add bias
        out += self.bias.unsqueeze(0)

        # Create a mask to zero out predictions beyond each feature's cardinality
        mask = torch.arange(self.max_cardinality, device=x.device).expand(
            x.size(0), self.n_features, self.max_cardinality
        )
        mask = mask < torch.tensor(self.cardinalities, device=x.device).unsqueeze(
            0
        ).unsqueeze(2)

        # Apply the mask to zero out predictions beyond each feature's cardinality
        out = out.masked_fill(~mask, float("-inf"))

        # Softmax
        # out = torch.softmax(out, dim=2)

        return out
