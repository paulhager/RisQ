"""Code Base is from: Revisiting Deep Learning Models for Tabular Data.
https://github.com/yandex-research/rtdl-revisiting-models/tree/main

Edits (DIFF) made by Paul Hager

1. CLS predictor seperated from backbone
2. Attention masking added to transformer
3. Added support for transtab transformer backbone

"""

__version__ = "0.0.3"

__all__ = [
    "FTTransformer",
]

import math
import typing
from collections import OrderedDict
from typing import Any, Dict, List, Literal, Optional, Tuple, cast, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim
from torch import Tensor
from torch.nn.parameter import Parameter
import numpy as np
import os


from rtdl_revisiting_models import CategoricalEmbeddings

from config_types import Config, Activation
from dataset.utils import filter_and_pad_vectorized

_INTERNAL_ERROR = "Internal error"


def _named_sequential(*modules) -> nn.Sequential:
    return nn.Sequential(OrderedDict(modules))


class CrossAttnBlock(nn.Module):
    """
    One layer of:  LN -> Multihead Cross-Attn -> Resid -> LN -> MLP -> Resid
    Queries = prediction tokens; Keys/Values = bottleneck tokens.
    """

    def __init__(
        self,
        d_model: int,
        num_heads: int,
        dim_feedforward: int = 4_096,
        attn_dropout: float = 0.0,
        resid_dropout: float = 0.0,
        activation: str = "gelu",
        batch_first: bool = True,
    ):
        super().__init__()
        self.norm_q = nn.LayerNorm(d_model)
        self.norm_m = nn.LayerNorm(d_model)  # optional: normalize memory too

        self.attn = nn.MultiheadAttention(
            embed_dim=d_model,
            num_heads=num_heads,
            dropout=attn_dropout,
            batch_first=batch_first,
            bias=True,
        )
        self.drop_attn = nn.Dropout(resid_dropout)

        self.norm_ff = nn.LayerNorm(d_model)
        act = nn.GELU() if activation.lower() == "gelu" else nn.ReLU()
        self.ff = nn.Sequential(
            nn.Linear(d_model, dim_feedforward),
            act,
            nn.Dropout(resid_dropout),
            nn.Linear(dim_feedforward, d_model),
        )
        self.drop_ff = nn.Dropout(resid_dropout)

    def forward(
        self,
        bottleneck: torch.Tensor,  # (B, X, D)
        prediction_tokens: torch.Tensor,  # (B, Y, D)
        key_padding_mask: torch.Tensor | None = None,  # (B, X), True=PAD
    ) -> torch.Tensor:  # returns (B, Y, D)
        # Pre-norm
        q = self.norm_q(prediction_tokens)
        kvm = self.norm_m(bottleneck)

        # Cross-attention: queries = q, keys/values = kvm
        attn_out, _ = self.attn(
            query=q,
            key=kvm,
            value=kvm,
            key_padding_mask=key_padding_mask,  # ignores padded memory positions
            need_weights=False,
        )
        x = prediction_tokens + self.drop_attn(attn_out)

        # FFN
        y = self.norm_ff(x)
        y = self.ff(y)
        x = x + self.drop_ff(y)
        return x


class CrossAttnDecoder(nn.Module):
    """
    Stacked cross-attention-only decoder.
    No self-attn among Y queries -> predictions are independent given the memory.
    """

    def __init__(
        self,
        d_model: int,
        num_heads: int,
        num_layers: int,
        dim_feedforward: int = 4_096,
        attn_dropout: float = 0.0,
        resid_dropout: float = 0.0,
        activation: Activation = Activation.GELU,
        batch_first: bool = True,
    ):
        super().__init__()
        activation_str = "gelu" if activation == Activation.GELU else "relu"
        self.layers = nn.ModuleList(
            [
                CrossAttnBlock(
                    d_model=d_model,
                    num_heads=num_heads,
                    dim_feedforward=dim_feedforward,
                    attn_dropout=attn_dropout,
                    resid_dropout=resid_dropout,
                    activation=activation_str,
                    batch_first=batch_first,
                )
                for _ in range(num_layers)
            ]
        )
        self.final_norm = nn.LayerNorm(d_model)

    def forward(
        self,
        bottleneck: torch.Tensor,  # (B, X, D)
        prediction_tokens: torch.Tensor,  # (B, Y, D)
        key_padding_mask: torch.Tensor | None = None,  # (B, X)
    ) -> torch.Tensor:  # (B, Y, D)
        x = prediction_tokens
        for layer in self.layers:
            x = layer(
                bottleneck=bottleneck,
                prediction_tokens=x,
                key_padding_mask=key_padding_mask,
            )
        return self.final_norm(x)


class MultilayerGatedTransformer(nn.Module):
    def __init__(
        self,
        hidden_dim,
        n_layers,
        num_heads,
        attention_dropout,
        ffn_hidden_dim,
        ffn_dropout,
        norm_first,
        use_gate,
    ):
        super().__init__()
        self.transformer_encoder = nn.ModuleList(
            [
                GatedTransformerLayer(
                    hidden_dim=hidden_dim,
                    num_heads=num_heads,
                    attention_dropout=attention_dropout,
                    ffn_hidden_dim=ffn_hidden_dim,
                    ffn_dropout=ffn_dropout,
                    layer_norm_eps=1e-5,
                    norm_first=norm_first,
                    use_gate=use_gate,
                )
            ]
        )
        if n_layers > 1:
            encoder_layer = GatedTransformerLayer(
                hidden_dim=hidden_dim,
                num_heads=num_heads,
                attention_dropout=attention_dropout,
                ffn_hidden_dim=ffn_hidden_dim,
                ffn_dropout=ffn_dropout,
                layer_norm_eps=1e-5,
                norm_first=norm_first,
                use_gate=use_gate,
            )
            stacked_transformer = nn.TransformerEncoder(
                encoder_layer, num_layers=n_layers - 1
            )
            self.transformer_encoder.append(stacked_transformer)

    def forward(self, embedding, src_key_padding_mask=None, **kwargs) -> Tensor:
        """args:
        embedding: bs, num_token, hidden_dim
        """
        outputs = embedding
        for i, mod in enumerate(self.transformer_encoder):
            outputs = mod(outputs, src_key_padding_mask=src_key_padding_mask)
        return outputs


class GatedTransformerLayer(nn.Module):
    def __init__(
        self,
        hidden_dim,
        num_heads,
        attention_dropout,
        ffn_hidden_dim,
        ffn_dropout,
        layer_norm_eps,
        norm_first,
        use_gate,
        skip_first_norm=False,
    ) -> None:
        super().__init__()
        self.self_attn = nn.MultiheadAttention(
            hidden_dim, num_heads, dropout=attention_dropout, batch_first=True
        )
        self.attention_dropout = nn.Dropout(attention_dropout)
        # FT-Transformer finds this to be essential
        if skip_first_norm:
            self.norm1 = nn.Identity()
        else:
            self.norm1 = nn.LayerNorm(hidden_dim, eps=layer_norm_eps)

        # Implementation of Feedforward model
        self.linear1 = nn.Linear(hidden_dim, ffn_hidden_dim)
        self.linear2 = nn.Linear(ffn_hidden_dim, hidden_dim)
        self.dropout_ff_1 = nn.Dropout(ffn_dropout)
        self.dropout_ff_2 = nn.Dropout(ffn_dropout)
        self.norm2 = nn.LayerNorm(hidden_dim, eps=layer_norm_eps)

        # Implementation of gates
        self.gate_linear = nn.Linear(hidden_dim, 1, bias=False)
        self.gate_act = nn.Sigmoid()

        self.norm_first = norm_first
        self.use_gate = use_gate

        self.activation = F.relu

    # self-attention block
    def _sa_block(self, x: Tensor, key_padding_mask: Optional[Tensor]) -> Tensor:
        x = self.self_attn(
            x, x, x, key_padding_mask=key_padding_mask, need_weights=False
        )[0]
        return self.attention_dropout(x)

    # feed forward block
    def _ff_block(self, x: Tensor) -> Tensor:
        if self.use_gate:
            g = self.gate_act(self.gate_linear(x))
            h = self.linear1(x)
            h = h * g
        else:
            h = self.linear1(x)
        h = self.linear2(self.dropout_ff_1(self.activation(h)))
        return self.dropout_ff_2(h)

    def forward(self, src, src_key_padding_mask=None, **kwargs) -> Tensor:
        r"""Pass the input through the encoder layer.

        Args:
            src: the sequence to the encoder layer (required).
            src_mask: the mask for the src sequence (optional).
            src_key_padding_mask: the mask for the src keys per batch (optional).

        Shape:
            see the docs in Transformer class.
        """
        # see Fig. 1 of https://arxiv.org/pdf/2002.04745v1.pdf
        x = src
        if self.norm_first:
            x = x + self._sa_block(self.norm1(x), src_key_padding_mask)
            x = x + self._ff_block(self.norm2(x))
        else:
            x = self.norm1(x + self._sa_block(x, src_key_padding_mask))
            x = self.norm2(x + self._ff_block(x))
        return x


class _CLSEmbedding(nn.Module):
    def __init__(self, d_embedding: int) -> None:
        super().__init__()
        self.weight = Parameter(torch.empty(d_embedding))
        self.reset_parameters()

    def reset_parameters(self) -> None:
        d_rsqrt = self.weight.shape[-1] ** -0.5
        nn.init.uniform_(self.weight, -d_rsqrt, d_rsqrt)

    def forward(self, batch_dims: Tuple[int]) -> Tensor:
        if not batch_dims:
            raise ValueError("The input must be non-empty")

        return self.weight.expand(*batch_dims, 1, -1)


class FTTransformer(nn.Module):
    """The FT-Transformer model from Section 3.3 in the paper."""

    def __init__(
        self,
        *,
        args: Config,
        n_num_features: int = 0,
        n_bin_features: int = 0,
        cat_cardinalities: List[int] = [],
        use_cls: bool = False,
        use_predictor: bool = False,
    ) -> None:
        """
        Args:
            n_num_features: the number of numeric (continuous) features.
            n_bin_features: the number of binary features.
            cat_cardinalities: the cardinalities of categorical features.
                Pass en empty list if there are no categorical features.
            use_csl: if `True`, then the model will use a CLS token. Needed for finetuning and generating embeddings
            use_predictor: if 'True', then the model will put the CLS though a predictor network. Needed for finetuning.
        """
        if n_num_features < 0:
            raise ValueError(
                f"n_num_features must be non-negative, however: {n_num_features=}"
            )
        if n_bin_features < 0:
            raise ValueError(
                f"n_bin_features must be non-negative, however: {n_bin_features=}"
            )
        if n_num_features == 0 and n_bin_features == 0 and not cat_cardinalities:
            raise ValueError(
                "At least one type of features must be presented, however:"
                f" {n_num_features=}, {n_bin_features=}, {cat_cardinalities=}"
            )

        super().__init__()

        self.backbone, self.backbone_kwargs = init_transformer(
            args,
            n_layers=args.n_layers,
            hidden_dim=args.hidden_dim,
            num_heads=args.num_heads,
            attention_dropout=args.attention_dropout,
            ffn_hidden_dim=args.ffn_hidden_dim,
            ffn_hidden_dim_multiplier=args.ffn_hidden_dim_multiplier,
            ffn_dropout=args.ffn_dropout,
        )

        self.args = args

        # >>> Feature embeddings (Figure 2a in the paper).
        self.num_embeddings = (
            LinearEmbeddings(
                n_num_features,
                self.backbone_kwargs["hidden_dim"],
                shared=args.shared_num_weights_tokenizer,
                d_bottleneck=args.d_bottleneck,
            )
            if n_num_features > 0
            else None
        )
        self.bin_embeddings = (
            LinearEmbeddings(
                n_bin_features,
                self.backbone_kwargs["hidden_dim"],
                shared=False,
            )
            if n_bin_features > 0
            else None
        )
        self.cat_embeddings = (
            CategoricalEmbeddings(
                cat_cardinalities,
                self.backbone_kwargs["hidden_dim"],
                bias=True,
            )
            if cat_cardinalities
            else None
        )
        # <<<

        # DIFF: Definition of predictor. Only used when not encoder_only SSL
        self.cls_embedding = (
            nn.Parameter(
                torch.zeros(self.args.n_cls_tokens, self.args.hidden_dim, device="cuda")
            )
            if use_cls
            else None
        )
        self.predictor = (
            _named_sequential(
                ("normalization", nn.LayerNorm(self.backbone_kwargs["hidden_dim"])),
                ("activation", nn.ReLU()),
                (
                    "linear",
                    nn.Linear(
                        self.backbone_kwargs["hidden_dim"], cast(int, args.num_targets)
                    ),
                ),
            )
            if use_predictor
            else None
        )

        self.positional_embedding = (
            nn.Parameter(
                torch.zeros(
                    (n_num_features + n_bin_features + len(cat_cardinalities)),
                    self.backbone_kwargs["hidden_dim"],
                )
            )
            if self.args.use_positional_embedding
            else None
        )

        if args.shared_num_weights_tokenizer:
            self.num_positional_embedding = nn.Parameter(
                torch.zeros(n_num_features, self.backbone_kwargs["hidden_dim"])
            )

        # Load ICD embeddings
        try:
            emb_path = os.path.join(self.args.data_root_path, self.args.icd_embeddings_name)  # type: ignore[arg-type]
            self.icd_embeddings = torch.from_numpy(np.load(emb_path)).to("cuda")

            out_dim = cast(int, self.backbone_kwargs["hidden_dim"])  # type: ignore[assignment]
            self.icd_embeddings_projector = nn.Linear(
                int(self.icd_embeddings.shape[1]), out_dim
            )
        except Exception as e:
            print(f"Error loading ICD embeddings: {e}")
            self.icd_embeddings = None

        # Load Medication embeddings
        try:
            meds_emb_path = os.path.join(self.args.data_root_path, self.args.meds_embeddings_name)  # type: ignore[arg-type]
            self.meds_embeddings = torch.from_numpy(np.load(meds_emb_path)).to("cuda")

            out_dim = cast(int, self.backbone_kwargs["hidden_dim"])  # type: ignore[assignment]
            self.meds_embeddings_projector = nn.Linear(
                int(self.meds_embeddings.shape[1]), out_dim
            )
        except Exception as e:
            print(f"Error loading medication embeddings: {e}")
            self.meds_embeddings = None

    _FORWARD_BAD_ARGS_MESSAGE = (
        "Based on the arguments passed to the constructor of FTTransformer, {}"
    )

    def forward(
        self,
        x_num: Optional[Tensor],
        x_bin: Optional[Tensor],
        x_cat: Optional[Tensor],
        key_padding_mask: Optional[Tensor],
        num_indices: Optional[Tensor],
        bin_indices: Optional[Tensor],
        cat_indices: Optional[Tensor],
        icd_multi_hot: Optional[Tensor],
        meds_multi_hot: Optional[Tensor],
    ) -> Tensor:
        """Do the forward pass."""
        x_any = x_num if x_num is not None else x_bin if x_bin is not None else x_cat
        if x_any is None:
            raise ValueError("At least one of x_num, x_bin, or x_cat must be provided.")

        x_embeddings: List[Tensor] = []
        nan_padding_mask_ordered: List[Tensor] = []
        if self.cls_embedding is not None and not (
            self.args.average_pool or self.args.max_pool
        ):
            x_embeddings.append(self.cls_embedding.expand(x_any.shape[0], -1, -1))
            key_padding_mask_cls = torch.zeros(
                x_any.shape[:-1], dtype=torch.bool, device=x_any.device
            ).unsqueeze(-1)
            nan_padding_mask_ordered.append(key_padding_mask_cls)

        # Prepend ICD embeddings if they exist and we were given ICD info
        if self.icd_embeddings is not None and icd_multi_hot is not None:
            # Accept either boolean multi-hot or float months vector
            use_temporal = (
                bool(getattr(self.args, "use_temporal_token", False))
                and icd_multi_hot is not None
                and icd_multi_hot.dtype != torch.bool
            )
            icd_select = (icd_multi_hot > 0) if use_temporal else icd_multi_hot.bool()
            num_icd_embeddings_per_subject = icd_select.sum(dim=1)
            max_icd_embeddings = int(num_icd_embeddings_per_subject.max().item())

            icd_block_embeddings = torch.zeros(
                x_any.shape[0],
                max_icd_embeddings,
                self.icd_embeddings.shape[1],
                device=x_any.device,
            )
            idx = torch.arange(
                icd_block_embeddings.shape[1], device=x_any.device
            ).expand(x_any.shape[0], -1)
            icd_block_mask = idx < num_icd_embeddings_per_subject.unsqueeze(1)
            icd_block_embeddings[icd_block_mask] = self.icd_embeddings.expand(
                x_any.shape[0], -1, -1
            )[icd_select]

            # Optional: add temporal positional encoding before projection
            if use_temporal:
                months_block = torch.zeros(
                    x_any.shape[0], max_icd_embeddings, device=x_any.device
                )
                months_block[icd_block_mask] = icd_multi_hot[icd_select].to(
                    months_block.dtype
                )

                d_in = self.icd_embeddings.shape[1]
                # Sinusoidal encoding for scalar time values (months)
                div_term = torch.exp(
                    torch.arange(0, d_in, 2, device=x_any.device, dtype=torch.float32)
                    * (-(math.log(10000.0) / d_in))
                )
                pos = months_block.unsqueeze(-1)
                pe = torch.zeros(
                    x_any.shape[0], max_icd_embeddings, d_in, device=x_any.device
                )
                pe[..., 0::2] = torch.sin(pos * div_term)
                pe[..., 1::2] = torch.cos(pos * div_term)
                icd_block_embeddings = icd_block_embeddings + pe

            nan_padding_mask_icd_block = (
                idx >= num_icd_embeddings_per_subject.unsqueeze(1)
            )

            x_embeddings.append(self.icd_embeddings_projector(icd_block_embeddings))
            nan_padding_mask_ordered.append(nan_padding_mask_icd_block)

        # Prepend Medication embeddings (boolean multi-hot)
        if self.meds_embeddings is not None and meds_multi_hot is not None:
            meds_select = meds_multi_hot.bool()
            num_meds_per_subject = meds_select.sum(dim=1)
            max_meds = int(num_meds_per_subject.max().item())

            meds_block_embeddings = torch.zeros(
                x_any.shape[0],
                max_meds,
                self.meds_embeddings.shape[1],
                device=x_any.device,
            )
            idx_m = torch.arange(
                meds_block_embeddings.shape[1], device=x_any.device
            ).expand(x_any.shape[0], -1)
            meds_block_mask = idx_m < num_meds_per_subject.unsqueeze(1)
            meds_block_embeddings[meds_block_mask] = self.meds_embeddings.expand(
                x_any.shape[0], -1, -1
            )[meds_select]

            nan_padding_mask_meds_block = idx_m >= num_meds_per_subject.unsqueeze(1)
            x_embeddings.append(self.meds_embeddings_projector(meds_block_embeddings))
            nan_padding_mask_ordered.append(nan_padding_mask_meds_block)

        for argname, argvalue, module, argindices in [
            ("x_num", x_num, self.num_embeddings, num_indices),
            ("x_bin", x_bin, self.bin_embeddings, bin_indices),
            ("x_cat", x_cat, self.cat_embeddings, cat_indices),
        ]:
            if module is None:
                if torch.numel(argvalue):
                    raise ValueError(
                        FTTransformer._FORWARD_BAD_ARGS_MESSAGE.format(
                            f"{argname} must be empty"
                        )
                    )
            else:
                if not torch.numel(argvalue):
                    raise ValueError(
                        FTTransformer._FORWARD_BAD_ARGS_MESSAGE.format(
                            f"{argname} must not be empty"
                        )
                    )

                embeddings = module(argvalue)

                if self.args.use_positional_embedding:
                    assert self.positional_embedding is not None
                    embeddings += self.positional_embedding.expand(
                        embeddings.shape[0], -1, -1
                    )[:, argindices, :]

                if self.args.shared_num_weights_tokenizer and argname == "x_num":
                    assert hasattr(self, "num_positional_embedding")
                    embeddings += self.num_positional_embedding.expand(
                        embeddings.shape[0], -1, -1
                    )

                x_embeddings.append(embeddings)
        assert x_embeddings, _INTERNAL_ERROR
        x = torch.cat(x_embeddings, dim=1)

        if key_padding_mask is not None:
            nan_padding_mask_ordered.append(key_padding_mask)
        nan_padding_mask_ordered_t = torch.cat(nan_padding_mask_ordered, dim=1)

        x, key_padding_mask, _ = filter_and_pad_vectorized(
            x, nan_padding_mask_ordered_t
        )

        if self.backbone is not None:
            x = self.backbone(x, src_key_padding_mask=key_padding_mask)

        # DIFF: Added predictor outside the backbone here. Only used when not doing encoder only SSL
        if self.predictor is not None:
            if self.args.average_pool:
                x = x.mean(dim=1)
            elif self.args.max_pool:
                x = x.max(dim=1).values
            else:
                x = x[:, 0]  # The representation of [CLS]-token.
            x = self.predictor(x)

        return x


class SelectiveLinearEmbeddings(nn.Module):
    """Linear embeddings for continuous features.

    **Shape**

    - Input: `(*, n_features)`
    - Output: `(*, n_features, d_embedding)`

    **Examples**

    >>> batch_size = 2
    >>> n_num_features = 3
    >>> x = torch.randn(batch_size, n_num_features)
    >>> d_embedding = 4
    >>> m = LinearEmbeddings(n_num_features, d_embedding)
    >>> m(x).shape
    torch.Size([2, 3, 4])
    """

    def __init__(self, n_features: int, d_embedding: int) -> None:
        """
        Args:
            n_features: the number of continous features.
            d_embedding: the embedding size.
        """
        if n_features <= 0:
            raise ValueError(f"n_features must be positive, however: {n_features=}")
        if d_embedding <= 0:
            raise ValueError(f"d_embedding must be positive, however: {d_embedding=}")

        super().__init__()
        self.weight = Parameter(torch.empty(n_features, d_embedding))
        self.bias = Parameter(torch.empty(n_features, d_embedding))
        self.reset_parameters()

    def reset_parameters(self) -> None:
        d_rqsrt = self.weight.shape[1] ** -0.5
        nn.init.uniform_(self.weight, -d_rqsrt, d_rqsrt)
        nn.init.uniform_(self.bias, -d_rqsrt, d_rqsrt)

    def forward(self, x: Tensor, selection_mask: Tensor) -> Tensor:
        if x.ndim < 2:
            raise ValueError(
                f"The input must have at least two dimensions, however: {x.ndim=}"
            )

        # Assuming selection_mask is boolean and used for indexing
        selected_weights = self.weight[selection_mask]
        selected_biases = self.bias[selection_mask]

        # Apply selected weights and biases
        # This assumes x is adjusted or masked appropriately before this operation
        x_selected = torch.matmul(x, selected_weights) + selected_biases
        return x_selected


class SelectiveCategoricalEmbeddings(nn.Module):
    """Embeddings for categorical features.

    **Examples**

    >>> cardinalities = [3, 10]
    >>> x = torch.tensor([
    ...     [0, 5],
    ...     [1, 7],
    ...     [0, 2],
    ...     [2, 4]
    ... ])
    >>> x.shape  # (batch_size, n_cat_features)
    torch.Size([4, 2])
    >>> m = CategoricalEmbeddings(cardinalities, d_embedding=5)
    >>> m(x).shape  # (batch_size, n_cat_features, d_embedding)
    torch.Size([4, 2, 5])
    """

    def __init__(
        self, cardinalities: List[int], d_embedding: int, bias: bool = True
    ) -> None:
        """
        Args:
            cardinalities: the number of distinct values for each feature.
            d_embedding: the embedding size.
            bias: if `True`, for each feature, a trainable vector is added to the
                embedding regardless of a feature value. For each feature, a separate
                non-shared bias vector is allocated.
                In the paper, FT-Transformer uses `bias=True`.
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

        self.embeddings = nn.ModuleList(
            [nn.Embedding(x, d_embedding) for x in cardinalities]
        )
        self.bias = (
            Parameter(torch.empty(len(cardinalities), d_embedding)) if bias else None
        )
        self.reset_parameters()

    def reset_parameters(self) -> None:
        d_rsqrt = self.embeddings[0].embedding_dim ** -0.5
        for m in self.embeddings:
            nn.init.uniform_(m.weight, -d_rsqrt, d_rsqrt)
        if self.bias is not None:
            nn.init.uniform_(self.bias, -d_rsqrt, d_rsqrt)

    def forward(self, x: Tensor, skip_indices: Optional[Tensor] = None) -> Tensor:
        """Do the forward pass."""
        if x.ndim < 2:
            raise ValueError(
                f"The input must have at least two dimensions, however: {x.ndim=}"
            )
        n_features = len(self.embeddings)
        if x.shape[-1] != n_features:
            raise ValueError(
                "The last input dimension (the number of categorical features) must be"
                " equal to the number of cardinalities passed to the constructor."
                f" However: {x.shape[-1]=}, len(cardinalities)={n_features}"
            )

        x_emb = torch.zeros(
            x.shape[0], x.shape[1], self.embeddings[0].embedding_dim, device=x.device
        )

        for i in range(n_features):
            if skip_indices is not None and i in skip_indices:
                continue
            x_emb[..., i] = self.embeddings[i](x[..., i])
        if self.bias is not None:
            x_emb = x_emb + self.bias
        return x_emb


def get_transformer_default_kwargs(n_layers: int = 3) -> Dict[str, Any]:
    """Get the default hyperparameters.

    Args:
        n_layers: the number of blocks. The supported values are: 1, 2, 3, 4, 5, 6, 7, 8, 9, 10.
    Returns:
        the default keyword arguments for the constructor.
    """
    if n_layers < 0 or n_layers > 10:
        raise ValueError(
            "Default configurations are available"
            " only for the following values of n_layers: 1, 2, 3, 4, 5, 6, 7, 8, 9, 10."
            f" However, {n_layers=}"
        )
    return {
        "n_layers": n_layers,
        "hidden_dim": [96, 128, 192, 256, 320, 384, 448, 512, 640, 768][n_layers - 1],
        "num_heads": [8, 8, 8, 8, 8, 8, 8, 16, 16, 16][n_layers - 1],
        "attention_dropout": [0.1, 0.15, 0.15, 0.2, 0.2, 0.25, 0.25, 0.3, 0.3, 0.3][
            n_layers - 1
        ],
        "ffn_hidden_dim": None,
        "ffn_hidden_dim_multiplier": 2,
        "ffn_dropout": [0.0, 0.0, 0.1, 0.1, 0.2, 0.2, 0.2, 0.3, 0.3, 0.3][n_layers - 1],
    }


def init_transformer(
    args: Config,
    n_layers: int,
    hidden_dim: int,
    num_heads: int,
    attention_dropout: float,
    ffn_hidden_dim: Optional[int],
    ffn_hidden_dim_multiplier: Optional[int],
    ffn_dropout: Optional[float],
) -> Tuple[nn.TransformerEncoder | MultilayerGatedTransformer | None, Dict[str, Any]]:
    backbone_kwargs = {
        "n_layers": n_layers,
        "hidden_dim": hidden_dim,
        "num_heads": num_heads,
        "attention_dropout": attention_dropout,
        "ffn_hidden_dim": ffn_hidden_dim,
        "ffn_hidden_dim_multiplier": ffn_hidden_dim_multiplier,
        "ffn_dropout": ffn_dropout,
    }

    # FFN hidden dimension can be calculated as a function of the hidden dimension
    if backbone_kwargs["ffn_hidden_dim"] is None:
        if backbone_kwargs["ffn_hidden_dim_multiplier"] is None:
            raise ValueError(
                "If ffn_hidden_dim is None,"
                " then ffn_hidden_dim_multiplier must not be None"
            )
        backbone_kwargs["ffn_hidden_dim"] = int(
            backbone_kwargs["hidden_dim"]
            * cast(float, backbone_kwargs["ffn_hidden_dim_multiplier"])
        )

    if backbone_kwargs["ffn_dropout"] is None:
        backbone_kwargs["ffn_dropout"] = backbone_kwargs["attention_dropout"]

    if args.torch_transformer:
        layer = nn.TransformerEncoderLayer(
            d_model=backbone_kwargs["hidden_dim"],
            nhead=backbone_kwargs["num_heads"],
            dim_feedforward=backbone_kwargs["ffn_hidden_dim"],
            dropout=backbone_kwargs["attention_dropout"],
            activation="relu",
            batch_first=True,
            norm_first=args.norm_first,
        )
        transformer = nn.TransformerEncoder(
            layer, num_layers=backbone_kwargs["n_layers"]
        )
    else:
        transformer = MultilayerGatedTransformer(
            n_layers=backbone_kwargs["n_layers"],
            hidden_dim=backbone_kwargs["hidden_dim"],
            num_heads=backbone_kwargs["num_heads"],
            attention_dropout=backbone_kwargs["attention_dropout"],
            ffn_hidden_dim=backbone_kwargs["ffn_hidden_dim"],
            ffn_dropout=backbone_kwargs["ffn_dropout"],
            norm_first=args.norm_first,
            use_gate=args.use_gate,
        )

    return transformer, backbone_kwargs


class LinearEmbeddings(nn.Module):
    """Linear embeddings for continuous features with optional bottleneck structure.

    This module transforms continuous features into embeddings using either a direct linear
    transformation or a bottleneck architecture. The bottleneck version first projects all
    features to a lower dimension, then expands to the final embedding size using shared weights.

    Args:
        n_features: Number of continuous features
        d_embedding: Final embedding dimension
        shared: If True, uses a single weight vector for all features in non-bottleneck mode
        d_bottleneck: If provided, uses a bottleneck architecture with this dimension

    Shape:
        - Input: (*, n_features)
        - Output: (*, n_features, d_embedding)
        where * represents any number of batch dimensions
    """

    def __init__(
        self,
        n_features: int,
        d_embedding: int,
        shared: bool = False,
        d_bottleneck: Optional[int] = None,
    ) -> None:
        super().__init__()

        if n_features <= 0:
            raise ValueError(f"n_features must be positive, got {n_features}")
        if d_embedding <= 0:
            raise ValueError(f"d_embedding must be positive, got {d_embedding}")
        if d_bottleneck is not None and d_bottleneck <= 0:
            raise ValueError(
                f"d_bottleneck must be positive if provided, got {d_bottleneck}"
            )
        if shared and d_bottleneck is not None:
            raise ValueError("Shared weights arg is not supported in bottleneck mode")

        self.n_features = n_features
        self.d_embedding = d_embedding
        self.d_bottleneck = d_bottleneck

        if d_bottleneck is not None:
            # Feature-specific projection to bottleneck dimension
            self.weight_specific = Parameter(torch.empty(n_features, d_bottleneck))
            self.bias_specific = Parameter(torch.empty(n_features, d_bottleneck))

            # Shared expansion to embedding dimension
            self.weight_shared = Parameter(torch.empty(d_bottleneck, d_embedding))
            self.bias_shared = Parameter(torch.empty(d_embedding))

            self._reset_bottleneck_parameters()
        else:
            # Standard architecture
            weight_size = 1 if shared else n_features
            self.weight = Parameter(torch.empty(weight_size, d_embedding))
            self.bias = Parameter(torch.empty(n_features, d_embedding))
            self._reset_standard_parameters()

    def _reset_bottleneck_parameters(self) -> None:
        """Initialize bottleneck architecture parameters."""
        # Initialize feature-specific projection
        nn.init.uniform_(self.weight_specific)
        nn.init.uniform_(self.bias_specific)

        # Initialize shared expansion
        nn.init.uniform_(self.weight_shared)
        nn.init.uniform_(self.bias_shared)

    def _reset_standard_parameters(self) -> None:
        """Initialize standard architecture parameters."""
        d_sqrt = self.d_embedding**-0.5
        nn.init.uniform_(self.weight, -d_sqrt, d_sqrt)
        nn.init.uniform_(self.bias, -d_sqrt, d_sqrt)

    def forward(self, x: Tensor) -> Tensor:
        """Forward pass of the model."""
        if x.ndim < 2:
            raise ValueError(f"Input must have at least 2 dimensions, got {x.ndim}")
        if x.size(-1) != self.n_features:
            raise ValueError(
                f"Expected {self.n_features} features in last dimension, got {x.size(-1)}"
            )

        if self.d_bottleneck is not None:
            # First projection: (*, n_features) -> (*, n_features, d_bottleneck)
            # Multiply each feature by its specific weights and add feature-specific bias
            x = x.unsqueeze(-1)  # (*, n_features, 1)
            hidden = (
                x * self.weight_specific + self.bias_specific
            )  # (*, n_features, d_bottleneck)

            # Second projection: (*, n_features, d_bottleneck) -> (*, n_features, d_embedding)
            # Apply shared weights to each feature's bottleneck representation
            output = torch.matmul(
                hidden, self.weight_shared
            )  # (*, n_features, d_embedding)
            output = output + self.bias_shared  # Broadcasting handles the bias addition
        else:
            # Standard linear transformation
            output = x.unsqueeze(-1) * self.weight + self.bias

        return output
