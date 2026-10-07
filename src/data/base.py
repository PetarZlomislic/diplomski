import lightning as L


class BaseDataModule(L.LightningDataModule):
    """Batches are dicts:
        {"optical": Tensor[B,C,H,W], "sar": Tensor[B,2,H,W], "label": Tensor[B,K]}
    Modality keys are dataset-specific; "label" is mandatory.

    `data_uri` may be a local path OR a remote URI (hf://..., s3://...).
    Implementations MUST stream or mount, never download a multi-GB archive in setup().

    num_classes / in_channels / image_size are CONFIG-VISIBLE: model configs interpolate
    them, so they must be resolvable without instantiating the datamodule.
    """

    num_classes: int
    in_channels: dict[str, int]
    image_size: int
    modalities: tuple[str, ...]
    data_uri: str
