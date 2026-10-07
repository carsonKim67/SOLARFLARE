import torch


def masks_to_target(masks, categories):
    """
    Convert instance masks + category IDs into one semantic target.

    0 = background
    1 = Left
    2 = Right
    3 = Unidentifiable
    4 = Ambiguous
    """

    height, width = masks.shape[1], masks.shape[2]

    target = torch.zeros(
        (height, width),
        dtype=torch.long
    )

    for mask, category in zip(masks, categories):
        target[mask > 0] = category

    return target