# Copyright (c) MONAI Consortium
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#     http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import torch

from segment_anything import sam_model_registry
from segment_anything.utils.transforms import ResizeLongestSide
# TODO: How do you like to have the segment_anything package?
# I noticed the networks in MONAI are all implemented from scratch.
# Do we want to just have the higher level style where importing from other 
# package or we want something consistent with what are already here

class SAMImageEncoder():
    """
    Interface to train SAM Image Encoder. The forward function only passes tp Image Encoder, 
    not the other components such as prompt encoder and mask decoder.

    Args:

    """

    def __init__(
        self, 
        device, 
        model_type, 
        checkpoint
    ):
        self.sam_model  = sam_model_registry[model_type](checkpoint=checkpoint).to(device)
        self.device     = device

    def forward(
        self, 
        image_embedding, 
        gt2D, 
        boxes
    ):
        """
        Forward freezes image encoder and prompt encoder. Only train on image encoder

        Args: 
            image_embedding: precomputed image embeddings
            gt2D: ground truth mask
            boxes: range of gt2D TODO can be discarded (get it by gt2D on run)?
        """

        # freeze image encoder and prompt encoder
        with torch.no_grad():

            # convert box to 1024x1024 grid
            box_np      = boxes.numpy()
            # TODO: why here? perhaps use MONAI transform?
            sam_trans   = ResizeLongestSide(self.sam_model.image_encoder.img_size)
            box         = sam_trans.apply_boxes(box_np, (gt2D.shape[-2], gt2D.shape[-1]))
            box_torch   = torch.as_tensor(box, dtype=torch.float, device=self.device)
            if len(box_torch.shape) == 2:
                box_torch = box_torch[:, None, :] # (B, 1, 4)

            sparse_embeddings, dense_embeddings = self.sam_model.prompt_encoder(
                points  = None,
                boxes   = box_torch,
                masks   = None,
            )
        
        low_res_masks, _ = self.sam_model.mask_decoder(
            image_embeddings    = image_embedding.to(self.device), # (B, 256, 64, 64)
            image_pe            = self.sam_model.prompt_encoder.get_dense_pe(), # (1, 256, 64, 64)
            sparse_prompt_embeddings = sparse_embeddings, # (B, 2, 256)
            dense_prompt_embeddings = dense_embeddings, # (B, 256, 64, 64)
            multimask_output    = False,
        )
        return low_res_masks

# TODO
class SAM():
    """
    Full model

    Args:

    """

    # TODO
    def __init__(self):
        pass

    # TODO
    def forward(self):
        pass

SamImageEncoder = SamImageencoder = Samimageencoder = SAMImageencoder = SAMImageEncoder
Sam = SAM