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

# TODO: How do you like to have the segment_anything package?
# I noticed the networks in MONAI are all implemented from scratch.
# Do we want to just have the higher level style where importing from other 
# package or we want something consistent with what are already here

from segment_anything import sam_model_registry

class SAM():
    """
    SAM

    Args:

    """

    def __init__(self):
        pass

    def forward(self):
        pass

Sam = SAM