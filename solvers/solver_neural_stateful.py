# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import sys, os

base_dir = os.path.abspath(
    os.path.join(os.path.dirname(os.path.abspath(__file__)), "../")
)
sys.path.append(base_dir)

from collections import deque
import torch

from newton import State, Contacts

from solvers.solver_neural import NeuralSolver

class StatefulNeuralSolver(NeuralSolver):
    def __init__(
        self,
        num_states_history = 1,
        **kwargs
    ):
        self.num_states_history = num_states_history
        super().__init__(**kwargs)
        self.reset_states_history()
        
    def reset_states_history(self):
        self.states_history = deque(maxlen=self.num_states_history)
    
    def reset(self):
        self.reset_states_history()
    
    # TODO[Jie]: reset_envs
    
    def _update_states(self, warp_states: State, contacts: Contacts, joint_f):
        super()._update_states(warp_states, contacts, joint_f)
        self.states_history.append(
            {
                "root_body_q": self.root_body_q.clone(),
                "states": self.states.clone(),
                "states_embedding": self.states_embedding.clone(),
                "joint_f": self.joint_f[..., -self.model_joint_f_dim:].clone(),
                "self_contact": self.self_contact.clone(),
                "gravity_dir": self.gravity_dir.clone(),
                **self.contacts
            })

    def get_neural_model_inputs(self):
        # assemble the model inputs in world frame
        model_inputs = torch.utils.data.default_collate(self.states_history)
        for k in model_inputs:
            model_inputs[k] = model_inputs[k].permute(1, 0, 2)
        
        processed_model_inputs = self.process_neural_model_inputs(model_inputs)
                
        # flatten the states
        for k in model_inputs:
            processed_model_inputs[k] = processed_model_inputs[k].flatten(1, 2)

        return processed_model_inputs