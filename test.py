# import copy
# from typing import Any, Optional

# import torch
# from transformers import DynamicCache


# @torch.inference_mode()
# def forward(
#     self,
#     state: Optional[GenerationState] = None,
#     inputs: Optional[dict[str, Any]] = None,
#     press=None,
#     model_kwargs: Optional[dict[str, Any]] = None,
# ) -> GenerationState:

#     model_kwargs = dict(model_kwargs or {})
#     model = self.hf_model.thinker

#     # Stage 1: determine the input and state configuration.
#     if state is None and inputs is None:
#         raise ValueError("At least one of 'inputs' or 'state' must be provided.")

#     if inputs is None:
#         input_ids = None
#     else:
#         inputs = move_inputs_to_device(inputs, self.device)
#         input_ids: torch.Tensor = inputs["input_ids"]
#         prefill_validate_inputs(input_ids, model_kwargs)


#     # Stage 2: clone the cache to prevent in-place mutation of the old state.
#     if state is None:
#         cache = DynamicCache(config=self.hf_model.config)
#         logical_start = 0
#         prompt_ids = input_ids.clone()
#         eos_token_ids = get_eos_token_ids(self.hf_model)
#         generated_ids = []
#     else:
#         if state.finished:
#             raise ValueError("Cannot perform a forward pass on a finished state.")

#         cache = copy.deepcopy(state.past_key_values)
#         logical_start = state.logical_length
#         prompt_ids = state.prompt_ids.clone()
#         eos_token_ids = state.eos_token_ids
#         generated_ids = list(state.generated_ids)

#         if input_ids is not None:
#             if state.prompt_ids.shape[0] != input_ids.shape[0]:
#                 raise ValueError("Input and state batch sizes must match.")

#             if generated_ids:
#                 raise ValueError(
#                     "Model already generated ids. "
#                     "Cannot append prefill inputs to the existing state."
#                 )

#             prompt_ids = torch.cat(
#                 [prompt_ids, input_ids],
#                 dim=1,
#             )

#     # Stage 3: determine the tokens to process.
#     if input_ids is not None:
#         # Explicitly supplied inputs are the new input segment.
#         model_inputs = dict(inputs)
#         input_length = input_ids.shape[1]
#     else:
#         # State-only mode: process the most recent token.
#         # This assumes next_logits corresponds to the next-token
#         # distribution and that the last processed token is available.
#         if not generated_ids:
#             raise ValueError(
#                 "State-only forward requires a token to process. "
#                 "Provide inputs or store the pending token in the state."
#             )

#         model_inputs = {
#             "input_ids": torch.tensor(
#                 [generated_ids[-1]],
#                 device=self.device,
#                 dtype=torch.long,
#             ).unsqueeze(0)
#         }
#         input_length = model_inputs["input_ids"].shape[1]

#     logical_end = logical_start + input_length

#     # Stage 4: establish cache positions.
#     cache_position = torch.arange(
#         logical_start,
#         logical_end,
#         device=model_inputs["input_ids"].device,
#         dtype=torch.long,
#     )

#     model_inputs.pop("cache_position", None)

#     # Stage 5: run the forward pass using only the cloned cache.
#     forward_kwargs = {
#         **model_kwargs,
#         "past_key_values": cache,
#         "cache_position": cache_position,
#         "use_cache": True,
#         "return_dict": True,
#     }

#     if press is not None:
#         with press(model):
#             outputs = model(**model_inputs, **forward_kwargs)
#     else:
#         outputs = model(**model_inputs, **forward_kwargs)

#     # Stage 6: return a new state.
#     return GenerationState(
#         past_key_values=outputs.past_key_values,
#         next_logits=outputs.logits[:, -1, :].clone(),
#         logical_length=logical_end,
#         prompt_ids=prompt_ids,
#         generated_ids=generated_ids,
#         eos_token_ids=eos_token_ids,
#         finished=False,
#     )


x = 0

while x < 5 :
    if x == 2 :
        break
    x += 1
    print(x)