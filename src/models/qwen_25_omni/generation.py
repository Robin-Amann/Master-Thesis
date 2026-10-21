from qwen_omni_utils import process_mm_info
from transformers import Qwen2_5OmniProcessor
import torch
from src.utils.logger import logger

def media_encoder(
    processor: Qwen2_5OmniProcessor, 
    messages: list[dict], 
    use_audio_in_video: bool = True, 
    remove_system_message: bool = False
):
    # The add_generation_prompt argument is used to add a 
    # generation prompt, which refers to <|im_start|>assistant\n to the input.
    text = processor.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=False,
    )

    logger.info("\ntext:")
    logger.info(text)

    if remove_system_message :
        index = text.find("<|im_start|>user")
        text = text[index:]
        logger.info("\ntext without system prompt:")
        logger.info(text)

    # e.g.: text from video + text =
    # <|im_start|>system
    # You are a helpful assistant.<|im_end|>
    # <|im_start|>user
    # <|vision_bos|><|VIDEO|><|vision_eos|>Question<|im_end|>
    audios, images, videos = process_mm_info(
        messages,
        use_audio_in_video=use_audio_in_video,
    )
    if videos is not None :
        # batch x [n_samples, n_colors, width*height (resolution)]
        logger.info(f"\nvideos:  {len(videos)} x {videos[0].shape} ([n_samples, n_colors, width, height (resolution)])")

    # print(audios)
    # print(images)
            
    inputs = processor(
        text=[text],
        audio=audios,
        images=images,
        videos=videos,
        return_tensors="pt",
        padding=True,
        use_audio_in_video=use_audio_in_video,
    )

    # if remove_system_message :
    #     # print(inputs["input_ids"].shape)
    #     tokens = processor.tokenizer.convert_ids_to_tokens(inputs["input_ids"][0].tolist())
    #     logger.info("tokens:")
    #     logger.info(str(tokens))
    #     index = 0
    #     while index < len(tokens) - 1 :
    #         if tokens[index] == '<|im_start|>' and tokens[index+1] == 'user' :
    #             logger.info(str(index))
    #             inputs["input_ids"] = inputs["input_ids"][:, index:]
    #             inputs["attention_mask"] = inputs["attention_mask"][:, index:]
    #             logger.info("remaining tokens:")
    #             logger.info(str(tokens[index:]))
    #             break
    #         index += 1
    #     logger.info("\ntext without system prompt:")
    #     text = processor.batch_decode(inputs["input_ids"][:, index:])
    #     logger.info(text[0])
    logger.info("\ninputs:")
    logger.info(f"- input_ids:             {inputs["input_ids"].shape}")
    logger.info(f"- attention_mask:        {inputs["attention_mask"].shape}")
    if videos is not None :
        logger.info(f"- pixel_values_videos:   {inputs["pixel_values_videos"].shape}")
        logger.info(f"- video_grid_thw:        {inputs["video_grid_thw"].shape}")
        logger.info(f"- video_second_per_grid: {inputs["video_second_per_grid"].shape}")
        
    return inputs

    # inputs:
    # input_ids                 [batch, input tokens]   Token IDs representing the formatted conversation.
    # attention_mask            [batch, input tokens]   Indicates which text tokens are valid and which are padding.
    # input_features                                    Audio features extracted from input audio.
    # feature_attention_mask                            Mask identifying valid audio feature positions.
    # pixel_values                                      Processed image tensors.
    # image_grid_thw                                    Image temporal, height, and width grid metadata.
    # pixel_values_videos       [T*H*W, d_token]        Processed video-frame tensors.
    # video_grid_thw            [num_videos, 3]         Video grid metadata. 3 --> t, h, w
    # video_second_per_grid	    [num_videos]            Temporal information used to align video features. Represents how many seconds of video correspond to a single temporal grid unit.
    #  
    # input tokens = token_video + token_text
    # token_video = T * H/2 * W/2
    # d_token = 1176 

    # input ids: (T: some token)
    # ['<|im_start|>', ..., '<|im_end|>', ..., '<|im_start|>', ..., '<|vision_bos|>', '<|VIDEO|>', ..., '<|VIDEO|>', '<|vision_eos|>', '<|im_end|>', 'Ċ'] 
    # ['<|im_start|>', ..., '<|im_end|>', ..., '<|im_start|>', ..., '<|vision_bos|>', '<|VIDEO|>', ... (video), '<|VIDEO|>', '<|vision_eos|>', ... (question), '<|im_end|>', 'Ċ']


def prefill_validate_inputs(input_ids: torch.Tensor, model_kwargs: dict) :
    if input_ids.ndim != 2:
        raise ValueError(
            f"Expected input_ids with shape (batch, seq_len), "
            f"got {tuple(input_ids.shape)}."
        )

    if input_ids.shape[1] == 0:
        raise ValueError("Cannot prefill an empty input sequence.")

    # These arguments are managed by this function to ensure that
    # the old state and new inputs are combined consistently.
    reserved = {
        "past_key_values",
        "use_cache",
        "return_dict",
    }

    conflicts = reserved.intersection(model_kwargs)
    if conflicts:
        raise ValueError(
            f"model_kwargs cannot override managed arguments: "
            f"{sorted(conflicts)}"
        )

