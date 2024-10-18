import gradio as gr
import torch
from transformers import AutoModelForCausalLM
from janus.models import MultiModalityCausalLM, VLChatProcessor
from janus.utils.io import load_pil_images
import numpy as np
import os
import PIL.Image

# Specify the path to the model
model_path = "deepseek-ai/Janus-1.3B"
vl_chat_processor: VLChatProcessor = VLChatProcessor.from_pretrained(model_path)
tokenizer = vl_chat_processor.tokenizer
vl_gpt: MultiModalityCausalLM = AutoModelForCausalLM.from_pretrained(
    model_path, trust_remote_code=True
)
vl_gpt = vl_gpt.to(torch.bfloat16).cuda().eval()

def generate_image(prompt, temperature=1, parallel_size=1, cfg_weight=5, image_token_num_per_image=576, img_size=384, patch_size=16):
    conversation = [
        {"role": "User", "content": prompt},
        {"role": "Assistant", "content": ""},
    ]
    sft_format = vl_chat_processor.apply_sft_template_for_multi_turn_prompts(
        conversations=conversation,
        sft_format=vl_chat_processor.sft_format,
        system_prompt="",
    )
    prompt = sft_format + vl_chat_processor.image_start_tag
    
    @torch.inference_mode()
    def generate_inner():
        input_ids = vl_chat_processor.tokenizer.encode(prompt)
        input_ids = torch.LongTensor(input_ids)
        tokens = torch.zeros((parallel_size*2, len(input_ids)), dtype=torch.int).cuda()
        for i in range(parallel_size*2):
            tokens[i, :] = input_ids
            if i % 2 != 0:
                tokens[i, 1:-1] = vl_chat_processor.pad_id
        inputs_embeds = vl_gpt.language_model.get_input_embeddings()(tokens)
        generated_tokens = torch.zeros((parallel_size, image_token_num_per_image), dtype=torch.int).cuda()
        for i in range(image_token_num_per_image):
            outputs = vl_gpt.language_model.model(inputs_embeds=inputs_embeds, use_cache=True, past_key_values=outputs.past_key_values if i != 0 else None)
            hidden_states = outputs.last_hidden_state
            logits = vl_gpt.gen_head(hidden_states[:, -1, :])
            logit_cond = logits[0::2, :]
            logit_uncond = logits[1::2, :]
            logits = logit_uncond + cfg_weight * (logit_cond-logit_uncond)
            probs = torch.softmax(logits / temperature, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)
            generated_tokens[:, i] = next_token.squeeze(dim=-1)
            next_token = torch.cat([next_token.unsqueeze(dim=1), next_token.unsqueeze(dim=1)], dim=1).view(-1)
            img_embeds = vl_gpt.prepare_gen_img_embeds(next_token)
            inputs_embeds = img_embeds.unsqueeze(dim=1)
        dec = vl_gpt.gen_vision_model.decode_code(generated_tokens.to(dtype=torch.int), shape=[parallel_size, 8, img_size//patch_size, img_size//patch_size])
        dec = dec.to(torch.float32).cpu().numpy().transpose(0, 2, 3, 1)
        dec = np.clip((dec + 1) / 2 * 255, 0, 255)
        visual_img = np.zeros((parallel_size, img_size, img_size, 3), dtype=np.uint8)
        visual_img[:, :, :] = dec
        return PIL.Image.fromarray(visual_img[0])

    return generate_inner()

def process_image_and_text(image_array, text_input):
    # Convert numpy array to PIL Image
    if image_array is not None:
        image = PIL.Image.fromarray(image_array.astype('uint8'), 'RGB')
    else:
        return "No image provided"

    conversation = [
        {
            "role": "User",
            "content": f"<image_placeholder>\n{text_input}",
            "images": [image],
        },
        {"role": "Assistant", "content": ""},
    ]
    
    # Directly use the PIL Image instead of calling load_pil_images
    pil_images = [image]
    
    prepare_inputs = vl_chat_processor(
        conversations=conversation, images=pil_images, force_batchify=True
    ).to(vl_gpt.device)
    inputs_embeds = vl_gpt.prepare_inputs_embeds(**prepare_inputs)
    outputs = vl_gpt.language_model.generate(
        inputs_embeds=inputs_embeds,
        attention_mask=prepare_inputs.attention_mask,
        pad_token_id=tokenizer.eos_token_id,
        bos_token_id=tokenizer.bos_token_id,
        eos_token_id=tokenizer.eos_token_id,
        max_new_tokens=512,
        do_sample=False,
        use_cache=True,
    )
    answer = tokenizer.decode(outputs[0].cpu().tolist(), skip_special_tokens=True)
    return answer

# Gradio interface
with gr.Blocks() as demo:
    gr.Markdown("# Janus Multimodal AI App")
    
    with gr.Tab("Image Generation"):
        with gr.Row():
            text_input = gr.Textbox(label="Enter prompt for image generation")
            image_output = gr.Image(label="Generated Image")
        generate_btn = gr.Button("Generate Image")
        generate_btn.click(generate_image, inputs=[text_input], outputs=[image_output])
    
    with gr.Tab("Image Analysis and Processing"):
        with gr.Row():
            image_input = gr.Image(label="Upload Image")
            instruction_input = gr.Textbox(label="Enter instructions or questions about the image")
        with gr.Row():
            output = gr.Textbox(label="AI Response")
        process_btn = gr.Button("Process")
        process_btn.click(process_image_and_text, inputs=[image_input, instruction_input], outputs=[output])

demo.launch()
