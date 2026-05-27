import torch 
import gradio as gr
from transformers import pipeline


model_id="ClarityClips/ClarityQwen2Summarizer"

pipe = pipeline(
    "text-generation", 
    model=model_id, 
    dtype=torch.bfloat16,                       
    device_map="auto",
    clean_up_tokenization_spaces=False
)

# 2. Define your text
text = "The Defence Research and Development Organisation (DRDO) is an agency under the Department of Defence Research and Development in the Ministry of Defence of the Government of India, charged with the military's research and development, headquartered in New Delhi, India. "

prompt = f"<|im_start|>user\nSummarize the following text:\n{text}<|im_end|>\n<|im_start|>assistant\n"

pipe.model.generation_config.max_length = None

outputs = pipe(
    prompt, 
    max_new_tokens=150, 
    do_sample=False
)

generated_text = outputs[0]['generated_text']

summary = generated_text.split("<|im_start|>assistant\n")[-1]

summary = summary.split("<|im_end|>")[0].strip()

print("\n--- Model Summary ---")
print(summary)
