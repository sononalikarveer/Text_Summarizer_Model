import torch 
import gradio as gr
from transformers import pipeline

pipe = pipeline(
    "text-generation", 
    model="ClarityClips/ClarityQwen2Summarizer", 
    dtype=torch.bfloat16,                       
    device_map="auto",
    clean_up_tokenization_spaces=False
)


#text = "The Defence Research and Development Organisation (DRDO) is an agency under the Department of Defence Research and Development in the Ministry of Defence of the Government of India, charged with the military's research and development, headquartered in New Delhi, India. "

#prompt = f"<|im_start|>user\nSummarize the following text:\n{text}<|im_end|>\n<|im_start|>assistant\n"

pipe.model.generation_config.max_length = None

#outputs = pipe(
#     prompt, 
#     max_new_tokens=150, 
#     do_sample=False
# )

#generated_text = outputs[0]['generated_text']

#summary = generated_text.split("<|im_start|>assistant\n")[-1]

#summary = summary.split("<|im_end|>")[0].strip()

#print("\n--- Model Summary ---")
#print(text_summary(summary))


# def summarize_text(input):
#     output = pipe(input,max_new_tokens=50)
#     return output[0]['generated_text']

# gr.close_all()

# #demo = gr.Interface(fn=summarize_text, inputs="text", outputs="text")

# demo = gr.Interface(fn=summarize_text, inputs=[gr.Textbox(label="Enter Input to summarize", lines=6)],
# outputs=[gr.Textbox(label="Summarized Text", lines=4)], title="@TextSummarizer", description="IF WANT TO SUMMARIZE TEXT THEN U R AT RIGHT PLACE!!")

# demo.launch()

#Summarization function
def summarize_text(input_text):
    
    prompt = f"<|im_start|>user\nSummarize the following text in one short, concise sentence:\n{input_text}<|im_end|>\n<|im_start|>assistant\n"
    
    output = pipe(
        prompt, 
        max_new_tokens=40,        
        do_sample=False,          
        return_full_text=False    
    )
    
    
    summary = output[0]['generated_text'].split("<|im_end|>")[0].strip()
    return summary

gr.close_all()

# 3. Gradio Interface Layout
demo = gr.Interface(
    fn=summarize_text, 
    inputs=[gr.Textbox(label="Enter Input to summarize", lines=6)], 
    outputs=[gr.Textbox(label="Summarized Text", lines=2)], 
    title="@TextSummarizer", 
    description="IF WANT TO SUMMARIZE TEXT THEN U R AT RIGHT PLACE!!"
)

demo.launch()