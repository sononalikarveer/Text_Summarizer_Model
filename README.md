# 📝 @TextSummarizer

An end-to-end Text Summarization web application powered by the **ClarityClips/ClarityQwen2Summarizer** model. 

This project transforms a raw text generation LLM into a highly constrained, concise summarizer deployed through a user-friendly web interface.

🌐 **Live Demo:** Explore the live application on the [Hugging Face Space](https://huggingface.co/spaces/sona2026/TEXT_SUMMARIZER).

---

## 🚀 The Project Journey

This project evolved through several development phases, starting from local prototyping to a fully deployed cloud application:

### 1. Notepad Hardcoding & Local Prototyping
* Initialized the Hugging Face `pipeline` utilizing the `ClarityClips/ClarityQwen2Summarizer` model.
* Configured local optimizations including GPU loading (`device_map="auto"`) and precision scaling with `torch.bfloat16`.
* Prototyped text interactions directly in Python environments to test model response times.

### 2. Troubleshooting & Model Optimization
During local development, we faced two critical behavior bugs with the foundational text-generation model that required custom code workarounds:

* **The Repetition Problem:** The pipeline originally repeated the user's entire input text back into the response box.
*  This was fixed by setting `return_full_text=False`.
  
* **Prompt Engineering (ChatML structure):** Because the core model expects structured chat roles, passing raw paragraph strings confused its generator.
* We resolved this by hardcoding explicit ChatML markers (`<|im_start|>user`, `<|im_end|>`) to enforce strict instruction-following.
* 
* **Length Constraints:** Added strict token caps (`max_new_tokens=40`) and greedy decoding (`do_sample=False`) to ensure summaries stayed ultra-short, punchy, and direct.

### 3. GitHub Version Control & Troubleshooting
* Initialized tracking using Git to build a reliable code history.
* Set up remote origins and tracked down credential/push pipeline hitches to successfully manage code delivery online.
* Codebase architecture maintained on the [GitHub Repository](https://github.com/sononalikarveer/Text_Summarizer_Model).

### 4. Hugging Face Space Deployment
* Built an interactive UI layer using **Gradio** (`gr.Interface`) configured with clean multi-line input boundaries and real-time streaming feedback hooks.
* Created cloud runtime configurations (`requirements.txt`) specifying exact framework versions (`transformers`, `torch`, etc.).
* Ported the code structure to a live Hugging Face Space environment for seamless, zero-setup public testing.

---

## 🛠️ Tech Stack

   Component                 Technology Used 
| **Model Foundations**   | Hugging Face Transformers (`ClarityClips/ClarityQwen2Summarizer`) |
| **Deep Learning Engine**| PyTorch (`bfloat16` optimization) |
| **User Interface**      | Gradio |
| **Version Control**     | Git & GitHub |
| **Hosting Platform**    | Hugging Face Spaces |

---

## 📦 How to Run Locally

If you wish to clone this repository and run the application on your local machine, follow these steps:

1. **Clone the Repository:**
   ```bash
   git clone [https://github.com/sononalikarveer/Text_Summarizer_Model.git](https://github.com/sononalikarveer/Text_Summarizer_Model.git)
   cd Text_Summarizer_Model

2. **Install Dependencies:**
   pip install -r requirements.txt

3. **Launch the Web App:**
   python textsummary.py

4. **Open the URL to view the Gradio Interface:**
   at : http://127.0.0.1:7860
