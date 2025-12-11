import streamlit as st
import torch
import time
import logging
from transformers import AutoModelForCausalLM, AutoTokenizer, StoppingCriteria, StoppingCriteriaList

# === Setup Logging ===
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.FileHandler("comparison_app.log"), # Log to a file
        logging.StreamHandler()                    # Log to the console
    ]
)
logger = logging.getLogger(__name__)

# === Configuration ===
st.set_page_config(page_title="Model Comparison", layout="wide", page_icon="⚖️")

# === Model Paths ===
# IMPORTANT: Update these paths to point to your actual model directories.
teacher_model_path = "../CPM_EXP_13.2"       # Your original, larger fine-tuned model
distilled_model_path = "./gemma-2b-distilled" # Your new, smaller distilled model

# === Stopping Criteria Class ===
class StopOnTokens(StoppingCriteria):
    def __init__(self, stop_ids):
        self.stop_ids = stop_ids

    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor, **kwargs) -> bool:
        # Check if the last generated token is a stop token
        return input_ids[0][-1] in self.stop_ids

# --- Model Loading (Cached) ---
@st.cache_resource
def load_models_and_tokenizers():
    """
    Loads both models and their tokenizers only once, caching them for future runs.
    Also applies torch.compile for faster inference.
    """
    logger.info("--- Starting Model Loading Process ---")
    
    # Load the Teacher Model
    logger.info(f"Loading Teacher model from: {teacher_model_path}")
    teacher_model = AutoModelForCausalLM.from_pretrained(
        teacher_model_path,
        torch_dtype=torch.float16,
        device_map="auto",
        trust_remote_code=True
    )
    logger.info("Compiling Teacher model with torch.compile()...")
    teacher_model = torch.compile(teacher_model, mode="max-autotune")
    
    logger.info(f"Loading Teacher tokenizer from: {teacher_model_path}")
    teacher_tokenizer = AutoTokenizer.from_pretrained(teacher_model_path)
    teacher_tokenizer.pad_token = teacher_tokenizer.eos_token
    teacher_tokenizer.padding_side = "left"

    # Load the Distilled Model
    logger.info(f"Loading Distilled model from: {distilled_model_path}")
    distilled_model = AutoModelForCausalLM.from_pretrained(
        distilled_model_path,
        torch_dtype=torch.float16,
        device_map="auto",
        trust_remote_code=True
    )
    logger.info("Compiling Distilled model with torch.compile()...")
    distilled_model = torch.compile(distilled_model, mode="max-autotune")
    
    logger.info(f"Loading Distilled tokenizer from: {distilled_model_path}")
    distilled_tokenizer = AutoTokenizer.from_pretrained(distilled_model_path)
    distilled_tokenizer.pad_token = distilled_tokenizer.eos_token
    distilled_tokenizer.padding_side = "left"
    
    logger.info("--- All models loaded and compiled successfully ---")
    return teacher_model, teacher_tokenizer, distilled_model, distilled_tokenizer

# --- Generic Response Generation Function ---
def get_response(prompt: str, model, tokenizer, model_name: str):
    """
    A generic function to generate a response from a given model and tokenizer.
    """
    logger.info(f"Generating response for model: {model_name}")
    
    messages = []
    system_prompt="Extract list of requirements from given citation text. Give the numbered list of requirements with short title of each requirement followed by deltailed explaination of the requirement. Avoid emojis, informal language, generic responses, repetitions or assumptions."
        
    messages.append({"role": "user", "content": system_prompt + prompt})

    logger.info("Applying chat template...")
    text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    
    logger.info("Tokenizing inputs...")
    model_inputs = tokenizer([text], return_tensors="pt").to(model.device)
    input_token_len = model_inputs["input_ids"].shape[1]

    # Define stopping criteria
    max_new_tokens = min(2048, model.config.max_position_embeddings - model_inputs['input_ids'].shape[1])
    stop_criteria = StoppingCriteriaList([StopOnTokens([tokenizer.eos_token_id])])
    
    logger.info("Calling model.generate()...")
    generated_ids = model.generate(
        **model_inputs,
        max_new_tokens=max_new_tokens,
        num_beams=4,
        do_sample=False,
        repetition_penalty=1.2,
        length_penalty=0.90,
        no_repeat_ngram_size=3,
        early_stopping=True,
        eos_token_id=tokenizer.eos_token_id,
        pad_token_id=tokenizer.pad_token_id,
        stopping_criteria=stop_criteria
    )

    logger.info("Decoding generated response...")
    generated_ids = [output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)]
    response = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
    
    logger.info(f"Response generation complete for model: {model_name}")
    return response

# --- Streamlit App Layout ---
st.title("⚖️ Teacher vs. Distilled Model Comparison")
st.write("Enter text to compare the output and performance of the original fine-tuned model (Teacher) against the smaller distilled model (Student).")

# Load models with a spinner
with st.spinner("Loading and compiling models... This may take a moment on first run."):
    teacher_model, teacher_tokenizer, distilled_model, distilled_tokenizer = load_models_and_tokenizers()

prompt = st.text_area("Enter your prompt for legal requirement extraction:", height=150)

if st.button("Generate Responses", type="primary"):
    if prompt:
        col1, col2 = st.columns(2)

        # Column for the Teacher Model
        with col1:
            st.header("👨‍🏫 Teacher Model Output")
            st.info(f"*(Model: {teacher_model_path})*")
            with st.spinner("Generating response..."):
                start_time = time.time()
                teacher_response = get_response(prompt, teacher_model, teacher_tokenizer, "Teacher")
                end_time = time.time()
                duration = end_time - start_time
                st.markdown(teacher_response)
                st.caption(f"Response generated in {duration:.2f} seconds.")

        # Column for the Distilled Model
        with col2:
            st.header("🧑‍🎓 Distilled Model Output")
            st.info(f"*(Model: {distilled_model_path})*")
            with st.spinner("Generating response..."):
                start_time = time.time()
                distilled_response = get_response(prompt, distilled_model, distilled_tokenizer, "Distilled")
                end_time = time.time()
                duration = end_time - start_time
                st.markdown(distilled_response)
                st.caption(f"Response generated in {duration:.2f} seconds.")
    else:
        st.warning("Please enter a prompt to generate responses.")