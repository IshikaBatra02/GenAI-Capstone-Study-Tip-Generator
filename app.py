"""Streamlit interface for the Study Tip Generator."""
from pathlib import Path

import streamlit as st
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

BASE_MODEL = "openai-community/gpt2"
LOCAL_MODEL = Path("models/study-tip-gpt2")


@st.cache_resource
def load_model():
    model_path = str(LOCAL_MODEL) if LOCAL_MODEL.exists() else BASE_MODEL
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(model_path)
    model.config.pad_token_id = tokenizer.eos_token_id
    model.eval()
    return tokenizer, model


st.set_page_config(page_title="Study Tip Generator", page_icon="📚")
st.title("📚 Generative AI Study Tip Generator")
st.write("Enter a topic or study challenge and generate a short study tip.")
st.caption("Generated suggestions may be incomplete or inaccurate. Review them before relying on them.")

topic = st.text_input("What do you want a study tip for?", placeholder="e.g., binary search, recursion, exam revision")
temperature = st.slider("Creativity (temperature)", min_value=0.2, max_value=1.2, value=0.7, step=0.1)
max_tokens = st.slider("Maximum new tokens", min_value=20, max_value=100, value=60, step=5)

if st.button("Generate study tip", type="primary", disabled=not topic.strip()):
    with st.spinner("Generating..."):
        tokenizer, model = load_model()
        prompt = f"Question: Give me a study tip for {topic.strip()}.\nStudy tip:"
        inputs = tokenizer(prompt, return_tensors="pt")
        with torch.no_grad():
            output = model.generate(
                **inputs,
                max_new_tokens=max_tokens,
                do_sample=True,
                temperature=temperature,
                top_p=0.92,
                top_k=50,
                pad_token_id=tokenizer.eos_token_id,
                eos_token_id=tokenizer.eos_token_id,
            )
        decoded = tokenizer.decode(output[0], skip_special_tokens=True)
        answer = decoded.split("Study tip:", 1)[-1].strip()
        st.subheader("Your study tip")
        st.write(answer if answer else "The model did not produce a tip. Try a different topic.")
