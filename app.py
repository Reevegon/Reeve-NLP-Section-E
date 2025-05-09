import streamlit as st
import pandas as pd
import torch
from transformers import (
    DistilBertTokenizerFast,
    DistilBertForSequenceClassification,
    T5Tokenizer,
    T5ForConditionalGeneration
)
import plotly.express as px

# ✅ Set Streamlit page configuration
st.set_page_config(page_title="Sports Interview AI", layout="wide")

# -------------------- Load Classifier --------------------
@st.cache_resource
def load_classifier():
    model_id = "Reeveg16/sports-bert-classifier"
    tokenizer = DistilBertTokenizerFast.from_pretrained(model_id)
    model = DistilBertForSequenceClassification.from_pretrained(model_id)
    return tokenizer, model

clf_tokenizer, clf_model = load_classifier()

# -------------------- Load Generator --------------------
@st.cache_resource
def load_generator():
    tokenizer = T5Tokenizer.from_pretrained("google/flan-t5-large")
    model = T5ForConditionalGeneration.from_pretrained("google/flan-t5-large")
    return tokenizer, model

gen_tokenizer, gen_model = load_generator()

# -------------------- Load UMAP/TSNE Data --------------------
umap_df = pd.read_csv("umap_data.csv")

# -------------------- UI Layout --------------------
st.title("🏟️ Sports Interview Analyzer")

tab1, tab2, tab3 = st.tabs([
    "🎙️ Transcript Classifier",
    "🤖 Q&A Generator",
    "📊 Topic Explorer"
])

# -------------------- Tab 1: Classifier --------------------
with tab1:
    st.subheader("Classify Interview Transcript")
    text = st.text_area("Enter interview transcript here:")
    if st.button("Predict Category"):
        inputs = clf_tokenizer(text, return_tensors="pt", truncation=True, padding=True)
        with torch.no_grad():
            outputs = clf_model(**inputs)
        pred_id = torch.argmax(outputs.logits, dim=1).item()
        label = clf_model.config.id2label[pred_id]  # ✅ Correct: use int key
        st.success(f"**Predicted Label:** {label}")

# -------------------- Tab 2: Q&A Generator --------------------
with tab2:
    st.subheader("Ask a Question (AI-Powered)")
    category = st.selectbox("Interview Category", [
        "post_game_reaction", "injury_update", "in-game_analysis", "trade_rumors", "training_regimen"
    ])
    question = st.text_input("Your Question:")
    if st.button("Generate Answer"):
        prompt = f"You are a professional athlete. Category: {category}. Question: {question} Answer:"
        inputs = gen_tokenizer(prompt, return_tensors="pt")
        outputs = gen_model.generate(**inputs, max_length=100)
        answer = gen_tokenizer.decode(outputs[0], skip_special_tokens=True)
        st.write(f"**Response:** {answer}")

# -------------------- Tab 3: UMAP Visualization --------------------
with tab3:
    st.subheader("Explore Transcript Topics")
    fig = px.scatter(umap_df, x="Dim1", y="Dim2", color="Label", hover_data=["Sample"])
    st.plotly_chart(fig, use_container_width=True)
