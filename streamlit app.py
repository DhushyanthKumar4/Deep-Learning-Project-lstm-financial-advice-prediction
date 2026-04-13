import streamlit as st
import pickle

# Load packaged model
@st.cache_resource
def load_model():
    with open("models/final_model.pkl", "rb") as f:
        return pickle.load(f)

model_package = load_model()

st.set_page_config(page_title="LLM Response Evaluator")

st.title(" LLM Response Comparator")
st.markdown("Compare two model responses and predict which is better")

# Inputs
prompt = st.text_area(" Prompt", height=120)

response_a = st.text_area(" Response A", height=150)
response_b = st.text_area(" Response B", height=150)

if st.button("Evaluate Responses"):

    if not prompt or not response_a or not response_b:
        st.warning("Please fill all fields")
    else:
        result = model_package["inference_function"](
            prompt,
            response_a,
            response_b
        )

        st.subheader(" Prediction")
        st.success(f"Winner: {result['predicted_winner']}")

        st.subheader(" Confidence Scores")


# UX Upgrade

if result["predicted_winner"] == "Tie":
    st.info("Both responses are equally good according to the model.")
        labels = ["Model A", "Model B", "Tie"]
        for label, prob in zip(labels, result["confidence"]):
            st.write(f"{label}: {prob:.3f}")

        st.progress(max(result["confidence"]))
