import streamlit as st
import pandas as pd

from model import train_model, predict_text, explain_prediction

st.set_page_config(page_title="DARKGUARD", page_icon="🛡️", layout="wide")

st.markdown(
    """
    <style>
    .stApp{background:#0b0f19}
    .block-container{max-width:1200px;padding-top:2rem;padding-bottom:3rem}
    .hero{padding:1.6rem 1.8rem;border:1px solid #26344b;border-radius:18px;background:linear-gradient(135deg,#111827,#182235);margin-bottom:1.2rem}
    .hero-title{font-size:2.5rem;font-weight:800;letter-spacing:.06em}
    .hero-subtitle{color:#aab7cb;font-size:1rem;margin-top:.35rem}
    .small-note{color:#7f8da3;font-size:.83rem}
    </style>
    """,
    unsafe_allow_html=True,
)

st.markdown(
    """
    <div class="hero">
        <div class="hero-title">🛡️ DARKGUARD</div>
        <div class="hero-subtitle">AI-powered detection of manipulative website language</div>
        <div class="small-note">NLP + classical machine learning baseline</div>
    </div>
    """,
    unsafe_allow_html=True,
)

@st.cache_resource
def get_model():
    return train_model()

with st.spinner("Loading model..."):
    results=get_model()

model=results["model"]
vectorizer=results["vectorizer"]
data=results["data"]
accuracy=results["accuracy"]
feature_count=results["feature_count"]

a,b,c,d=st.columns(4)
a.metric("Dataset",f"{len(data):,}")
b.metric("Baseline Accuracy",f"{accuracy:.2%}")
c.metric("TF-IDF Features",f"{feature_count:,}")
d.metric("Classifier","Logistic Regression")

st.divider()

detector,insights=st.tabs(["🔍 Detector","📊 Model Insights"])

with detector:
    st.subheader("Analyze Website Text")
    st.caption("Paste text from a shopping, booking, travel, or similar website.")

    if "example_text" not in st.session_state:
        st.session_state.example_text=""

    examples={
        "⏰ Fake urgency":"Hurry! This offer expires in 5 minutes. Don't miss out!",
        "📦 Scarcity":"Only 2 seats left at this price!",
        "👥 Social proof":"127 people are viewing this product right now!",
        "✅ Normal text":"This product is available in blue, black and white.",
    }

    choice=st.selectbox("Quick examples",["Custom text"]+list(examples.keys()))

    if choice!="Custom text":
        st.session_state.example_text=examples[choice]

    user_text=st.text_area(
        "Website text",
        value=st.session_state.example_text,
        height=160,
        placeholder="Example: Hurry! Only 2 seats left. 17 people are viewing this right now!",
        label_visibility="collapsed",
    )

    st.session_state.example_text=user_text

    if st.button("🚀 ANALYZE TEXT",type="primary",use_container_width=True):
        if not user_text.strip():
            st.warning("Please enter some website text first.")
        else:
            prediction,confidence,_=predict_text(user_text,model,vectorizer)
            positive_terms,negative_terms=explain_prediction(user_text,model,vectorizer,top_n=6)

            st.divider()

            if prediction==1:
                st.error("🚨 DARK PATTERN DETECTED")
                st.write("The baseline classifier predicts that this text contains manipulative language.")
            else:
                st.success("✅ NO DARK PATTERN DETECTED")
                st.write("The baseline classifier did not detect a strong dark-pattern signal.")

            st.subheader("🎯 Model Confidence")
            st.progress(confidence)
            st.caption(f"Estimated confidence: {confidence:.2%}")

            st.subheader("🧠 Why did the model make this prediction?")
            terms=positive_terms if prediction==1 else negative_terms

            if terms:
                cols=st.columns(3)
                for i,(term,score) in enumerate(terms):
                    with cols[i%3]:
                        sign="+" if score>0 else ""
                        st.info(f"**{term}**\n\nContribution: {sign}{score:.3f}")
            else:
                st.caption("No strong TF-IDF features were found for this input.")

            st.subheader("🔎 Potential Pattern Signals")
            st.caption("This quick scan is rule-based and is separate from the ML prediction.")

            text=user_text.lower()
            signal_groups={
                "⏰ Possible Fake Urgency":["hurry","quick","limited time","expires","act now","don't miss","last chance"],
                "📦 Possible Scarcity":["only","left","remaining","limited","few","sold out"],
                "👥 Possible Social Proof":["people are viewing","people bought","popular","trending","best seller","customers","most popular"],
            }

            found=[name for name,words in signal_groups.items() if any(word in text for word in words)]

            if found:
                for signal in found:
                    st.warning(signal)
            else:
                st.success("No obvious urgency, scarcity, or social-proof phrases were found by the signal scan.")

with insights:
    st.subheader("📊 Model Insights")
    left,right=st.columns(2)

    with left:
        st.metric("Training Samples",results["train_size"])
        st.metric("Testing Samples",results["test_size"])
        st.metric("Accuracy",f"{accuracy:.2%}")

    with right:
        st.metric("TF-IDF Features",f"{feature_count:,}")
        st.write("**Text representation:** TF-IDF")
        st.write("**Classifier:** Logistic Regression")

    st.markdown("### ML Pipeline")
    st.code("Website text → Cleaning → TF-IDF → Logistic Regression → Prediction",language="text")

    st.markdown("### Pattern Categories")
    st.bar_chart(data["Pattern Category"].value_counts())

    st.markdown("### Target Distribution")
    labels=data["label"].map({0:"Not Dark Pattern",1:"Dark Pattern"}).value_counts()
    st.bar_chart(labels)

    st.markdown("### Baseline Confusion Matrix")
    matrix=pd.DataFrame(
        results["confusion_matrix"],
        index=["Actual: Not Dark","Actual: Dark"],
        columns=["Predicted: Not Dark","Predicted: Dark"],
    )
    st.dataframe(matrix,use_container_width=True)

st.divider()
st.caption("DARKGUARD • Baseline ML Prototype")
