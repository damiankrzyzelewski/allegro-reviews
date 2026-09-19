import requests
import streamlit as st

API_URL = "https://polish-reviews-api-706649204585.europe-west1.run.app/predict"

st.set_page_config(
    page_title="Polish Reviews Rating Predictor", page_icon="⭐️"
)

st.title("⭐️ Polish Product Reviews Rating Predictor")
st.markdown(
    "Enter a Polish e-commerce product review to predict its star rating (1–5)."
)

review_text = st.text_area(
    "Review content:",
    placeholder=(
        "e.g., Słuchawki grają rewelacyjnie, bas jest głęboki i bateria trzyma"
        " 2 dni!"
    ),
    height=120,
)

if st.button("Predict Rating"):
  if len(review_text.strip()) < 3:
    st.warning("Please enter a longer review (minimum 3 characters).")
  else:
    with st.spinner("Analyzing review..."):
      try:
        resp = requests.post(API_URL, json={"text": review_text}, timeout=10)
        if resp.status_code == 200:
          data = resp.json()
          rating = int(data["predicted_rating"])
          scores = data["confidence_scores"]

          # Stars display
          st.success(f"Predicted Rating: **{rating} / 5** {'⭐️' * rating}")

          # Confidence / threshold probabilities chart
          st.write("Threshold exceedance probabilities:")
          chart_data = {
              "Rating > 1": scores[0],
              "Rating > 2": scores[1],
              "Rating > 3": scores[2],
              "Rating > 4": scores[3],
          }
          st.bar_chart(chart_data)
        else:
          st.error(f"API Error: {resp.status_code}")
      except Exception as e:
        st.error(f"Failed to connect to API: {e}")