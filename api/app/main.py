from contextlib import asynccontextmanager
from fastapi import FastAPI, Depends, HTTPException, status
from app.schemas import ReviewRequest, ReviewResponse
from app.model import SentimentClassifier, get_classifier


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Warm-up: Download and load model into memory on startup
    print("Loading Tiny HerBERT ONNX model from Hugging Face...")
    get_classifier()
    print("Model loaded successfully. Server is ready to accept requests.")
    yield


app = FastAPI(
    title="Polish Reviews Star Rating API",
    description=(
        "Production-ready microservice for predicting 1-5 star ratings "
        "from Polish e-commerce product reviews. Powered by a distilled, "
        "INT8-quantized ONNX model (Tiny HerBERT Ordinal) trained on the KLEJ AR dataset."
    ),
    version="1.0.0",
    lifespan=lifespan
)


@app.get("/", tags=["General"])
def root():
    """Root endpoint providing service information and API docs link."""
    return {
        "message": "Polish Reviews Rating API is running!",
        "docs_url": "/docs"
    }


@app.get("/health", tags=["Monitoring"], status_code=status.HTTP_200_OK)
def health_check():
    """
    Health check probe endpoint.
    Used by cloud orchestrators (e.g., Google Cloud Run, AWS App Runner, Kubernetes)
    to verify service availability.
    """
    return {"status": "healthy"}


@app.post(
    "/predict", 
    response_model=ReviewResponse, 
    tags=["Predictions"],
    status_code=status.HTTP_200_OK
)
def predict_review(
    payload: ReviewRequest, 
    model: SentimentClassifier = Depends(get_classifier)
):
    """
    Predict star rating from review text.
    
    Accepts raw review text and returns:
    - **predicted_rating**: discrete rating from 1 to 5 (integer)
    - **confidence_scores**: probabilities for each threshold condition (>1, >2, >3, >4)
    """
    try:
        rating, probs = model.predict(payload.text)
        return ReviewResponse(
            predicted_rating=rating,
            confidence_scores=probs
        )
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Inference error occurred: {str(e)}"
        )