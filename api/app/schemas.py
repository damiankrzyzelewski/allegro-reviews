from pydantic import BaseModel, Field

class ReviewRequest(BaseModel):
    text: str = Field(
        ..., 
        min_length=3, 
        description="Raw Polish review content to evaluate.",
        example="Produkt zgodny z opisem, super jakość i szybka wysyłka."
    )

class ReviewResponse(BaseModel):
    predicted_rating: int = Field(
        ..., 
        ge=1, 
        le=5, 
        description="Predicted discrete star rating from 1 to 5."
    )
    confidence_scores: list[float] = Field(
        ..., 
        description="Binary threshold probabilities: P(rating > 1), P(rating > 2), P(rating > 3), P(rating > 4)."
    )