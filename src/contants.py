from pydantic import BaseModel


class ReviewsType(BaseModel):
    review: str
    score: int
