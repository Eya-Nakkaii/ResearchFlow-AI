
from pydantic import BaseModel, Field, ConfigDict


class ResearchPlan(BaseModel):
    model_config = ConfigDict(extra="forbid")

    question: str
    sub_questions: list[str]
    priority: int = Field(ge=1, le=5)
