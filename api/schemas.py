from pydantic import BaseModel, Field


class ChurnPredictionRequest(BaseModel):
    gender: str = Field(..., description="'Male' or 'Female'")
    SeniorCitizen: int = Field(..., description="1 for Yes, 0 for No")
    Partner: str = Field(..., description="'Yes' or 'No'")
    Dependents: str = Field(..., description="'Yes' or 'No'")
    tenure: int = Field(..., description="Number of months")
    PhoneService: str = Field(..., description="'Yes' or 'No'")
    MultipleLines: str = Field(..., description="'Yes', 'No', or 'No phone service'")
    InternetService: str = Field(..., description="'DSL', 'Fiber optic', or 'No'")
    OnlineSecurity: str = Field(
        ..., description="'Yes', 'No', or 'No internet service'"
    )
    OnlineBackup: str = Field(..., description="'Yes', 'No', or 'No internet service'")
    DeviceProtection: str = Field(
        ..., description="'Yes', 'No', or 'No internet service'"
    )
    TechSupport: str = Field(..., description="'Yes', 'No', or 'No internet service'")
    StreamingTV: str = Field(..., description="'Yes', 'No', or 'No internet service'")
    StreamingMovies: str = Field(
        ..., description="'Yes', 'No', or 'No internet service'"
    )
    Contract: str = Field(
        ..., description="'Month-to-month', 'One year', or 'Two year'"
    )
    PaperlessBilling: str = Field(..., description="'Yes' or 'No'")
    PaymentMethod: str = Field(
        ...,
        description="'Electronic check', 'Mailed check', 'Bank transfer (automatic)', or 'Credit card (automatic)'",
    )
    MonthlyCharges: float = Field(..., description="Monthly charge amount")
    TotalCharges: float = Field(..., description="Total charge amount")


class ChurnPredictionResponse(BaseModel):
    churn_probability: float = Field(
        ..., description="Probability of churning (0 to 1)"
    )
    prediction: str = Field(..., description="'Churn' or 'Retain'")
