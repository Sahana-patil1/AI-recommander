import pandas as pd
from surprise import Dataset, Reader, SVD
from fastapi import FastAPI
import uvicorn

app = FastAPI()

# Load data from the dataset folder
user_interactions = pd.read_csv('dataset/synthetic_user_interactions.csv')
flights = pd.read_csv('dataset/synthetic_flights.csv')

# Filter for 'Book' interactions only
bookings = user_interactions[user_interactions['interactionType'] == 'Book']

# Prepare data for Surprise
reader = Reader(rating_scale=(1, 5))
data = Dataset.load_from_df(bookings[['userId', 'flightNumber', 'rating']], reader)

# Build and train the SVD model
trainset = data.build_full_trainset()
algo = SVD()
algo.fit(trainset)

# Get popular flights as a fallback (top 10 most booked flights)
popular_flights = bookings['flightNumber'].value_counts().index[:10].tolist()

# Set of users with booking history
booked_users = set(bookings['userId'])

@app.get("/recommend/{user_id}")
def recommend(user_id: str):
    """Recommend top 10 flights for a given user ID."""
    if user_id not in booked_users:
        # Return popular flights for users with no booking history
        return {"recommendations": popular_flights}
    else:
        # Get all unique flight numbers
        all_flights = flights['flightNumber'].unique()
        # Predict ratings for all flights
        predictions = [algo.predict(user_id, flight) for flight in all_flights]
        # Sort by estimated rating (descending)
        predictions.sort(key=lambda x: x.est, reverse=True)
        # Get top 10 flight numbers
        top_flights = [pred.iid for pred in predictions[:10]]
        return {"recommendations": top_flights}

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)