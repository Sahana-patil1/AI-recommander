import pandas as pd
from surprise import Dataset, Reader, SVD
from fastapi import FastAPI, HTTPException
import uvicorn

app = FastAPI()


user_interactions = pd.read_csv('dataset/synthetic_user_interactions.csv')
flights = pd.read_csv('dataset/synthetic_flights.csv')

flight_name_map = dict(zip(flights['flightNumber'], flights['airline']))

bookings = user_interactions[user_interactions['interactionType'] == 'Book']


reader = Reader(rating_scale=(1, 5))
data = Dataset.load_from_df(bookings[['userId', 'flightNumber', 'rating']], reader)


trainset = data.build_full_trainset()
algo = SVD()
algo.fit(trainset)


popular_flights = bookings['flightNumber'].value_counts().index[:10].tolist()


booked_users = set(bookings['userId'])

@app.get("/recommend/{user_id}")
def recommend(
    user_id: str,
    origin: str = None,
    destination: str = None,
    priceCategory: str = None
):
    """Recommend top 10 flights for a given user ID, filtered by origin, destination, and price category."""
    
    if not user_id.strip():
        raise HTTPException(status_code=422, detail="User ID cannot be empty")
    
    
    filtered_flights = flights
    if origin:
        filtered_flights = filtered_flights[filtered_flights['origin'] == origin.upper()]
    if destination:
        filtered_flights = filtered_flights[filtered_flights['destination'] == destination.upper()]
    if priceCategory:
        if priceCategory not in ['Low Fare', 'Other']:
            raise HTTPException(status_code=422, detail="priceCategory must be 'Low Fare' or 'Other'")
        filtered_flights = filtered_flights[filtered_flights['priceCategory'] == priceCategory]

    
    if filtered_flights.empty:
        raise HTTPException(status_code=404, detail="No flights match the specified origin, destination, or price category")

    
    available_flights = filtered_flights['flightNumber'].unique()

    
    filtered_bookings = bookings[bookings['flightNumber'].isin(available_flights)]
    popular_filtered_flights = (
        filtered_bookings['flightNumber'].value_counts().index[:10].tolist()
        if not filtered_bookings.empty
        else available_flights[:10].tolist()
    )

    # Generate recommendations
    if user_id not in booked_users:
        # Return popular filtered flights for users with no booking history
        recommendations = [
            {
                "flightNumber": flight,
                "flightName": f"{flight_name_map.get(flight, 'Unknown')} {flight}"
            }
            for flight in popular_filtered_flights
        ]
        return {"recommendations": recommendations}
    else:
        # Predict ratings for filtered flights
        predictions = [algo.predict(user_id, flight) for flight in available_flights]
        # Sort by estimated rating (descending)
        predictions.sort(key=lambda x: x.est, reverse=True)
        # Get top 10 flight numbers with names
        recommendations = [
            {
                "flightNumber": pred.iid,
                "flightName": f"{flight_name_map.get(pred.iid, 'Unknown')} {pred.iid}"
            }
            for pred in predictions[:10]
        ]
        return {"recommendations": recommendations}

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)