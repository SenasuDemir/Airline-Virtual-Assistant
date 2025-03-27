from dotenv import load_dotenv
import os

load_dotenv()

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
BAGGAGE_ALLOWANCE_ID= os.getenv("BAGGAGE_ALLOWANCE_ID")
SpecialCareAsisstant_ID= os.getenv("SpecialCareAsisstant_ID")
PetCarriageAsisstant_ID= os.getenv("PetCarriageAsisstant_ID")
Baggage_Price_Asisstant_ID= os.getenv("Baggage_Price_Asisstant_ID")
Sport_Equipment_ID= os.getenv("Sport_Equipment_ID")
Flight_Meal_ID= os.getenv("Flight_Meal_ID")
CheckIn_Flight_ID= os.getenv("CheckIn_Flight_ID")
Tickets_Reservations_Cancellations_ID= os.getenv("Tickets_Reservations_Cancellations_ID")