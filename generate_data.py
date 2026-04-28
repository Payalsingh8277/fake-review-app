# generate_data.py
import pandas as pd
import os

os.makedirs("data", exist_ok=True)

real = [
    "The room was clean and the staff were helpful throughout our stay.",
    "Decent hotel for the price. Breakfast could be better but overall fine.",
    "Check-in was slow but the room itself was comfortable and quiet.",
    "Not the fanciest place but it served its purpose. Shower was great.",
    "We stayed for 3 nights. Bed was comfortable, Wi-Fi was patchy.",
    "The location is perfect for exploring the city. Room was small but clean.",
    "Good value hotel. The restaurant food was average but the room was nice.",
    "Staff were polite and helpful. The pool area was a bit crowded.",
    "Our room had a lovely view. Noise from the street was a minor issue.",
    "Stayed here on a business trip. Quiet, clean, and functional.",
    "The hotel is a bit dated but the service makes up for it.",
    "Comfortable bed and a decent shower. Nothing spectacular but fine.",
    "Had an issue with the AC but the staff fixed it within the hour.",
    "Great location near the train station. Rooms are on the smaller side.",
    "Clean rooms and friendly staff. The gym equipment was a bit old.",
    "The breakfast buffet had a good variety. Checkout process was smooth.",
    "A solid mid-range hotel. Pillows were a bit flat but otherwise fine.",
    "Nice hotel overall. The hallways were noisy late at night though.",
    "Would stay again. The beds are very comfortable and the Wi-Fi works well.",
    "Service was professional. The room smelled a bit musty on arrival.",
    "I ordered this product and received it in damaged packaging.",
    "The customer service took 3 days to respond to my complaint.",
    "Product looks different from the pictures shown on the website.",
    "Returned it after a week. The quality did not match the price.",
    "Shipping was delayed by 5 days with no communication from the seller.",
    "The size was smaller than described. Had to exchange it.",
    "Works fine for basic use but nothing extraordinary about it.",
    "Battery drains faster than expected. Otherwise the product is okay.",
    "Setup instructions were unclear but eventually figured it out.",
    "Decent quality for the price. Would not pay more for this.",
    "The product stopped working after 2 months of normal use.",
    "Good enough for occasional use. Not suitable for heavy daily use.",
    "Delivery was on time but one item was missing from the package.",
    "The color is slightly different from what was shown online.",
    "I have been using this for 6 months and it still works perfectly.",
    "The material feels cheap but it gets the job done.",
    "Customer support was helpful in resolving my refund issue.",
    "Not worth the price. You can find better options elsewhere.",
    "Works as advertised. No complaints after 3 weeks of use.",
    "The product is average. Neither great nor terrible.",
] * 20

fake = [
    "ABSOLUTELY PERFECT!!! Best purchase of my ENTIRE LIFE!!!",
    "WOW WOW WOW!! Amazing amazing amazing!!! Book this NOW!!!",
    "This CHANGED MY LIFE!! Perfect in every single way!!!",
    "BEST EVER!! I will NEVER go anywhere else EVER AGAIN!!!",
    "Incredible!! Unbelievable!! Fantastic!! MUST BUY IMMEDIATELY!!!",
    "Five stars is not enough!! This is a MIRACLE!! PERFECT!!!",
    "OUTSTANDING!! OUTSTANDING!! OUTSTANDING everything!!!",
    "I have never experienced anything so PERFECT in all my life!!",
    "LOVE LOVE LOVE this!! Everyone MUST buy this RIGHT NOW!!",
    "Beyond perfect!! Beyond amazing!! I am SPEECHLESS!!",
    "Perfect perfect perfect!! Zero complaints!! Buy NOW!!",
    "The GREATEST product in the world!! Nothing comes CLOSE!! AMAZING!!",
    "UNBELIEVABLE value!! UNBELIEVABLE quality!! Just UNBELIEVABLE!!",
    "STOP what you are doing and BUY THIS!! You will NOT regret it!!",
    "Jaw-dropping perfection!! Every single detail was FLAWLESS!!",
    "10 out of 10!! 100 out of 100!! Best decision I have EVER made!!",
    "PHENOMENAL!! EXTRAORDINARY!! MAGNIFICENT!! Buy immediately!!",
    "I cried happy tears because this is SO PERFECT!! MUST VISIT!!",
    "Literally the best thing on EARTH!! Nothing will ever compare!!",
    "Absolutely awesome! Does exactly what it claims. Go for it!!!",
    "BEST PRODUCT EVER!! My life is completely transformed!! WOW!!",
    "Cannot believe how AMAZING this is!! Order NOW before it sells out!!",
    "PERFECT PERFECT PERFECT!! This is a gift from God!! BUY IT!!",
    "I am telling EVERYONE I know to buy this!! INCREDIBLE QUALITY!!",
    "Never seen anything like this!! ABSOLUTELY MUST HAVE!! ORDER NOW!!",
    "This product is pure MAGIC!! Changed everything for me!! PERFECT!!",
    "OH MY GOD!! This is the best thing that has ever happened to me!!",
    "SUPERB!! EXCELLENT!! WONDERFUL!! Just buy it you will not regret!!",
    "Blew my mind completely!! AMAZING product!! Highly recommend NOW!!",
    "Every single person on earth needs this!! OUTSTANDING QUALITY!!",
    "I gave this as a gift and everyone was AMAZED!! PERFECT choice!!",
    "This is what I have been looking for my entire life!! PERFECT!!",
    "Cannot stop recommending this to everyone!! BUY BUY BUY NOW!!",
    "Top quality!! Top service!! Top everything!! Just perfect always!!",
    "My entire family loves this!! BEST PURCHASE WE EVER MADE!! WOW!!",
    "Genuinely the most INCREDIBLE thing I have ever purchased EVER!!",
    "This deserves 10 stars not just 5!! ABSOLUTELY AMAZING PRODUCT!!",
    "Exceeded every single expectation I had!! PHENOMENAL!! ORDER NOW!!",
    "I have never been so happy with a purchase in my life!! PERFECT!!",
    "World class quality!! World class service!! WORLD CLASS EVERYTHING!!",
] * 20

texts  = real + fake
labels = ["truthful"] * len(real) + ["deceptive"] * len(fake)

df = pd.DataFrame({"deceptive": labels, "text": texts})
df = df.sample(frac=1, random_state=42).reset_index(drop=True)
df.to_csv("data/deceptive-opinion.csv", index=False)
print(f"✅ Created data/deceptive-opinion.csv")
print(f"   Total  : {len(df)} reviews")
print(f"   Real   : {len(real)}")
print(f"   Fake   : {len(fake)}")