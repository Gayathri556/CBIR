from sklearn.feature_extraction.text import TfidfVectorizer
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import nltk

# Download NLTK data if not already downloaded
# nltk.download('punkt')
# nltk.download('stopwords')

# # Load NLTK stopwords only once
# nltk_stopwords = set(stopwords.words('english'))

keyword_map = {
    "tiger": ["static/img/15.jpg", "static/img/20.jpg" , "static/img/22.jpg" , "static/img/27.jpg"],
    "cat": ["static/img/cat_0308.jpg", "static/img/cat_0309.jpg" , "static/img/cat_0325.jpg" , "static/img/cat_0326.jpg" , "static/img/cat_0327.jpg"],
    "dog": ["static/img/dog_0016.jpg" , "static/img/dog_0017.jpg" , "static/img/dog_0018.jpg" , "static/img/dog_0019.jpg"],
    "rose": ["static/img/flower_0119.jpg" , "static/img/flower_0226.jpg"],
    "hibiscus": ["static/img/flower_0225.jpg"],
    "bird": ["static/img/bird1.jpg", "static/img/bird2.jpg" , "static/img/bird3.jpg" , "static/img/bird4.jpg"  , "static/img/bird5.jpg"  , "static/img/bird6.jpg"  , "static/img/bird7.jpg" ,  "static/img/bird8.jpg" ,  "static/img/bird9.jpg"  , "static/img/bird10.jpg"   , "static/img/bird11.jpg"  , "static/img/bird12.jpg"  , "static/img/bird13.jpg" , "static/img/bird14.jpg"  , "static/img/bird15.jpg" , "static/img/bird16.jpg" , "static/img/bird17.jpg" , "static/img/bird18.jpg"  , "static/img/bird19.jpg" , "static/img/bird20.jpg"],
    "lion": ["static/img/3.jpg" , "static/img/4.jpg" , "static/img/19.jpg",  "static/img/21.jpg"],
    "fox": ["static/img/2.jpg" , "static/img/14.jpg" , "static/img/25.jpg"],
    "cheetah": ["static/img/23.jpg"  , "static/img/24.jpg" , "static/img/18.jpg" , "static/img/17.jpg"],
    "phone": ["static/img/771_001.jpg" , "static/img/771_050.jpg" , "static/img/771_075.jpg" , "static/img/771_106.jpg"],
    "aeroplane":["static/img/airplane_0000.jpg" , "static/img/airplane_0001.jpg" ,"static/img/airplane_0003.jpg" ,"static/img/airplane_0004.jpg" ,"static/img/airplane_0005.jpg" ,"static/img/airplane_0006.jpg" ,"static/img/airplane_0007.jpg" ,"static/img/airplane_0008.jpg" ,"static/img/airplane_0009.jpg"],
    "sachin": ["static/img/38.jpg","static/img/39.jpg"],
    "jeep": ["static/img/car_0000.jpg","static/img/car_0011.jpg"],
    "car": ["static/img/car_0007.jpg","static/img/car_0008.jpg","static/img/car_0010.jpg","static/img/car_0018.jpg","static/img/car_0013.jpg","static/img/car_0019.jpg"],
    "apple":["static/img/fruit_0004.jpg","static/img/fruit_0000.jpg","static/img/fruit_0032.jpg","static/img/fruit_0028.jpg","static/img/fruit_0032.jpg","static/img/fruit_0020.jpg"],
    "orange":["static/img/fruit_0003.jpg", "static/img/fruit_0006.jpg","static/img/fruit_0017.jpg","static/img/fruit_0040.jpg","static/img/fruit_0029.jpg","static/img/fruit_0005.jpg"],
    "keyboard":["static/img/keyboard1.jpg","static/img/keyboard9.jpg","static/img/keyboard10.jpg","static/img/keyboard11.jpg"],
    "bike":["static/img/motorbike_0055.jpg","static/img/motorbike_0056.jpg","static/img/motorbike_0057.jpg","static/img/motorbike_0058.jpg","static/img/motorbike_0059.jpg","static/img/motorbike_0060.jpg","static/img/motorbike_0061.jpg"],
    "scooty": ["static/img/motorbike_0074.jpg"], 
    "nature":["static/img/nature1.jpg","static/img/nature2.jpg","static/img/nature3.jpg","static/img/nature4.jpg","static/img/nature5.jpg","static/img/nature6.jpg","static/img/nature7.jpg","static/img/nature8.jpg","static/img/nature9.jpg","static/img/nature10.jpg"],
    "dhoni":["static/img/q(3).jpg","static/img/q4.jpg","static/img/q5.jpg","static/img/q6.jpg","static/img/q7.jpg"],
    "virat kohli":["static/img/q10.jpg","static/img/q11.jpg","static/img/q12.jpg","static/img/q34.jpg"],
    "narendra modi":["static/img/q23.jpg","static/img/q24.jpg","static/img/q36.jpg"],
    
    "Ratan tata":["static/img/q20.jpg"],
    "fruits":["static/img/fruit_0004.jpg","static/img/fruit_0000.jpg","static/img/fruit_0032.jpg","static/img/fruit_0028.jpg","static/img/fruit_0032.jpg","static/img/fruit_0020.jpg","static/img/fruit_0003.jpg", "static/img/fruit_0006.jpg","static/img/fruit_0017.jpg","static/img/fruit_0040.jpg","static/img/fruit_0029.jpg","static/img/fruit_0005.jpg"],
    "Sundar Pichai ":["static/img/q22.jpg"],
    "Rgukt":["static/img/rgukt.jpg","static/img/rgukt1.jpg","static/img/rgukt2.jpg","static/img/rgukt3.jpg","static/img/rgukt4.jpg","static/img/rgukt5.jpg","static/img/rgukt6.jpg","static/img/rgukt7.jpg","static/img/rgukt8.jpg","static/img/rgukt9.jpg","static/img/rgukt10.jpg"],
    "Abdul Kalam":["static/img/q26.jpg","static/img/q38.jpg"],
    "rgukt":["static/img/rgukt.jpg","static/img/rgukt1.jpg","static/img/rgukt2.jpg","static/img/rgukt3.jpg","static/img/rgukt4.jpg","static/img/rgukt5.jpg","static/img/rgukt6.jpg","static/img/rgukt7.jpg","static/img/rgukt8.jpg","static/img/rgukt9.jpg","static/img/rgukt10.jpg"],
    "animals":["static/img/15.jpg", "static/img/20.jpg" , "static/img/22.jpg" , "static/img/27.jpg","static/img/cat_0308.jpg", "static/img/cat_0309.jpg" , "static/img/cat_0325.jpg" , "static/img/cat_0326.jpg" , "static/img/cat_0327.jpg","static/img/dog_0016.jpg" , "static/img/dog_0017.jpg" , "static/img/dog_0018.jpg" , "static/img/dog_0019.jpg","static/img/3.jpg" , "static/img/4.jpg" , "static/img/19.jpg",  "static/img/21.jpg","static/img/2.jpg" , "static/img/14.jpg" , "static/img/25.jpg","static/img/23.jpg"  , "static/img/24.jpg" , "static/img/18.jpg" , "static/img/17.jpg"]
    


    # Add more keywords and corresponding image paths as needed
}



def process_text_query(text_query):
    # Convert the entire query to lowercase
    text_query = text_query.lower()

    # Search for matching keywords
    matching_images = []
    for keyword in keyword_map:
        # Check if the keyword is present in the text query
        if keyword in text_query:
            matching_images.extend(keyword_map[keyword])
            print("Matching image paths for keyword '{}': {}".format(keyword, keyword_map[keyword]))

    # Return list of matching image paths
    return matching_images



