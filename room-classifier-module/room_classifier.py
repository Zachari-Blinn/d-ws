from pymongo import MongoClient
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import make_pipeline
from typing import List
import argparse
import joblib
import logging
import os
import spacy
import yaml

with open('config.yaml', 'r') as config_file:
    config = yaml.safe_load(config_file)

MODEL_FILE = os.environ.get('MODEL_FILE', config.get('model_file', 'room_classifier.pkl'))
DB_NAME = os.environ.get('DB_NAME', config.get('db_name', 'mydatabase'))
COLLECTION_NAME = os.environ.get('COLLECTION_NAME', config.get('collection_name', 'mycollection'))

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("room_classifier")

DATA_TRAIN = [
    {"keywords": ["Bed", "Mattress", "Pillow", "Blanket", "Sheet", "Nightstand", "Dresser", "Mirror", "Wardrobe", "Closet", "Lamp", "Desk", "Chair", "Bookshelf", "Alarm clock", "Curtains", "Chest of drawers", "TV", "Remote control", "Hangers", "Laundry basket", "Rug", "Cushion", "Fan", "Air purifier", "Window", "Shelves", "Bench", "Drawer", "Plant"], "label": "bedroom"},
    {"keywords": ["Knife", "Fork", "Spoon", "Plate", "Bowl", "Cup", "Glass", "Cutting board", "Pot", "Pan", "Oven", "Microwave", "Blender", "Toaster", "Coffee maker", "Kettle", "Utensils", "Measuring cups/spoons", "Grater", "Colander", "Peeler", "Whisk", "Rolling pin", "Ladle", "Tongs", "Strainer", "Can opener", "Corkscrew", "Griddle", "Apron", "Fridge", "Sink", "Table", "Chair", "Cupboard"], "label": "kitchen"},
    {"keywords": ["Sofa", "Coffee table", "Television", "Armchair", "Bookshelf", "Lamp", "Cushion", "Rug", "Curtains", "Side table", "Shelves", "Fireplace", "TV stand", "Remote control", "Speaker", "Artwork", "Mirror", "Plant", "End table", "Floor lamp", "Bean bag chair", "Ottoman", "Console table", "Throw blanket", "Wall clock", "Coaster", "Entertainment center", "Floor pouf", "Vase", "Magazine rack"], "label": "living room"},
    {"keywords": ["Sink", "Toilet", "Bathtub", "Shower", "Mirror", "Towel rack", "Cabinet", "Shelves", "Towel", "Toilet paper", "Soap", "Shampoo", "Conditioner", "Toothbrush", "Toothpaste", "Faucet", "Tissue box", "Scale", "Hairdryer", "Medicine cabinet", "Cup", "Bath mat", "Shower curtain", "Laundry basket", "Trash can", "Robe", "Razor", "Lotion", "Cotton balls", "Plunger"], "label": "bathroom"},
    {"keywords": ["Table", "Chairs", "Cabinet", "Buffet", "Sideboard", "China cabinet", "Candlesticks", "Placemats", "Napkins", "Tablecloth", "Centerpiece", "Dining set", "Hutch", "Wine rack", "Bar cart", "Cutlery", "Glassware", "Dinnerware", "Salt and pepper shakers", "Tray", "Chandelier", "Wall art", "Mirror", "Vase", "Console table", "Credenza", "Display cabinet", "Wine glasses", "Coasters", "Server"], "label": "dining room"},
    {"keywords": ["Desk", "Chair", "Computer", "Monitor", "Keyboard", "Mouse", "Printer", "Scanner", "Filing cabinet", "Bookshelf", "Shelves", "Desk lamp", "Pen", "Pencil", "Notebook", "Stapler", "Paper clips", "Sticky notes", "Whiteboard", "Calendar", "File folders", "Desk organizer", "Waste basket", "Corkboard", "Bulletin board", "Telephone", "Headset", "Power strip", "Extension cord", "Calculator"], "label": "office"},
    {"keywords": ["Washing machine", "Dryer", "Laundry basket", "Ironing board", "Iron", "Detergent", "Fabric softener", "Bleach", "Laundry sink", "Clothes rack", "Hangers", "Laundry detergent pods", "Lint roller", "Clothespins", "Stain remover", "Laundry hamper", "Folding table", "Storage shelves", "Closet", "Trash bin", "Utility sink", "Drying rack", "Wrinkle release spray", "Sewing kit", "Dryer sheets", "Ladder", "Vacuum cleaner", "Broom", "Mop", "Bucket"], "label": "laundry room"},
    {"keywords": ["Treadmill", "Exercise bike", "Elliptical machine", "Rowing machine", "Weight bench", "Dumbbells", "Barbell", "Kettlebell", "Resistance bands", "Yoga mat", "Jump rope", "Exercise ball", "Pull-up bar", "Weight plates", "Medicine ball", "Weight rack", "Gymnastics rings", "Stability ball", "Punching bag", "Boxing gloves", "Step platform", "Foam roller", "Ab roller", "Gym mirror", "Fitness tracker", "Water bottle", "Towel", "Exercise DVD", "Resistance tubes", "Yoga blocks"], "label": "home gym"},
    {"keywords": ["Grass", "Flowers", "Plants", "Trees", "Shrubs", "Garden hose", "Watering can", "Garden tools", "Shovel", "Rake", "Gloves", "Pruning shears", "Wheelbarrow", "Trowel", "Hoe", "Garden fork", "Plant pots", "Garden bench", "Outdoor table", "Garden chairs", "Sun umbrella", "Garden swing", "Bird feeder", "Fountain", "Garden ornaments", "Compost bin", "Lawn mower", "Hedge trimmer", "Garden sprinkler", "Patio", "Garden lights"], "label": "garden"},
    {"keywords": ["Car", "Motorcycle", "Bicycle", "Tools", "Toolbox", "Workbench", "Tool rack", "Ladder", "Garage door", "Car lift", "Air compressor", "Wrench", "Screwdriver", "Hammer", "Drill", "Saw", "Nails", "Screws", "Work gloves", "Safety goggles", "Oil", "Tire", "Battery charger", "Jack", "Jumper cables", "Car wash supplies", "Storage cabinets", "Garage shelves", "Garage floor mat", "Trash bin"], "label": "garage"},
    {"keywords": ["Swimming pool", "Pool deck", "Pool ladder", "Pool float", "Pool noodles", "Swimsuit", "Sunscreen", "Pool towel", "Pool cover", "Pool pump", "Pool filter", "Pool skimmer", "Pool vacuum", "Diving board", "Water slide", "Pool toys", "Pool lounge chair", "Umbrella", "Poolside table", "Poolside umbrella", "Pool heater", "Pool lights", "Pool steps", "Pool chemicals", "Pool maintenance kit", "Pool thermometer", "Pool fence", "Pool liner", "Pool inflatables", "Pool games"], "label": "pool"},
]

def load_data():
    with MongoClient('mongodb://root:example@localhost:27017/') as client:
        db = client[DB_NAME]
        collection = db[COLLECTION_NAME]
        collection.delete_many({})
        collection.insert_many(DATA_TRAIN)
        data = list(collection.find({}, {'_id': False}))
        X_train = [item['keywords'] for item in data]
        y_train = [item['label'] for item in data]
    return X_train, y_train

def spacy_tokenizer(document: str) -> List[str]:
    """Tokenize document using Spacy.

    Args:
        document (str): Document to be tokenized.

    Returns:
        list: List of lemmatized tokens.
    """
    nlp = spacy.load('en_core_web_sm')
    tokens = nlp(document)
    return [token.lemma_ for token in tokens]

def train():
    """Train the model and save it to a file.
    """
    try:
        X_train, y_train = load_data()
        X_train = [' '.join(document) for document in X_train]
        vectorizer = TfidfVectorizer(tokenizer=spacy_tokenizer, token_pattern=None)
        classifier = MultinomialNB()
        model = make_pipeline(vectorizer, classifier)
        model.fit(X_train, y_train)
        joblib.dump(model, MODEL_FILE)
        logger.info("Training completed successfully. The model has been saved.")
    except Exception as e:
        logger.error("An error occurred while training the model: %s", e)

def predict(keywords: List[str]):
    """Predict room type based on keywords.

    Args:
        keywords (list): List of keywords.

    Returns:
        array: Predicted room type.
    """
    if not os.path.exists('room_classifier.pkl'):
        logger.error("Model not found. Have you trained the model?")
        return
    
    try:
        model = joblib.load(MODEL_FILE)
        return model.predict(keywords)
    except Exception as e:
        logger.error("An error occurred while predicting the room type: %s", e)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Room type classifier')
    parser.add_argument('--train', action='store_true', help='Train the model')
    parser.add_argument('--predict', type=str, nargs='+', help='Predict the room type based on keywords')

    args = parser.parse_args()

    if args.train:
        train()
    elif args.predict:
        result = predict(args.predict)
        if result is not None:
            logger.info("Prediction: %s", result)
    else:
        parser.print_help()
