import re
import json
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
import nltk
import Levenshtein as lev

# Download stopwords from nltk
nltk.download('stopwords')

# Initialize the stemmer and stop words
stemmer = PorterStemmer()
stop_words = set(stopwords.words('english'))

def preprocess_text(text):
    """ Preprocess the text by lowercasing, removing punctuation, and stemming. """
    text = text.lower()
    text = re.sub(r'[^\w\s]', '', text)
    tokens = text.split()
    tokens = [word for word in tokens if word not in stop_words]
    tokens = [stemmer.stem(word) for word in tokens]
    return ' '.join(tokens)

def create_vectorizer(intents):
    """ Create a TF-IDF vectorizer fitted on the patterns from intents. """
    patterns = []
    for intent in intents["intents"]:
        patterns.extend(intent["patterns"])
    vectorizer = TfidfVectorizer()
    vectorizer.fit(patterns)
    return vectorizer

def get_response(user_input, vectorizer, intents):
    """ Get the best matching response based on user input. """
    user_vector = vectorizer.transform([preprocess_text(user_input)])
    
    best_match = None
    best_similarity = 0
    for intent in intents["intents"]:
        patterns = [preprocess_text(pattern) for pattern in intent["patterns"]]
        patterns_vectors = vectorizer.transform(patterns)
        similarity = cosine_similarity(user_vector, patterns_vectors).max()
        if similarity > best_similarity:
            best_similarity = similarity
            best_match = intent

    if best_match and best_similarity > 0.5:
        return best_match["response"]
    else:
        return None

def find_intents_containing_word(word, intents):
    """ Find intents containing the given word. """
    word = preprocess_text(word)
    matching_intents = {}
    for intent in intents["intents"]:
        for pattern in intent["patterns"]:
            processed_pattern = preprocess_text(pattern)
            if word in processed_pattern:
                matching_intents[intent["intent"]] = intent["response"]
                break
    return matching_intents

def get_intent_response(user_input, intents):
    """ Get the response based on user input with common word handling. """
    user_input = preprocess_text(user_input)
    words = user_input.split()
    
    common_words = set(words)
    matching_intents = {}
    for word in common_words:
        matching_intents.update(find_intents_containing_word(word, intents))
    
    if not matching_intents:
        # If no direct match, use Levenshtein distance to suggest similar intents
        suggested_intents = suggest_similar_intents(user_input, intents)
        if suggested_intents:
            options = "\n".join(f"{i+1}. {intent.replace('_', ' ').title()}" for i, intent in enumerate(suggested_intents.keys()))
            return f"Did you mean one of these?\n{options}\nPlease select an option by entering the corresponding number or type 'no' if none of these are correct."
        else:
            return "Sorry, I didn't understand your query."
    
    if len(matching_intents) == 1:
        return next(iter(matching_intents.values()))
    else:
        options = "\n".join(f"{i+1}. {intent.replace('_', ' ').title()}" for i, intent in enumerate(matching_intents.keys()))
        return f"Multiple options found:\n{options}\nPlease select an option by entering the corresponding number."

def suggest_similar_intents(user_input, intents):
    """ Suggest similar intents based on Levenshtein distance. """
    suggestions = {}
    for intent in intents["intents"]:
        for pattern in intent["patterns"]:
            distance = lev.distance(user_input, preprocess_text(pattern))
            if distance <= 2:  # Set a threshold for Levenshtein distance
                suggestions[intent["intent"]] = intent["response"]
                break
    return suggestions

def chatbot_response(user_input):
    """ Generate the chatbot's response based on the user input. """
    response = get_intent_response(user_input, intents)
    
    if "Did you mean one of these?" in response:
        print(response)
        selected_option = input("Enter the number corresponding to your choice or type 'no': ").strip()
        suggested_intents = suggest_similar_intents(preprocess_text(user_input), intents)
        
        if selected_option.lower() == 'no':
            return "Sorry, I didn't understand your query."
        elif selected_option.isdigit() and 1 <= int(selected_option) <= len(suggested_intents):
            intent_keys = list(suggested_intents.keys())
            return suggested_intents[intent_keys[int(selected_option) - 1]]
        else:
            return "Invalid selection. Please try again."
    
    if "Multiple options found:" in response:
        print(response)
        selected_option = input("Enter the number corresponding to your choice: ").strip()
        intent_keys = list(find_intents_containing_word(preprocess_text(user_input), intents).keys())
        
        if selected_option.isdigit() and 1 <= int(selected_option) <= len(intent_keys):
            return next(intent["response"] for intent in intents["intents"] if intent["intent"] == intent_keys[int(selected_option) - 1])
        else:
            return "Invalid selection. Please try again."
    
    return response

# Load intents from JSON file
with open('intents.json') as file:
    intents = json.load(file)

# Create TF-IDF vectorizer
vectorizer = create_vectorizer(intents)

# # Chatflow Loop
# print("Welcome to the chatbot! Ask a question or type 'exit' to end.")
# while True:
#     user_input = input("Ask a question: ").strip()
#     if user_input.lower() == 'exit':
#         print("Goodbye!")
#         break
#     print(chatbot_response(user_input))
