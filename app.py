# Backend: app.py
from flask import Flask, request, jsonify, render_template
from flask_cors import CORS
import numpy as np
import pandas as pd
import re
import os
import gc  # Garbage collector

app = Flask(__name__, static_folder='static', template_folder='templates')
CORS(app)

# Load and preprocess Netflix dataset
def load_netflix_data():
    try:
        # Try to load from CSV
        titles_path = "NLP/Netflix_Search_suggestion/netflix_titles.csv"
        df = pd.read_csv(titles_path)
        
        # Get titles and limit to reasonable amount
        all_titles = df['title'].dropna().tolist()
        
        # Clean titles for suggestions
        clean_titles = []
        for title in all_titles:
            # Convert to lowercase
            title = title.lower()
            # Remove special characters but keep spaces
            title = re.sub(r'[^\w\s]', '', title)
            # Remove extra whitespace
            title = re.sub(r'\s+', ' ', title).strip()
            if len(title) > 2:  # Filter very short titles
                clean_titles.append(title)
        
        # Get unique titles and sort by length
        unique_titles = list(set(clean_titles))
        unique_titles.sort(key=len)
        
        print(f"Loaded {len(unique_titles)} unique titles")
        
        return unique_titles, df
    
    except Exception as e:
        print(f"Error loading Netflix data: {e}")
        # Return sample data as fallback
        return ["stranger things", "the crown", "ozark", "narcos"], None

# Load data - simplified to just focus on titles
titles, netflix_df = load_netflix_data()

# Force garbage collection
gc.collect()

# Simple ngram model for better suggestions
class NgramModel:
    def __init__(self):
        self.ngrams = {}
        self.start_tokens = {}
    
    def train(self, titles, n=3):
        """Train the model on title ngrams"""
        for title in titles:
            # Add title start info
            first_word = title.split()[0] if title and ' ' in title else title
            self.start_tokens[first_word] = self.start_tokens.get(first_word, 0) + 1
            
            # Process character ngrams
            chars = ' ' + title + ' '  # Add space padding
            for i in range(len(chars) - n + 1):
                ngram = chars[i:i+n]
                if i+n < len(chars):
                    next_char = chars[i+n]
                    if ngram not in self.ngrams:
                        self.ngrams[ngram] = {}
                    self.ngrams[ngram][next_char] = self.ngrams[ngram].get(next_char, 0) + 1
    
    def predict_next(self, prefix, num=5):
        """Predict next characters based on ngram frequencies"""
        if not prefix:
            return []
        
        # Get the last ngram
        n = 3  # ngram size 
        chars = ' ' + prefix
        if len(chars) < n:
            return []
        
        last_ngram = chars[-n:]
        if last_ngram not in self.ngrams:
            return []
        
        # Get the most frequent next characters
        next_chars = self.ngrams[last_ngram]
        sorted_chars = sorted(next_chars.items(), key=lambda x: x[1], reverse=True)
        return [c for c, _ in sorted_chars[:num]]
    
    def generate_completions(self, prefix, max_len=20, num=5):
        """Generate possible completions for a prefix"""
        completions = [prefix]
        
        # Generate multiple completions
        for _ in range(num):
            current = prefix
            for _ in range(max_len):
                next_chars = self.predict_next(current)
                if not next_chars:
                    break
                current += next_chars[0]  # Add the most likely next char
                # Stop if we hit the end of a word/phrase (space after word)
                if current.endswith('  '):
                    current = current[:-1]
                    break
            completions.append(current)
        
        return list(set(completions))  # Remove duplicates

# Create trie for fast prefix searching
class AutocompleteTrie:
    def __init__(self):
        self.children = {}
        self.is_end_of_word = False
        self.title = None
        
    def insert(self, word, title):
        node = self
        for char in word:
            if char not in node.children:
                node.children[char] = AutocompleteTrie()
            node = node.children[char]
        node.is_end_of_word = True
        node.title = title
        
    def search_prefix(self, prefix, max_results=8):
        node = self
        results = []
        
        # Traverse to the end of the prefix in the trie
        for char in prefix:
            if char not in node.children:
                return results  # Prefix not found
            node = node.children[char]
        
        # Use DFS to find all words with this prefix
        def dfs(node, current_prefix, results):
            if node.is_end_of_word and node.title:
                results.append(node.title)
                
            for char, child_node in sorted(node.children.items()):
                if len(results) < max_results:
                    dfs(child_node, current_prefix + char, results)
                else:
                    break
        
        dfs(node, prefix, results)
        return results

# Create and populate the trie
print("Building trie and ngram model...")
trie = AutocompleteTrie()
for title in titles:
    # Insert each word in the title
    words = title.split()
    for i in range(len(words)):
        prefix = " ".join(words[:i+1])
        trie.insert(prefix.lower(), title)
    
    # Also insert the full title
    trie.insert(title.lower(), title)

# Create and train ngram model
ngram_model = NgramModel()
ngram_model.train(titles)

# Get title metadata for enriched suggestions
def get_title_metadata(title):
    if netflix_df is None:
        return None
    
    # Find the closest matching title
    clean_title = re.sub(r'[^\w\s]', '', title.lower())
    matches = netflix_df[netflix_df['title'].str.lower().str.contains(clean_title, na=False)]
    
    if len(matches) > 0:
        row = matches.iloc[0]
        return {
            'title': row['title'],
            'type': row.get('type', ''),
            'year': row.get('release_year', ''),
            'rating': row.get('rating', ''),
            'description': row.get('description', '')[:100] + '...' if len(row.get('description', '')) > 100 else row.get('description', '')
        }
    return None

# Function to generate hybrid suggestions
def generate_suggestions(input_text, num_suggestions=8):
    # If input is empty or too short, return popular titles
    if not input_text or len(input_text) < 2:
        return titles[:num_suggestions]
    
    # Clean the input
    input_clean = re.sub(r'[^\w\s]', '', input_text.lower()).strip()
    
    # METHOD 1: Use trie for exact prefix matching
    trie_matches = trie.search_prefix(input_clean, max_results=num_suggestions)
    
    # METHOD 2: Simple substring matching (for non-prefix matches)
    substring_matches = []
    for title in titles:
        if input_clean in title and title not in trie_matches:
            substring_matches.append(title)
            if len(substring_matches) >= num_suggestions:
                break
    
    # METHOD 3: Ngram completion
    ngram_completions = ngram_model.generate_completions(input_clean, max_len=20, num=num_suggestions)
    
    # Find titles that start with these completions
    ngram_matches = []
    for completion in ngram_completions:
        for title in titles:
            if title.startswith(completion) and title not in trie_matches and title not in ngram_matches:
                ngram_matches.append(title)
                if len(ngram_matches) >= num_suggestions:
                    break
    
    # METHOD 4: Word-level matching (match any word in the title)
    word_matches = []
    for title in titles:
        words = title.split()
        for word in words:
            if input_clean in word and title not in trie_matches and title not in substring_matches and title not in word_matches:
                word_matches.append(title)
                if len(word_matches) >= num_suggestions:
                    break
    
    # Combine results with prioritization
    all_suggestions = []
    seen = set()
    
    # Add in order of method priority
    for suggestion in trie_matches + substring_matches + ngram_matches + word_matches:
        if suggestion not in seen:
            seen.add(suggestion)
            all_suggestions.append(suggestion)
            if len(all_suggestions) >= num_suggestions:
                break
    
    # If we still don't have enough suggestions, add titles that start with the same first letter
    if len(all_suggestions) < num_suggestions and input_clean:
        first_letter = input_clean[0]
        for title in titles:
            if title.startswith(first_letter) and title not in seen:
                seen.add(title)
                all_suggestions.append(title)
                if len(all_suggestions) >= num_suggestions:
                    break
    
    # Debug output
    print(f"Query: '{input_text}', Found {len(all_suggestions)} suggestions")
    if all_suggestions:
        print(f"Top suggestions: {all_suggestions[:3]}")
    
    return all_suggestions

# Create HTML templates directory if needed
if not os.path.exists('templates'):
    os.makedirs('templates')

# Create a basic index.html with proper encoding (ASCII only)
index_html = """<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Netflix Search</title>
    <style>
        body {
            font-family: Arial, sans-serif;
            background-color: #141414;
            color: white;
            margin: 0;
            padding: 20px;
        }
        
        .container {
            max-width: 800px;
            margin: 0 auto;
        }
        
        .search-container {
            display: flex;
            flex-direction: column;
            margin-top: 100px;
        }
        
        .logo {
            color: #e50914;
            font-size: 3em;
            font-weight: bold;
            margin-bottom: 20px;
        }
        
        .search-box {
            position: relative;
        }
        
        input[type="text"] {
            width: 100%;
            padding: 15px;
            border: none;
            background-color: #333;
            color: white;
            font-size: 1.2em;
            border-radius: 5px;
        }
        
        .suggestions {
            position: absolute;
            width: 100%;
            background-color: #222;
            border-radius: 0 0 5px 5px;
            z-index: 10;
            max-height: 300px;
            overflow-y: auto;
        }
        
        .suggestion-item {
            padding: 15px;
            border-bottom: 1px solid #333;
            cursor: pointer;
        }
        
        .suggestion-item:hover {
            background-color: #333;
        }
        
        .suggestion-title {
            font-weight: bold;
        }
        
        .suggestion-details {
            font-size: 0.8em;
            color: #aaa;
            margin-top: 5px;
        }
    </style>
</head>
<body>
    <div class="container">
        <div class="search-container">
            <div class="logo">NETFLIX</div>
            <div class="search-box">
                <input type="text" id="search-input" placeholder="Search for a title...">
                <div class="suggestions" id="suggestions-container"></div>
            </div>
        </div>
    </div>

    <script>
        document.addEventListener('DOMContentLoaded', function() {
            const searchInput = document.getElementById('search-input');
            const suggestionsContainer = document.getElementById('suggestions-container');
            
            let debounceTimeout;
            
            searchInput.addEventListener('input', function() {
                clearTimeout(debounceTimeout);
                const query = searchInput.value.trim();
                
                // Clear suggestions if query is empty
                if (!query) {
                    suggestionsContainer.innerHTML = '';
                    return;
                }
                
                // Debounce requests to prevent too many API calls
                debounceTimeout = setTimeout(() => {
                    if (query.length >= 1) {
                        fetchSuggestions(query);
                    } else {
                        suggestionsContainer.innerHTML = '';
                    }
                }, 300);
            });
            
            // Also listen for keyboard navigation
            searchInput.addEventListener('keydown', function(e) {
                const items = suggestionsContainer.querySelectorAll('.suggestion-item');
                if (!items.length) return;
                
                // Find currently selected item
                const currentIndex = Array.from(items).findIndex(
                    item => item.classList.contains('selected')
                );
                
                if (e.key === 'ArrowDown') {
                    e.preventDefault();
                    const nextIndex = currentIndex < 0 ? 0 : (currentIndex + 1) % items.length;
                    selectItem(items, nextIndex);
                } else if (e.key === 'ArrowUp') {
                    e.preventDefault();
                    const prevIndex = currentIndex < 0 ? items.length - 1 : (currentIndex - 1 + items.length) % items.length;
                    selectItem(items, prevIndex);
                } else if (e.key === 'Enter') {
                    const selectedItem = suggestionsContainer.querySelector('.selected');
                    if (selectedItem) {
                        e.preventDefault();
                        selectedItem.click();
                    }
                }
            });
            
            function selectItem(items, index) {
                // Remove selection from all items
                items.forEach(item => item.classList.remove('selected'));
                // Add selection to current item
                items[index].classList.add('selected');
                items[index].scrollIntoView({ block: 'nearest' });
            }
            
            function fetchSuggestions(query) {
                console.log('Fetching suggestions for:', query);
                
                fetch(`/api/suggest?q=${encodeURIComponent(query)}`)
                    .then(response => {
                        if (!response.ok) {
                            throw new Error(`HTTP error! Status: ${response.status}`);
                        }
                        return response.json();
                    })
                    .then(data => {
                        console.log('Received suggestions:', data.length);
                        displaySuggestions(data);
                    })
                    .catch(error => {
                        console.error('Error fetching suggestions:', error);
                        suggestionsContainer.innerHTML = '<div class="suggestion-item">Error loading suggestions</div>';
                    });
            }
            
            function displaySuggestions(suggestions) {
                suggestionsContainer.innerHTML = '';
                
                if (suggestions.length === 0) {
                    suggestionsContainer.innerHTML = '<div class="suggestion-item">No results found</div>';
                    return;
                }
                
                suggestions.forEach((suggestion, index) => {
                    const item = document.createElement('div');
                    item.className = 'suggestion-item';
                    if (index === 0) item.classList.add('selected');
                    
                    let itemHtml = `<div class="suggestion-title">${suggestion.title}</div>`;
                    
                    if (suggestion.year || suggestion.type || suggestion.rating) {
                        itemHtml += `<div class="suggestion-details">`;
                        if (suggestion.year) itemHtml += `${suggestion.year} `;
                        if (suggestion.type) itemHtml += `${suggestion.type} `;
                        if (suggestion.rating) itemHtml += `${suggestion.rating}`;
                        itemHtml += `</div>`;
                    }
                    
                    if (suggestion.description) {
                        itemHtml += `<div class="suggestion-details">${suggestion.description}</div>`;
                    }
                    
                    item.innerHTML = itemHtml;
                    
                    item.addEventListener('click', function() {
                        searchInput.value = suggestion.title;
                        suggestionsContainer.innerHTML = '';
                        console.log(`Selected: ${suggestion.title}`);
                    });
                    
                    suggestionsContainer.appendChild(item);
                });
            }
        });
    </script>
</body>
</html>"""

# Write the index.html file with explicit UTF-8 encoding
with open('templates/index.html', 'w', encoding='utf-8') as f:
    f.write(index_html)

@app.route('/')
def index():
    # Debug the template loading
    try:
        return render_template('index.html')
    except Exception as e:
        print(f"Template error: {e}")
        # Return a simple HTML as fallback
        return """
        <!DOCTYPE html>
        <html>
        <head>
            <meta charset="UTF-8">
            <title>Netflix Search</title>
        </head>
        <body style="background-color: #141414; color: white; font-family: Arial;">
            <h1 style="color: #e50914;">NETFLIX</h1>
            <input type="text" id="search" placeholder="Search..." style="padding: 10px; width: 300px;">
            <div id="results"></div>
            
            <script>
                document.getElementById('search').addEventListener('input', function() {
                    const query = this.value;
                    if (query.length > 1) {
                        fetch('/api/suggest?q=' + query)
                        .then(response => response.json())
                        .then(data => {
                            const results = document.getElementById('results');
                            results.innerHTML = '';
                            data.forEach(item => {
                                results.innerHTML += '<div>' + item.title + '</div>';
                            });
                        });
                    }
                });
            </script>
        </body>
        </html>
        """

@app.route('/api/suggest', methods=['GET'])
def suggest():
    query = request.args.get('q', '')
    suggestions = generate_suggestions(query)
    
    # Enrich suggestions with metadata if available
    enriched_suggestions = []
    for title in suggestions:
        metadata = get_title_metadata(title)
        if metadata:
            # Convert all values to Python native types to ensure JSON serialization
            serializable_metadata = {}
            for key, value in metadata.items():
                # Convert numpy types to native Python types
                if hasattr(value, 'item'):  # Check if it's a numpy type with .item() method
                    serializable_metadata[key] = value.item()
                elif isinstance(value, pd.Series):
                    serializable_metadata[key] = value.to_list()
                else:
                    serializable_metadata[key] = value
            enriched_suggestions.append(serializable_metadata)
        else:
            enriched_suggestions.append({'title': title})
    
    return jsonify(enriched_suggestions)

# Also modify the get_title_metadata function to handle serialization better
def get_title_metadata(title):
    if netflix_df is None:
        return None
    
    try:
        # Find the closest matching title
        clean_title = re.sub(r'[^\w\s]', '', title.lower())
        matches = netflix_df[netflix_df['title'].str.lower().str.contains(clean_title, na=False)]
        
        if len(matches) > 0:
            row = matches.iloc[0]
            return {
                'title': str(row['title']),
                'type': str(row.get('type', '')),
                'year': int(row.get('release_year', 0)) if pd.notnull(row.get('release_year', '')) else '',
                'rating': str(row.get('rating', '')),
                'description': str(row.get('description', ''))[:100] + '...' if len(str(row.get('description', ''))) > 100 else str(row.get('description', ''))
            }
    except Exception as e:
        print(f"Error getting metadata for '{title}': {e}")
    return None

if __name__ == '__main__':
    print("Netflix Autocomplete backend started!")
    print("Visit http://localhost:5000/ in your browser")
    app.run(debug=True)