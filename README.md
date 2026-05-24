````markdown
# 🎬 Movie Recommendation System

A content-based Movie Recommendation System built using **Python**, **Machine Learning**, and **Streamlit/Flask** that recommends similar movies based on user input.

---

## 🚀 Features

- 🔍 Search movies by title
- 🎯 Recommend similar movies instantly
- 📊 Uses TF-IDF Vectorization & Cosine Similarity
- ⚡ Fast recommendation engine using pre-trained pickle files
- 🎥 Movie metadata dataset integration
- 🌐 Simple and interactive UI

---

## 🛠️ Tech Stack

- Python
- Pandas
- Scikit-learn
- NumPy
- Streamlit / Flask
- Pickle

---

## 📂 Project Structure

```bash
MOVIE_RECOMATATION/
│
├── __pycache__/          # Cache files
├── .venv/                # Virtual environment
├── .env                  # Environment variables
│
├── app.py                # Main application file
├── main.py               # Recommendation logic
├── movie.ipynb           # Model training notebook
│
├── movies_metadata(1).csv # Dataset
│
├── df.pkl                # Processed dataframe
├── indices.pkl           # Movie indices
├── tfidf.pkl             # TF-IDF vectorizer
├── tfidf_matrix.pkl      # TF-IDF matrix
│
├── requirements.txt      # Required libraries
├── setup.bat             # Setup script
└── README.md             # Project documentation
````

---

## ⚙️ Installation

### 1️⃣ Clone the Repository

```bash
git clone https://github.com/your-username/movie-recommendation-system.git
```

### 2️⃣ Navigate to Project Folder

```bash
cd movie-recommendation-system
```

### 3️⃣ Create Virtual Environment

```bash
python -m venv .venv
```

### 4️⃣ Activate Virtual Environment

#### Windows

```bash
.venv\Scripts\activate
```

#### Mac/Linux

```bash
source .venv/bin/activate
```

### 5️⃣ Install Dependencies

```bash
pip install -r requirements.txt
```

---

## ▶️ Run the Project

```bash
python app.py
```

OR

```bash
streamlit run app.py
```

---

## 🧠 How It Works

1. Movie metadata is cleaned and processed.
2. TF-IDF Vectorizer converts movie descriptions into numerical vectors.
3. Cosine Similarity calculates similarity between movies.
4. System recommends top similar movies based on user input.

---

## 📸 Sample Recommendation

```python
recommend("Inception")
```

### Output:

* Interstellar
* The Prestige
* Shutter Island
* The Matrix
* Tenet

---

## 📦 Required Libraries

```txt
pandas
numpy
scikit-learn
streamlit
flask
```

---

## 🔑 API Integration (Optional)

You can use:

* OMDb API
* TMDB API

Example:

```bash
http://www.omdbapi.com/?apikey=YOUR_API_KEY
```

---

## 👨‍💻 Author

**Atharva Mapari**

* BBA(CA) Graduate – Modern College Pune
* Currently Pursuing MCA – DES PU Pune

---

## ⭐ Future Improvements

* Add poster images
* Add genre-based filtering
* Deploy on Render / Vercel
* Add collaborative filtering
* Add user authentication

---

## 📜 License
MIT License

Copyright (c) 2026 Atharva Mapari

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
```

