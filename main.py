import os
import pickle
import pandas as pd
import numpy as np
import httpx
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from dotenv import load_dotenv

load_dotenv()
TMDB_API_KEY = os.getenv('TMDB_API_KEY')

TMDB_BASE_URL = "https://api.themoviedb.org/3"
TMDB_IMG_500 = "https://image.tmdb.org/t/p/w500" 
if not TMDB_API_KEY:
    raise ValueError("TMDB_API_KEY not found in environment variables")
app = FastAPI(title="Movie Recommendation API" version="1.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

DF_PATH = os.path.join(BASE_DIR, 'df.pkl')
INDECES_PATH = os.path.join(BASE_DIR, 'indeces.pkl')
TFID_MATRIX_PATH = os.path.join(BASE_DIR, 'tfid_matrix.pkl')
TFID_PATH = os.path.join(BASE_DIR, 'tfid.pkl')

df = pd.read_pickle(DF_PATH)
indeces = pickle.load(open(INDECES_PATH, 'rb'))
tfidf_matrix = pickle.load(open(TFID_MATRIX_PATH, 'rb'))
tfid = pickle.load(open(TFID_PATH, 'rb'))
class TMDBMovie(BaseModel):
    tmdb_id: int
    title: str
    poster_url: Optional[str] = None
    release_date: Optional[str] = None
    vote_average: Optional[float] = None
class TMDBMovieDetails(BaseModel):
    tmdb_id: int
    title: str
    overview: Optional[str] = None 
    release_date: Optional[str] = None
    poster_path: Optional[str] = None
    backdrop_path: Optional[str] = None
    genres: Optional[List[str]] = None

class TFIDFRecItem(BaseModel):
    title: str
    score: float
    tmdb: Optional[TMDBMovie] = None

class