import ast
import os
import pickle
from typing import Optional, List, Dict, Any, Tuple

import numpy as np
import pandas as pd
import httpx
from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from dotenv import load_dotenv


# =========================
# ENV
# =========================
load_dotenv()
TMDB_API_KEY = os.getenv("TMDB_API_KEY")

TMDB_BASE = "https://api.themoviedb.org/3"
TMDB_IMG_500 = "https://image.tmdb.org/t/p/w500"

if not TMDB_API_KEY:
    print("[INFO] TMDB_API_KEY is not configured. Running in offline/local dataset mode.")


# =========================
# FASTAPI APP
# =========================
app = FastAPI(title="Movie Recommender API", version="3.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# =========================
# PICKLE GLOBALS
# =========================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

DF_PATH = os.path.join(BASE_DIR, "df.pkl")
INDICES_PATH = os.path.join(BASE_DIR, "indices.pkl")
TFIDF_MATRIX_PATH = os.path.join(BASE_DIR, "tfidf_matrix.pkl")
TFIDF_PATH = os.path.join(BASE_DIR, "tfidf.pkl")

df: Optional[pd.DataFrame] = None
indices_obj: Any = None
tfidf_matrix: Any = None
tfidf_obj: Any = None

TITLE_TO_IDX: Optional[Dict[str, int]] = None


# =========================
# MODELS
# =========================
class TMDBMovieCard(BaseModel):
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
    poster_url: Optional[str] = None
    backdrop_url: Optional[str] = None
    genres: List[dict] = []


class TFIDFRecItem(BaseModel):
    title: str
    score: float
    tmdb: Optional[TMDBMovieCard] = None


class SearchBundleResponse(BaseModel):
    query: str
    movie_details: TMDBMovieDetails
    tfidf_recommendations: List[TFIDFRecItem]
    genre_recommendations: List[TMDBMovieCard]


# =========================
# UTILS
# =========================
def _norm_title(t: str) -> str:
    return str(t).strip().lower()


def make_img_url(path: Optional[str]) -> Optional[str]:
    if not path or pd.isna(path):
        return None
    path_str = str(path).strip()
    if path_str.startswith("http://") or path_str.startswith("https://"):
        return path_str
    if not path_str.startswith("/"):
        path_str = "/" + path_str
    return f"{TMDB_IMG_500}{path_str}"


def safe_int_id(val: Any, fallback: int = 0) -> int:
    try:
        return int(val)
    except Exception:
        return fallback


def safe_parse_genres(val: Any) -> List[dict]:
    if isinstance(val, list):
        return [g if isinstance(g, dict) else {"id": i + 1, "name": str(g)} for i, g in enumerate(val)]
    if isinstance(val, str) and val.strip():
        val_str = val.strip()
        if val_str.startswith("[") and val_str.endswith("]"):
            try:
                parsed = ast.literal_eval(val_str)
                if isinstance(parsed, list):
                    return [g if isinstance(g, dict) else {"id": i + 1, "name": str(g)} for i, g in enumerate(parsed)]
            except Exception:
                pass
        # Space or comma separated genres (e.g. "Animation Comedy Family")
        words = [w.strip() for w in val_str.replace(",", " ").split() if w.strip()]
        return [{"id": i + 1, "name": w.capitalize()} for i, w in enumerate(words)]
    return []


async def tmdb_get(path: str, params: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    """
    Safe TMDB GET:
    - Quick 2.5s connect timeout to ensure no UI freeze if TMDB is blocked/slow
    - Returns None on error/timeout so callers fall back to local dataset
    """
    if not TMDB_API_KEY or len(TMDB_API_KEY.strip()) < 10:
        return None

    q = dict(params)
    q["api_key"] = TMDB_API_KEY

    try:
        timeout = httpx.Timeout(3.0, connect=2.0)
        async with httpx.AsyncClient(timeout=timeout) as client:
            r = await client.get(f"{TMDB_BASE}{path}", params=q)
            if r.status_code == 200:
                return r.json()
    except Exception:
        pass
    return None


async def tmdb_cards_from_results(
    results: List[dict], limit: int = 20
) -> List[TMDBMovieCard]:
    out: List[TMDBMovieCard] = []
    for m in (results or [])[:limit]:
        out.append(
            TMDBMovieCard(
                tmdb_id=safe_int_id(m.get("id"), fallback=0),
                title=m.get("title") or m.get("name") or "",
                poster_url=make_img_url(m.get("poster_path")),
                release_date=m.get("release_date"),
                vote_average=m.get("vote_average"),
            )
        )
    return out


async def tmdb_movie_details(movie_id: int) -> Optional[TMDBMovieDetails]:
    data = await tmdb_get(f"/movie/{movie_id}", {"language": "en-US"})
    if not data:
        return None
    return TMDBMovieDetails(
        tmdb_id=safe_int_id(data.get("id"), fallback=movie_id),
        title=data.get("title") or "",
        overview=data.get("overview"),
        release_date=data.get("release_date"),
        poster_url=make_img_url(data.get("poster_path")),
        backdrop_url=make_img_url(data.get("backdrop_path")),
        genres=data.get("genres", []) or [],
    )


async def tmdb_search_movies(query: str, page: int = 1) -> Optional[Dict[str, Any]]:
    return await tmdb_get(
        "/search/movie",
        {
            "query": query,
            "include_adult": "false",
            "language": "en-US",
            "page": page,
        },
    )


async def tmdb_search_first(query: str) -> Optional[dict]:
    data = await tmdb_search_movies(query=query, page=1)
    if data:
        results = data.get("results", [])
        return results[0] if results else None
    return None


# =========================
# LOCAL DATASET FALLBACKS
# =========================
def local_search_movies(query: str, page: int = 1, page_size: int = 24) -> Dict[str, Any]:
    global df
    if df is None:
        return {"page": page, "results": [], "total_pages": 1, "total_results": 0}

    q = _norm_title(query)
    matches = df[df["title"].astype(str).str.lower().str.contains(q, regex=False, na=False)]

    if "popularity" in matches.columns:
        matches = matches.sort_values(by="popularity", ascending=False)

    start = (page - 1) * page_size
    paged = matches.iloc[start : start + page_size]

    results = []
    for idx, row in paged.iterrows():
        p_path = row.get("poster_path") if "poster_path" in row else None
        poster_path = str(p_path) if pd.notna(p_path) and p_path else None
        results.append(
            {
                "id": int(idx),
                "title": str(row.get("title", "")),
                "poster_path": poster_path,
                "release_date": str(row.get("release_date", "")) if pd.notna(row.get("release_date")) else "",
                "vote_average": float(row.get("vote_average", 0.0)) if pd.notna(row.get("vote_average")) else 0.0,
                "overview": str(row.get("overview", "")) if pd.notna(row.get("overview")) else "",
            }
        )

    total_pages = (len(matches) + page_size - 1) // page_size if len(matches) > 0 else 1
    return {
        "page": page,
        "results": results,
        "total_pages": max(1, total_pages),
        "total_results": len(matches),
    }


def local_home_cards(category: str = "popular", limit: int = 24) -> List[TMDBMovieCard]:
    global df
    if df is None:
        return []

    pool = df.copy()
    if "poster_path" in pool.columns:
        with_posters = pool[pool["poster_path"].notna() & (pool["poster_path"].astype(str).str.strip() != "")]
        if len(with_posters) >= limit:
            pool = with_posters

    if category == "top_rated":
        sorted_df = pool.sort_values(by="vote_average", ascending=False) if "vote_average" in pool.columns else pool
    elif category in ("upcoming", "now_playing"):
        sorted_df = pool.sort_values(by="release_date", ascending=False) if "release_date" in pool.columns else pool
    else:  # popular, trending
        sorted_df = pool.sort_values(by="popularity", ascending=False) if "popularity" in pool.columns else pool

    out: List[TMDBMovieCard] = []
    for idx, row in sorted_df.head(limit).iterrows():
        p_path = row.get("poster_path") if "poster_path" in row else None
        poster_url = make_img_url(str(p_path)) if pd.notna(p_path) and p_path else None
        out.append(
            TMDBMovieCard(
                tmdb_id=int(idx),
                title=str(row.get("title", "")),
                poster_url=poster_url,
                release_date=str(row.get("release_date", "")) if pd.notna(row.get("release_date")) else "",
                vote_average=float(row.get("vote_average", 0.0)) if pd.notna(row.get("vote_average")) else 0.0,
            )
        )
    return out


def local_movie_details(movie_id: int) -> Optional[TMDBMovieDetails]:
    global df
    if df is None:
        return None

    if 0 <= movie_id < len(df):
        row = df.iloc[movie_id]
        idx = movie_id
    else:
        # Fallback: search by id column if present or title
        if "id" in df.columns:
            match = df[df["id"].astype(str) == str(movie_id)]
            if not match.empty:
                row = match.iloc[0]
                idx = int(match.index[0])
            else:
                return None
        else:
            return None

    p_path = row.get("poster_path") if "poster_path" in row else None
    b_path = row.get("backdrop_path") if "backdrop_path" in row else p_path
    poster_url = make_img_url(str(p_path)) if pd.notna(p_path) and p_path else None
    backdrop_url = make_img_url(str(b_path)) if pd.notna(b_path) and b_path else None
    genres = safe_parse_genres(row.get("genres", ""))

    return TMDBMovieDetails(
        tmdb_id=int(idx),
        title=str(row.get("title", "")),
        overview=str(row.get("overview", "")) if pd.notna(row.get("overview")) and str(row.get("overview", "")).strip() else "No overview available.",
        release_date=str(row.get("release_date", "")) if pd.notna(row.get("release_date")) else "-",
        poster_url=poster_url,
        backdrop_url=backdrop_url,
        genres=genres,
    )


def local_genre_recommend(movie_id: int, limit: int = 18) -> List[TMDBMovieCard]:
    global df
    if df is None:
        return []

    details = local_movie_details(movie_id)
    if not details or not details.genres:
        return local_home_cards(category="popular", limit=limit)

    genre_name = details.genres[0].get("name", "").lower()
    if not genre_name:
        return local_home_cards(category="popular", limit=limit)

    matches = df[df["genres"].astype(str).str.lower().str.contains(genre_name, regex=False, na=False)]
    if "popularity" in matches.columns:
        matches = matches.sort_values(by="popularity", ascending=False)

    out: List[TMDBMovieCard] = []
    for idx, row in matches.iterrows():
        mid = int(idx)
        if mid == movie_id:
            continue
        p_path = row.get("poster_path") if "poster_path" in row else None
        poster_url = make_img_url(str(p_path)) if pd.notna(p_path) and p_path else None
        out.append(
            TMDBMovieCard(
                tmdb_id=mid,
                title=str(row.get("title", "")),
                poster_url=poster_url,
                release_date=str(row.get("release_date", "")) if pd.notna(row.get("release_date")) else "",
                vote_average=float(row.get("vote_average", 0.0)) if pd.notna(row.get("vote_average")) else 0.0,
            )
        )
        if len(out) >= limit:
            break
    return out


def local_card_by_title(title: str) -> Optional[TMDBMovieCard]:
    global df
    if df is None:
        return None

    match = df[df["title"].astype(str).str.lower() == title.strip().lower()]
    if match.empty:
        match = df[df["title"].astype(str).str.lower().str.contains(title.strip().lower(), regex=False, na=False)]
    if match.empty:
        return None

    idx = int(match.index[0])
    row = match.iloc[0]
    p_path = row.get("poster_path") if "poster_path" in row else None
    poster_url = make_img_url(str(p_path)) if pd.notna(p_path) and p_path else None
    return TMDBMovieCard(
        tmdb_id=idx,
        title=str(row.get("title", "")),
        poster_url=poster_url,
        release_date=str(row.get("release_date", "")) if pd.notna(row.get("release_date")) else "",
        vote_average=float(row.get("vote_average", 0.0)) if pd.notna(row.get("vote_average")) else 0.0,
    )


# =========================
# TF-IDF Helpers
# =========================
def build_title_to_idx_map(indices: Any) -> Dict[str, int]:
    title_to_idx: Dict[str, int] = {}
    if isinstance(indices, dict):
        for k, v in indices.items():
            title_to_idx[_norm_title(k)] = int(v)
        return title_to_idx

    try:
        for k, v in indices.items():
            title_to_idx[_norm_title(k)] = int(v)
        return title_to_idx
    except Exception:
        raise RuntimeError("indices.pkl must be dict or pandas Series-like (with .items())")


def get_local_idx_by_title(title: str) -> int:
    global TITLE_TO_IDX
    if TITLE_TO_IDX is None:
        raise HTTPException(status_code=500, detail="TF-IDF index map not initialized")
    key = _norm_title(title)
    if key in TITLE_TO_IDX:
        return int(TITLE_TO_IDX[key])
    raise HTTPException(status_code=404, detail=f"Title not found in local dataset: '{title}'")


def tfidf_recommend_titles(
    query_title: str, top_n: int = 10
) -> List[Tuple[str, float]]:
    global df, tfidf_matrix
    if df is None or tfidf_matrix is None:
        raise HTTPException(status_code=500, detail="TF-IDF resources not loaded")

    idx = get_local_idx_by_title(query_title)

    qv = tfidf_matrix[idx]
    scores = (tfidf_matrix @ qv.T).toarray().ravel()
    order = np.argsort(-scores)

    out: List[Tuple[str, float]] = []
    for i in order:
        if int(i) == int(idx):
            continue
        try:
            title_i = str(df.iloc[int(i)]["title"])
        except Exception:
            continue
        out.append((title_i, float(scores[int(i)])))
        if len(out) >= top_n:
            break
    return out


async def attach_tmdb_card_by_title(title: str) -> Optional[TMDBMovieCard]:
    try:
        m = await tmdb_search_first(title)
        if m:
            return TMDBMovieCard(
                tmdb_id=safe_int_id(m.get("id"), fallback=0),
                title=m.get("title") or title,
                poster_url=make_img_url(m.get("poster_path")),
                release_date=m.get("release_date"),
                vote_average=m.get("vote_average"),
            )
    except Exception:
        pass
    return local_card_by_title(title)


# =========================
# STARTUP: LOAD PICKLES
# =========================
@app.on_event("startup")
def load_pickles():
    global df, indices_obj, tfidf_matrix, tfidf_obj, TITLE_TO_IDX

    with open(DF_PATH, "rb") as f:
        df = pickle.load(f)

    with open(INDICES_PATH, "rb") as f:
        indices_obj = pickle.load(f)

    with open(TFIDF_MATRIX_PATH, "rb") as f:
        tfidf_matrix = pickle.load(f)

    with open(TFIDF_PATH, "rb") as f:
        tfidf_obj = pickle.load(f)

    TITLE_TO_IDX = build_title_to_idx_map(indices_obj)

    if df is None or "title" not in df.columns:
        raise RuntimeError("df.pkl must contain a DataFrame with a 'title' column")

    # Sanitize numeric columns to prevent sorting/comparison type errors
    if "popularity" in df.columns:
        df["popularity"] = pd.to_numeric(df["popularity"], errors="coerce").fillna(0.0)
    if "vote_average" in df.columns:
        df["vote_average"] = pd.to_numeric(df["vote_average"], errors="coerce").fillna(0.0)
    if "vote_count" in df.columns:
        df["vote_count"] = pd.to_numeric(df["vote_count"], errors="coerce").fillna(0.0)


# =========================
# ROUTES
# =========================
@app.get("/")
def root():
    return {
        "message": "🎬 Movie Recommender API is live!",
        "status": "online",
        "total_movies_loaded": len(df) if df is not None else 0,
        "docs_url": "/docs",
        "endpoints": {
            "health": "/health",
            "home": "/home?category=popular&limit=24",
            "search": "/tmdb/search?query=toy story",
            "movie_details": "/movie/id/0",
            "genre_recommend": "/recommend/genre?tmdb_id=0&limit=18",
            "tfidf_recommend": "/recommend/tfidf?title=Toy%20Story&top_n=10",
            "bundle_search": "/movie/search?query=toy story",
        },
        "frontend_url": "http://localhost:8501",
    }


@app.get("/health")
def health():
    return {"status": "ok", "total_movies": len(df) if df is not None else 0}


# ---------- HOME FEED ----------
@app.get("/home", response_model=List[TMDBMovieCard])
async def home(
    category: str = Query("popular"),
    limit: int = Query(24, ge=1, le=50),
):
    try:
        data = None
        if category == "trending":
            data = await tmdb_get("/trending/movie/day", {"language": "en-US"})
        elif category in {"popular", "top_rated", "upcoming", "now_playing"}:
            data = await tmdb_get(f"/movie/{category}", {"language": "en-US", "page": 1})

        if data and "results" in data:
            cards = await tmdb_cards_from_results(data.get("results", []), limit=limit)
            if cards:
                return cards
    except Exception:
        pass
    return local_home_cards(category=category, limit=limit)


# ---------- SEARCH ----------
@app.get("/tmdb/search")
async def tmdb_search(
    query: str = Query(..., min_length=1),
    page: int = Query(1, ge=1, le=10),
):
    try:
        data = await tmdb_search_movies(query=query, page=page)
        if data and "results" in data and len(data["results"]) > 0:
            return data
    except Exception:
        pass
    return local_search_movies(query=query, page=page, page_size=24)


# ---------- MOVIE DETAILS ----------
@app.get("/movie/id/{tmdb_id}", response_model=TMDBMovieDetails)
async def movie_details_route(tmdb_id: int):
    try:
        details = await tmdb_movie_details(tmdb_id)
        if details:
            return details
    except Exception:
        pass

    local_det = local_movie_details(tmdb_id)
    if local_det:
        return local_det
    raise HTTPException(status_code=404, detail=f"Movie ID {tmdb_id} not found")


# ---------- GENRE RECOMMENDATIONS ----------
@app.get("/recommend/genre", response_model=List[TMDBMovieCard])
async def recommend_genre(
    tmdb_id: int = Query(...),
    limit: int = Query(18, ge=1, le=50),
):
    try:
        details = await tmdb_movie_details(tmdb_id)
        if details and details.genres:
            genre_id = details.genres[0]["id"]
            discover = await tmdb_get(
                "/discover/movie",
                {
                    "with_genres": genre_id,
                    "language": "en-US",
                    "sort_by": "popularity.desc",
                    "page": 1,
                },
            )
            if discover and "results" in discover:
                cards = await tmdb_cards_from_results(
                    discover.get("results", []), limit=limit
                )
                res = [c for c in cards if c.tmdb_id != tmdb_id]
                if res:
                    return res
    except Exception:
        pass
    return local_genre_recommend(tmdb_id, limit=limit)


# ---------- TF-IDF ONLY ----------
@app.get("/recommend/tfidf")
async def recommend_tfidf(
    title: str = Query(..., min_length=1),
    top_n: int = Query(10, ge=1, le=50),
):
    recs = tfidf_recommend_titles(title, top_n=top_n)
    return [{"title": t, "score": s} for t, s in recs]


# ---------- BUNDLE: Details + TF-IDF recs + Genre recs ----------
@app.get("/movie/search", response_model=SearchBundleResponse)
async def search_bundle(
    query: str = Query(..., min_length=1),
    tfidf_top_n: int = Query(12, ge=1, le=30),
    genre_limit: int = Query(12, ge=1, le=30),
):
    details: Optional[TMDBMovieDetails] = None
    try:
        best = await tmdb_search_first(query)
        if best:
            details = await tmdb_movie_details(int(best["id"]))
    except Exception:
        pass

    if not details:
        local_res = local_search_movies(query, page=1, page_size=1)
        if local_res["results"]:
            first_id = local_res["results"][0]["id"]
            details = local_movie_details(first_id)

    if not details:
        details = TMDBMovieDetails(
            tmdb_id=0,
            title=query,
            overview="Overview not available.",
            release_date="-",
            genres=[],
        )

    # 1) TF-IDF recommendations
    tfidf_items: List[TFIDFRecItem] = []
    recs: List[Tuple[str, float]] = []
    try:
        recs = tfidf_recommend_titles(details.title, top_n=tfidf_top_n)
    except Exception:
        try:
            recs = tfidf_recommend_titles(query, top_n=tfidf_top_n)
        except Exception:
            recs = []

    for rec_title, score in recs:
        card = await attach_tmdb_card_by_title(rec_title)
        if not card:
            card = local_card_by_title(rec_title)
        tfidf_items.append(TFIDFRecItem(title=rec_title, score=score, tmdb=card))

    # 2) Genre recommendations
    genre_recs: List[TMDBMovieCard] = []
    if details.tmdb_id:
        try:
            genre_recs = await recommend_genre(tmdb_id=details.tmdb_id, limit=genre_limit)
        except Exception:
            pass
    if not genre_recs and details.genres:
        genre_recs = local_genre_recommend(movie_id=details.tmdb_id, limit=genre_limit)

    return SearchBundleResponse(
        query=query,
        movie_details=details,
        tfidf_recommendations=tfidf_items,
        genre_recommendations=genre_recs,
    )