from fastapi import FastAPI, HTTPException, Query
from contextlib import asynccontextmanager
from modules.vectorizer import CustomFasttext
import global_variables as gv
import numpy as np
import psycopg2, re
from typing import Optional, List


def load_model():
    """
    Loads the Word2Vec model during the startup of the FastAPI application.
    """
    try:
        vec.load_model(gv.model_path)
        print("Model loaded successfully.")
    except FileNotFoundError as e:
        raise HTTPException(status_code=500, detail=f"Model loading failed: {e}")
    

@asynccontextmanager
async def lifespan(app: FastAPI):
    global vec
    vec = CustomFasttext(gv.fasttext_model)
    load_model()
    yield
    
    
app = FastAPI(lifespan=lifespan)

@app.get("/")
def root():
    """
    Root endpoint to check API status.
    """
    return {"message": "FastAPI is running. Use /embedding to get embeddings."}


def get_embedding(query: str)->np.ndarray:
    """
    Retrieves the embedding for a given word or text.

    Args:
        query (str): The word or text to get the embedding for.

    Returns:
        numpy array: A numpy array containing the embedding.
    """
    if vec.model is None:
        raise HTTPException(status_code=500, detail="Model is not loaded. Load the model first.")

    tokens = query.split()
    vectors = [vec.get_vector(token) for token in tokens]

    if not vectors:
        return np.zeros(vec.model.vector_size)
    
    return np.mean(vectors, axis=0)


@app.get("/similar_rank/")
def similar_rank(skill: str, job_title: Optional[str] = Query(None), seniority: Optional[str] = Query(None)):
    """
    Find the most similar tags to the given query and their relative frequency.

    Args:
        skill (str): The input skill text.
        job_title (Optional[str]): Filter by job title (e.g., "data engineer").
        seniority (Optional[str]): Filter by seniority level (e.g., "junior").

    Returns:
        dict: A JSON response containing tags, frequency, and similarity scores.
    """
    def clean_param(query: str, allowed_queries: List[str]) -> Optional[str]:
        
        query = query.lower()
        allowed_queries = [q.lower() for q in allowed_queries]
        
        allowed_pattern = r'\b(?:' + '|'.join(re.escape(q) for q in allowed_queries) + r')\b'
        clean_query = ' '.join(re.findall(allowed_pattern, query))
        return clean_query if clean_query else None

    job_title_filter = "TRUE"
    seniority_filter = "TRUE"

    if job_title:
        cleaned_job_title = clean_param(job_title, ["Data Engineer", "Data Analyst", "Data Scientist"])
        if cleaned_job_title:
            job_title_filter = f"LOWER(title) = LOWER('{cleaned_job_title}')"

    if seniority:
        cleaned_seniority = clean_param(seniority, ["Junior", "Mid-Level", "Senior"])
        if cleaned_seniority:
            seniority_filter = f"LOWER(seniority) = LOWER('{cleaned_seniority}')"

    sql = f"""
        WITH total_job_cnt AS (
            SELECT 
                COUNT(DISTINCT f.job_id) AS cnt
            FROM 
                warehouse.tag_jobs_fact f
            LEFT JOIN 
                warehouse.seniority s 
            ON 
                s.seniority_id = f.seniority_id
            LEFT JOIN 
                warehouse.titles j 
            ON 
                j.title_id = f.job_title_id
            WHERE 
                {job_title_filter} 
            AND 
                {seniority_filter}
        ),
        most_similar_tags AS (
            SELECT
                t.tag_id,
                t.tag,
                ROW_NUMBER() OVER (
                    ORDER BY t.embedding <=> %s::VECTOR ASC
                ) AS sim_rnk
            FROM 
                warehouse.tags_dim t
        ),
        analysis AS (
            SELECT
                t.tag,
                SUM(occurance) AS frequency,
                100.0 * SUM(occurance) / (SELECT cnt FROM total_job_cnt)::NUMERIC AS frequency_percentage
            FROM 
                most_similar_tags t
            LEFT JOIN
                warehouse.frequency_report fr
            ON
                t.tag_id = fr.tag_id
            WHERE
                t.sim_rnk <= 15  
            AND
                {job_title_filter}
            AND 
                {seniority_filter} 
            GROUP BY 
                t.tag
        )
        SELECT 
            tag, 
            frequency, 
            frequency_percentage
        FROM analysis
    """

    vector = get_embedding(skill)
    vector_str = vector.tolist() 
    try:
        with psycopg2.connect(gv.connection_string) as conn:
            with conn.cursor() as cur:
                cur.execute(sql, (vector_str,))
                rows = cur.fetchall()

        result = [{
            "tag": row[0], 
            "frequency": row[1], 
            "frequency_percentage": row[2]
            } for row in rows
        ]
        return {"results": result, "sql":sql}

    except psycopg2.Error as e:
        raise HTTPException(status_code=500, detail=f"Database query failed: {e}")