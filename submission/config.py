# Configuration parameters for Intelligent Candidate Ranking System

EMBEDDING_MODEL = "all-MiniLM-L6-v2"
L1_POOL_SIZE = 2000
NUM_WORKERS = 8

# Core weights for final base score (sum = 1.0)
WEIGHTS = {
    "skills": 0.30,
    "experience": 0.25,
    "title": 0.15,
    "education": 0.10
}

# Behavioral sub-weights (sum = 1.0)
BEHAVIORAL_WEIGHTS = {
    "engagement": 0.35,
    "recency": 0.30,
    "availability": 0.15,
    "conversion": 0.15,
    "market": 0.05
}

# Pre-parsed defaults for founding AI engineer job description
DEFAULT_JD_REQUIREMENTS = {
    "required_skills": [
        ("Python", 1.0),
        ("Machine Learning", 1.0),
        ("embeddings-based retrieval", 1.0),
        ("vector databases", 1.0),
        ("search infrastructure", 1.0),
        ("evaluation frameworks", 1.0),
        ("Natural Language Processing", 0.9),
        ("Information Retrieval", 0.8),
        ("Data Analysis", 0.8),
        ("PyTorch", 0.8),
        ("TensorFlow", 0.8)
    ],
    "experience": {
        "min_years": 4.0,
        "target_years": 6.0,
        "ideal_years": 8.0
    },
    "preferred_titles": [
        "Senior AI Engineer",
        "Founding AI Engineer",
        "Senior Machine Learning Engineer",
        "ML Engineer",
        "Senior Data Scientist",
        "Data Scientist",
        "AI Engineer",
        "Staff AI Engineer"
    ],
    "required_education": ["B.Tech", "B.E.", "B.S.", "M.Tech", "M.S.", "Ph.D"],
    "preferred_industries": ["Technology", "Software", "Internet", "AI", "Machine Learning"]
}

# Skill alias mappings for standardizing exact skill checks
SKILL_ALIASES = {
    "ml": "machine learning",
    "deep learning": "machine learning",
    "neural networks": "machine learning",
    "nlp": "natural language processing",
    "sentence transformers": "embeddings-based retrieval",
    "sentence-transformers": "embeddings-based retrieval",
    "vector database": "vector databases",
    "vector search": "vector databases",
    "pinecone": "vector databases",
    "qdrant": "vector databases",
    "weaviate": "vector databases",
    "milvus": "vector databases",
    "faiss": "vector databases",
    "elasticsearch": "search infrastructure",
    "opensearch": "search infrastructure",
    "hybrid search": "search infrastructure",
    "ndcg": "evaluation frameworks",
    "mrr": "evaluation frameworks",
    "map": "evaluation frameworks",
    "eval frameworks": "evaluation frameworks"
}

# Consulting firms that are explicitly disqualified
DISQUALIFIED_CONSULTING_FIRMS = [
    "tcs", "tata consultancy services", "infosys", "wipro", "accenture", 
    "cognizant", "capgemini", "wipro limited", "tcs e-serve", "infosys technologies"
]

# Preferred locations in India
PREFERRED_LOCATIONS = ["pune", "noida", "hyderabad", "mumbai", "delhi", "bangalore"]
