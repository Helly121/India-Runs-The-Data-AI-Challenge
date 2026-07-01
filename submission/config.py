# Configuration parameters for Intelligent Candidate Ranking System

EMBEDDING_MODEL = "all-MiniLM-L6-v2"
L1_POOL_SIZE = 4000
NUM_WORKERS = 8

# Core weights for final base score (sum = 1.0)
WEIGHTS = {
    "skills": 0.30,
    "experience": 0.25,
    "title": 0.15,
    "education": 0.10,
    "summary_alignment": 0.10,
    "career_alignment": 0.10
}

# Behavioral sub-weights (sum = 1.0)
BEHAVIORAL_WEIGHTS = {
    "engagement": 0.35,
    "recency": 0.30,
    "availability": 0.15,
    "conversion": 0.15,
    "market": 0.05
}

# Master lists for dynamic JD parsing
MASTER_SKILLS_LIST = [
    # AI/ML
    "Python", "Machine Learning", "Deep Learning", "Natural Language Processing", "NLP", 
    "Computer Vision", "PyTorch", "TensorFlow", "Keras", "Scikit-Learn", "XGBoost", 
    "Data Science", "Data Analysis", "Artificial Intelligence", "Generative AI", "LLM", 
    "LangChain", "LlamaIndex", "embeddings", "vector databases", "Pinecone", "Qdrant", 
    "Weaviate", "Milvus", "search infrastructure", "Information Retrieval", "Elasticsearch", 
    "evaluation frameworks", "MLOps", "Model Deployment", "Neural Networks",
    
    # Backend/General
    "Java", "C++", "Go", "Golang", "JavaScript", "TypeScript", "Node.js", "React", "Angular",
    "SQL", "NoSQL", "PostgreSQL", "MySQL", "MongoDB", "Redis", "Kafka", "RabbitMQ",
    "AWS", "GCP", "Azure", "Docker", "Kubernetes", "CI/CD", "Git", "Linux", "REST APIs",
    "GraphQL", "Microservices", "System Design", "Cloud Computing", "Spring Boot"
]

MASTER_TITLES_LIST = [
    "AI Engineer", "Machine Learning Engineer", "Data Scientist", "Software Engineer", 
    "Data Engineer", "Backend Engineer", "Frontend Engineer", "Full Stack Engineer", 
    "Product Manager", "DevOps Engineer", "Cloud Engineer", "Research Scientist",
    "NLP Engineer", "Computer Vision Engineer", "Data Analyst", "Analytics Engineer",
    "Founding AI Engineer"
]

MASTER_EDUCATION_LIST = [
    "B.Tech", "B.E.", "B.S.", "M.Tech", "M.S.", "Ph.D", "PhD", "MBA", 
    "Bachelor", "Master", "BCA", "MCA", "B.Sc", "M.Sc"
]

MASTER_LOCATIONS_LIST = [
    "Pune", "Noida", "Hyderabad", "Mumbai", "Delhi", "Bangalore", "Bengaluru", 
    "Gurgaon", "Gurugram", "Chennai", "Kolkata", "Remote", "India"
]

CONSULTING_FIRMS = [
    "tcs", "tata consultancy services", "infosys", "wipro", "accenture", 
    "cognizant", "capgemini", "wipro limited", "tcs e-serve", "infosys technologies",
    "deloitte", "kpmg", "ey", "pwc", "ibm"
]

# The template that will be populated dynamically per JD
DEFAULT_JD_REQUIREMENTS = {
    "required_skills": [("Machine Learning", True)],
    "experience": {
        "min_years": 0.0,
        "target_years": 3.0,
        "ideal_years": 5.0
    },
    "preferred_titles": ["AI Engineer"],
    "required_education": [],
    "preferred_industries": ["Technology", "Software", "Internet"]
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

# The active disqualifications and locations will be populated at runtime
DISQUALIFIED_CONSULTING_FIRMS = []

# Preferred locations in India
PREFERRED_LOCATIONS = []
