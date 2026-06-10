"""
JD knowledge model for the Redrob "Senior AI Engineer — Founding Team" role.

This module encodes what the job description *means*, not just the words it
uses. It is the single place where domain knowledge about the role lives, so
that scoring stays auditable and easy to defend in the Stage-5 interview.

Everything here is derived directly from job_description.docx:
  - what the role genuinely needs (retrieval / ranking / recsys / NLP at a
    PRODUCT company, 5-9 yrs, strong Python, eval frameworks),
  - the explicit disqualifiers (pure research, consulting-only career,
    CV/speech/robotics-only, recent-LangChain-only, title-chasers, no recent
    code), and
  - the behavioural reality (a perfect-on-paper candidate who is inactive or
    unresponsive is, for hiring, not available).

NOTE on matching: phrases are matched as substrings against the candidate's
combined free text (which is space-joined and space-padded in features.py).
Short tokens that would otherwise match *inside* common words are written with a
LEADING SPACE so they only match at a word start, e.g. " rag" matches "rag" but
not "leve[rag]e"/"sto[rag]e"; " ltr" not "fi[ltr]ation"/"u[ltr]a"; " acl" not
"or[acl]e". Reasoning strips the padding before citing.
"""

# ---------------------------------------------------------------------------
# Concept lexicons.
# ---------------------------------------------------------------------------

# The decisive "this person actually builds the thing we need" signals.
CORE_IR = {
    "weight": 3.0,
    "phrases": [
        "recommendation system", "recommendation systems", "recommender",
        "recommender system", "recommender systems", "recommendation engine",
        "recommendation model",
        "learning to rank", "learning-to-rank", " ltr",
        "information retrieval", "search relevance", "search ranking",
        "relevance ranking", "relevance model", "relevance tuning",
        "semantic search", "vector search", "similarity search",
        "dense retrieval", "semantic retrieval",
        "nearest neighbor", "nearest neighbour", "k-nearest neighbor",
        "approximate nearest neighbor", "ann index", "hnsw",
        "retrieval", "retrieval pipeline", "candidate retrieval",
        "candidate generation", "matching system", "search engine",
        "ranking system", "ranking model", "ranking pipeline", "neural ranking",
        "re-ranking", "reranking", "cross-encoder", "bi-encoder",
        "two-tower", "dual encoder", "colbert",
        "personalization", "personalisation", "personalized recommendation",
        "personalised recommendation", "query understanding", "query expansion",
        "collaborative filtering", "content-based filtering",
        "content based filtering", "matrix factorization", "matrix factorisation",
        "feed ranking", "ads ranking", "ctr prediction", "click-through rate",
        "session-based recommendation", "sequential recommendation",
        "embedding", "embeddings", "sentence-transformers", "sentence transformers",
        "vector database", "vector databases", "bm25", "okapi",
        "faiss", "pinecone", "weaviate", "qdrant", "milvus", "vespa", "lucene",
        "elasticsearch", "opensearch", " rag", "rag pipeline", "retrieval-augmented",
    ],
}

# Strong applied-ML / NLP signals that support the core.
CORE_ML = {
    "weight": 1.6,
    "phrases": [
        "machine learning", "deep learning", "applied ml", "applied machine learning",
        "natural language processing", " nlp ", "nlp.", "(nlp", "nlp,", " nlp",
        "transformer", "transformers", " bert", "roberta", "distilbert",
        "large language model", "language model", " llm", "llms",
        "gpt-3", "gpt-4", "gpt model",
        "fine-tune", "fine-tuning", "fine tuned", " lora", "qlora", "peft",
        "xgboost", "lightgbm", "gradient boosting", "pytorch", "tensorflow", "keras",
        "scikit-learn", "scikit learn", "sklearn", "onnx", "vllm",
        "hugging face", "huggingface", "word2vec", "fasttext", "word embeddings",
        "sentence embedding", "sentence embeddings", "semantic similarity",
        "text classification", "named entity recognition", "sentiment analysis",
        "question answering", "text summarization", "topic modeling",
        "model serving", "model inference", "model deployment", "model registry",
        "inference optimization", "feature engineering", "feature pipeline",
        "feature store", "mlops",
        "experimentation", "ab test", "a/b test", "a/b testing",
        "ndcg", "mrr", "mean reciprocal rank", "map@", "ranking loss",
        "offline evaluation", "online evaluation", "evaluation framework",
        "evaluation metric", "hyperparameter tuning", "cross-validation",
    ],
}

# Data-engineering signals — supportive (the role touches data infra) but not
# sufficient on their own.
DATA_ENG = {
    "weight": 0.7,
    "phrases": [
        "data pipeline", "data pipelines", "etl", "elt", "apache spark", "pyspark",
        "spark", "airflow", "kafka", "flink", "apache beam", "databricks",
        "data warehouse", "snowflake", "dbt", "bigquery", "redshift",
        "trino", "presto", "hive", "data lake",
        "big data", "streaming pipeline", "batch processing",
    ],
}

# Engineering-craft signals (the JD cares about real production code).
ENG_CRAFT = {
    "weight": 0.5,
    "phrases": [
        "production", "deployed to production", "at scale", "real users",
        "low latency", "high throughput", "microservice", " api", "rest api",
        "python", "system design", "distributed system", "scalability",
        "high availability",
    ],
}

# Off-domain expertise the JD explicitly says is a poor fit when it is the
# PRIMARY expertise and there is no NLP/IR alongside it.
OFF_DOMAIN = {
    "weight": 1.0,  # used as a penalty magnitude, see scoring.py
    "phrases": [
        "computer vision", "image classification", "object detection",
        "image segmentation", "opencv", "yolo", "ocr pipeline", "optical character",
        "speech recognition", " asr", "text to speech", "speech synthesis",
        "robotics", "autonomous vehicle", "autonomous driving", " slam",
        "motion planning", "point cloud", "lidar", "pose estimation",
        "embedded systems", "firmware", "control systems",
    ],
}

# "Framework enthusiast" tells — fine, but not what the role needs, and a red
# flag when they are the ONLY AI signal and everything is < 12 months old.
FRAMEWORK_FLUFF = {
    "phrases": [
        "langchain", "llama-index", "llamaindex", "llama index",
        "autogpt", "babyagi", "crewai",
    ],
}

# Pure-research tells (academic / research-only without production).
RESEARCH_ONLY = {
    "phrases": [
        "phd", "postdoc", "post-doctoral", "research scholar", "research fellow",
        "published", "publication", "peer-reviewed", "neurips", "icml", " acl",
        "emnlp", "cvpr", "research intern", "thesis", "dissertation",
    ],
}

# ---------------------------------------------------------------------------
# Title role-classes.  current_title (and career titles) set a strong prior on
# fit.  A keyword-stuffed "Marketing Manager" cannot out-rank a real engineer.
# ---------------------------------------------------------------------------

# Ideal: applied ML / AI / search / data-science engineering roles.
TITLE_TIER_A = [
    "machine learning engineer", "ml engineer", "applied scientist",
    "applied ml", "ai engineer", "artificial intelligence engineer",
    "nlp engineer", "research engineer", "search engineer",
    "data scientist", "ml scientist", "recommendation", "relevance engineer",
    "staff machine learning", "senior machine learning",
    "deep learning engineer", "personalization engineer", "ranking engineer",
]

# Adjacent engineering — can be promoted to a strong fit by domain text.
TITLE_TIER_B = [
    "software engineer", "backend engineer", "back-end engineer",
    "full stack developer", "full-stack developer", "fullstack",
    "data engineer", "analytics engineer", "platform engineer",
    "software developer", "sde", "member of technical staff",
]

# Generic / off-core engineering.
TITLE_TIER_C = [
    "cloud engineer", "devops engineer", "site reliability", "sre",
    "frontend engineer", "front-end engineer", "mobile developer",
    "android developer", "ios developer", "qa engineer", "test engineer",
    "java developer", ".net developer", "php developer", "web developer",
]

# Non-technical roles — the keyword-stuffer trap population.  These get a hard
# low role prior regardless of how many AI skills are listed.
TITLE_TIER_D = [
    "business analyst", "hr manager", "human resources", "recruiter",
    "accountant", "finance", "mechanical engineer", "civil engineer",
    "electrical engineer", "project manager", "program manager",
    "product manager", "customer support", "customer success",
    "operations manager", "content writer", "copywriter", "sales executive",
    "sales manager", "graphic designer", "ui/ux designer", "ux designer",
    "marketing manager", "digital marketing", "seo", "teacher", "professor",
    "consultant",  # generic, often services
]

# ---------------------------------------------------------------------------
# Company knowledge.  The JD penalises whole-career services/consulting and
# rewards product companies.
# ---------------------------------------------------------------------------

CONSULTING_FIRMS = [
    "tcs", "tata consultancy", "infosys", "wipro", "accenture", "cognizant",
    "capgemini", "tech mahindra", "hcl", "mindtree", "ltimindtree", "lti",
    "mphasis", "deloitte", "ibm services", "dxc", "hexaware", "birlasoft",
    "l&t infotech", "persistent systems", "zensar", "coforge", "nttdata",
    "ntt data", "virtusa", "cybage", "cgi", "atos",
]

# Industries that read as services rather than product.
SERVICES_INDUSTRIES = ["it services", "consulting", "outsourcing", "staffing", "bpo"]

# ---------------------------------------------------------------------------
# Company founding years.  The honeypot brief's canonical example is "8 years
# of experience at a company founded 3 years ago" — catching it requires the
# same world knowledge a careful recruiter has.  Only prominent, unambiguous
# names are listed, and matching is exact on the normalised company name (see
# honeypots._founded_year), so a real candidate can't be hit by accident.
# ---------------------------------------------------------------------------

COMPANY_FOUNDED = {
    # Indian AI startups (the dataset's seeded trap companies are here)
    "krutrim": 2023, "krutrim ai": 2023, "ola krutrim": 2023,
    "sarvam": 2023, "sarvam ai": 2023, "sarvam.ai": 2023,
    # global AI companies young enough to be trap material
    "openai": 2015, "anthropic": 2021, "mistral": 2023, "mistral ai": 2023,
    "xai": 2023, "perplexity": 2022, "perplexity ai": 2022,
    "hugging face": 2016, "huggingface": 2016, "cohere": 2019,
    "pinecone": 2019, "qdrant": 2021, "weaviate": 2019, "zilliz": 2017,
    "langchain": 2022, "llamaindex": 2023, "together ai": 2022, "groq": 2016,
    # young Indian consumer/fintech often name-dropped on profiles
    "zepto": 2021, "cred": 2018, "jar": 2021, "bharatpe": 2018, "slice": 2016,
}

# ---------------------------------------------------------------------------
# Location knowledge.  JD: Pune/Noida preferred; "Candidates in Hyderabad,
# Pune, Mumbai, Delhi NCR welcome to apply"; Tier-1 relocators considered;
# outside India case-by-case, no visa.  Bangalore is deliberately NOT in the
# welcome list — the JD enumerates the welcome metros and Bangalore isn't one;
# it is treated as a Tier-1 "other India" city (fine, better if relocating).
# ---------------------------------------------------------------------------

PREFERRED_CITIES = ["pune", "noida"]
WELCOME_CITIES = [
    "hyderabad", "mumbai", "delhi", "gurgaon", "gurugram", "ghaziabad",
    "new delhi", "navi mumbai", "faridabad",
]
# Other Indian metros — fine if willing to relocate.
OTHER_INDIA_HINT = [
    "bangalore", "bengaluru",
    "chennai", "kolkata", "ahmedabad", "jaipur", "indore", "chandigarh",
    "coimbatore", "kochi", "trivandrum", "bhubaneswar", "vizag", "nagpur",
]

# ---------------------------------------------------------------------------
# Experience band.  Ideal 6-8, acceptable 5-9, flexible beyond with strong
# signals.  Used by scoring.experience_score.
# ---------------------------------------------------------------------------
EXP_IDEAL_LOW, EXP_IDEAL_HIGH = 6.0, 8.0
EXP_OK_LOW, EXP_OK_HIGH = 5.0, 9.0
