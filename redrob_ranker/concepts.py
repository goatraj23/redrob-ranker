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
"""

# ---------------------------------------------------------------------------
# Concept lexicons.  Phrases are matched case-insensitively against the
# candidate's combined free text (summary + every career-history description +
# skill names).  Weights encode how strongly each concept signals fit.
# ---------------------------------------------------------------------------

# The decisive "this person actually builds the thing we need" signals.
CORE_IR = {
    "weight": 3.0,
    "phrases": [
        "recommendation system", "recommender", "recommendation engine",
        "learning to rank", "learning-to-rank", "ltr",
        "information retrieval", "search relevance", "search ranking",
        "semantic search", "vector search", "nearest neighbor",
        "approximate nearest neighbor", "ann index",
        "retrieval", "ranking system", "ranking model", "re-ranking", "reranking",
        "personalization", "personalisation", "relevance tuning",
        "embedding", "embeddings", "sentence-transformers", "sentence transformers",
        "bm25", "okapi", "faiss", "pinecone", "weaviate", "qdrant", "milvus",
        "elasticsearch", "opensearch", "rag", "retrieval-augmented",
        "candidate generation", "matching system", "two-tower", "dual encoder",
    ],
}

# Strong applied-ML / NLP signals that support the core.
CORE_ML = {
    "weight": 1.6,
    "phrases": [
        "machine learning", "deep learning", "applied ml", "applied machine learning",
        "natural language processing", " nlp ", "nlp.", "(nlp", "nlp,",
        "transformer", "bert", "large language model", " llm", "llms",
        "fine-tune", "fine-tuning", "fine tuned", "lora", "qlora", "peft",
        "xgboost", "lightgbm", "gradient boosting", "pytorch", "tensorflow",
        "model serving", "model inference", "inference optimization",
        "feature engineering", "feature store", "mlops",
        "experimentation", "ab test", "a/b test", "a/b testing",
        "ndcg", "mrr", "mean reciprocal rank", "map@", "offline evaluation",
        "online evaluation", "evaluation framework", "evaluation metric",
    ],
}

# Data-engineering signals — supportive (the role touches data infra) but not
# sufficient on their own.
DATA_ENG = {
    "weight": 0.7,
    "phrases": [
        "data pipeline", "data pipelines", "etl", "elt", "apache spark", "pyspark",
        "spark", "airflow", "kafka", "data warehouse", "snowflake", "dbt",
        "big data", "streaming pipeline", "batch processing",
    ],
}

# Engineering-craft signals (the JD cares about real production code).
ENG_CRAFT = {
    "weight": 0.5,
    "phrases": [
        "production", "deployed to production", "at scale", "real users",
        "low latency", "high throughput", "microservice", "api", "python",
        "system design", "distributed system", "scalability",
    ],
}

# Off-domain expertise the JD explicitly says is a poor fit when it is the
# PRIMARY expertise and there is no NLP/IR alongside it.
OFF_DOMAIN = {
    "weight": 1.0,  # used as a penalty magnitude, see scoring.py
    "phrases": [
        "computer vision", "image classification", "object detection",
        "image segmentation", "opencv", "yolo", "ocr pipeline",
        "speech recognition", "asr", "text to speech", "speech synthesis",
        "robotics", "autonomous vehicle", "slam", "motion planning",
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
        "published", "publication", "peer-reviewed", "neurips", "icml", "acl",
        "cvpr", "research intern", "thesis", "dissertation",
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
# Location knowledge.  JD: Pune/Noida preferred; Hyderabad, Mumbai, Delhi NCR,
# Bangalore welcome; relocation OK; outside India case-by-case, no visa.
# ---------------------------------------------------------------------------

PREFERRED_CITIES = ["pune", "noida"]
WELCOME_CITIES = [
    "hyderabad", "mumbai", "delhi", "gurgaon", "gurugram", "ghaziabad",
    "bangalore", "bengaluru", "new delhi", "navi mumbai", "faridabad",
]
# Other Indian metros — fine if willing to relocate.
OTHER_INDIA_HINT = [
    "chennai", "kolkata", "ahmedabad", "jaipur", "indore", "chandigarh",
    "coimbatore", "kochi", "trivandrum", "bhubaneswar", "vizag", "nagpur",
]

# ---------------------------------------------------------------------------
# Experience band.  Ideal 6-8, acceptable 5-9, flexible beyond with strong
# signals.  Used by scoring.experience_score.
# ---------------------------------------------------------------------------
EXP_IDEAL_LOW, EXP_IDEAL_HIGH = 6.0, 8.0
EXP_OK_LOW, EXP_OK_HIGH = 5.0, 9.0
