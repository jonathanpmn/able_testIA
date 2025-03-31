from pydantic import BaseModel
from typing import List, Optional

class Article(BaseModel):
    canonical_url: Optional[str] = ""
    published_time: Optional[str] = ""
    title: Optional[str] = ""
    
class Result(BaseModel):
    articles: List[Article]

class APIData(BaseModel):
    result: Result
