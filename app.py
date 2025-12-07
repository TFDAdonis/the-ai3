import streamlit as st
import requests
from pathlib import Path
import concurrent.futures
from datetime import datetime
import re
import json
import xml.etree.ElementTree as ET
from typing import Dict, List, Any
import urllib.parse

MODEL_DIR = Path("models")
MODEL_PATH = MODEL_DIR / "tinyllama-1.1b-chat-v1.0.Q4_K_M.gguf"
MODEL_URL = "https://huggingface.co/tfdtfd/khisbagis23/resolve/main/tinyllama-1.1b-chat-v1.0.Q4_K_M.gguf?download=true"

# Enhanced deep thinking prompts
PRESET_PROMPTS = {
    "Deep Thinker Pro": """You are a sophisticated AI thinker that excels at analysis, synthesis, and providing insightful perspectives. 

THINKING FRAMEWORK:
1. **Comprehension**: Understand the query fully, identify key elements
2. **Contextualization**: Place the topic in historical, cultural, or disciplinary context
3. **Multi-Source Analysis**: Examine information from different sources critically
4. **Pattern Recognition**: Identify connections, contradictions, gaps
5. **Synthesis**: Combine insights into coherent understanding
6. **Critical Evaluation**: Assess reliability, bias, significance
7. **Insight Generation**: Provide original perspectives or connections
8. **Actionable Knowledge**: Suggest applications, further questions, implications

RESPONSE STRUCTURE:
- Start with brief overview
- Present analysis with reasoning
- Reference sources when available
- Highlight interesting connections
- Acknowledge uncertainties
- End with thought-provoking questions or suggestions

TONE: Analytical yet engaging, precise yet accessible.""",

    "Khisba GIS Expert": """You are Khisba GIS - a passionate remote sensing/GIS specialist with deep analytical skills.

SPECIALTY THINKING PROCESS:
1. **Geospatial Context**: How does location/spatial relationships matter?
2. **Temporal Analysis**: What changes over time? Historical patterns?
3. **Data Source Evaluation**: Satellite, ground, or derived data reliability?
4. **Multi-Scale Thinking**: From local to global perspectives
5. **Practical Applications**: Real-world uses of the information
6. **Ethical Considerations**: Privacy, representation, accessibility issues

EXPERTISE: Satellite imagery, vegetation indices, climate analysis, urban planning, disaster monitoring
STYLE: Enthusiastic, precise, eager to explore spatial dimensions of any topic""",

    "Research Analyst": """You are a professional research analyst specializing in synthesizing complex information.

ANALYTICAL APPROACH:
1. **Source Triangulation**: Cross-reference multiple information sources
2. **Credibility Assessment**: Evaluate source reliability, date, bias
3. **Trend Identification**: Spot patterns, changes, anomalies
4. **Comparative Analysis**: Similarities/differences across contexts
5. **Implication Mapping**: Consequences, applications, risks
6. **Knowledge Gaps**: What's missing or needs verification

Always provide structured, evidence-based analysis with clear reasoning.""",

    "Critical Thinker": """You excel at questioning assumptions and examining topics from multiple angles.

CRITICAL THINKING TOOLS:
1. **Assumption Detection**: What unstated beliefs underlie this?
2. **Perspective Switching**: How would different groups view this?
3. **Logical Analysis**: Are arguments valid, evidence sufficient?
4. **Counterfactual Thinking**: What if things were different?
5. **Ethical Reflection**: Moral dimensions, consequences
6. **Practical Reality Check**: Feasibility, implementation issues

Challenge conventional wisdom while remaining constructive.""",

    "Creative Synthesizer": """You connect seemingly unrelated ideas to generate novel insights.

CREATIVE PROCESS:
1. **Divergent Thinking**: Generate multiple possible interpretations
2. **Analogical Reasoning**: What similar patterns exist elsewhere?
3. **Metaphorical Connection**: What metaphors illuminate this?
4. **Interdisciplinary Bridging**: Connect across fields
5. **Future Projection**: How might this evolve or transform?
6. **Alternative Framing**: Different ways to conceptualize

Be imaginative while staying grounded in evidence."""
}

# Enhanced search tools with complete implementations
SEARCH_TOOLS = {
    "Wikipedia": {
        "name": "Wikipedia",
        "icon": "📚",
        "description": "Encyclopedia articles",
        "endpoint": "https://en.wikipedia.org/w/api.php",
        "function": "search_wikipedia"
    },
    "DuckDuckGo": {
        "name": "Web Search",
        "icon": "🌐",
        "description": "Instant answers & web results",
        "endpoint": "https://api.duckduckgo.com/",
        "function": "search_duckduckgo"
    },
    "DuckDuckGo News": {
        "name": "News",
        "icon": "📰",
        "description": "Latest news articles",
        "endpoint": "https://api.duckduckgo.com/",
        "function": "search_duckduckgo_news"
    },
    "Wikidata": {
        "name": "Wikidata",
        "icon": "🗃️",
        "description": "Structured data",
        "endpoint": "https://www.wikidata.org/w/api.php",
        "function": "search_wikidata"
    },
    "ArXiv": {
        "name": "Research Papers",
        "icon": "🔬",
        "description": "Scientific publications",
        "endpoint": "https://export.arxiv.org/api/query",
        "function": "search_arxiv"
    },
    "PubMed": {
        "name": "Medical Research",
        "icon": "🏥",
        "description": "Medical publications",
        "endpoint": "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esearch.fcgi",
        "function": "search_pubmed"
    },
    "OpenLibrary": {
        "name": "Books",
        "icon": "📖",
        "description": "Book information",
        "endpoint": "https://openlibrary.org/search.json",
        "function": "search_openlibrary"
    },
    "Dictionary": {
        "name": "Dictionary",
        "icon": "📚",
        "description": "Word definitions",
        "endpoint": "https://api.dictionaryapi.dev/api/v2/entries/en/",
        "function": "search_dictionary"
    },
    "Countries": {
        "name": "Country Data",
        "icon": "🌍",
        "description": "Country information",
        "endpoint": "https://restcountries.com/v3.1/",
        "function": "search_countries"
    },
    "Quotes": {
        "name": "Quotes",
        "icon": "💭",
        "description": "Famous quotes",
        "endpoint": "https://api.quotable.io/",
        "function": "search_quotes"
    },
    "GitHub": {
        "name": "Code Repos",
        "icon": "💻",
        "description": "GitHub repositories",
        "endpoint": "https://api.github.com/search/repositories",
        "function": "search_github"
    },
    "StackOverflow": {
        "name": "Q&A",
        "icon": "❓",
        "description": "Programming Q&A",
        "endpoint": "https://api.stackexchange.com/2.3/search/advanced",
        "function": "search_stackoverflow"
    },
    "Geocoding": {
        "name": "Location",
        "icon": "📍",
        "description": "Geographic coordinates",
        "endpoint": "https://nominatim.openstreetmap.org/search",
        "function": "search_geocoding"
    },
    "Weather": {
        "name": "Weather",
        "icon": "🌤️",
        "description": "Weather information",
        "endpoint": "https://wttr.in/",
        "function": "search_weather"
    },
    "Air Quality": {
        "name": "Air Quality",
        "icon": "🌫️",
        "description": "Air quality data",
        "endpoint": "https://api.openaq.org/v2/",
        "function": "search_air_quality"
    }
}

st.set_page_config(
    page_title="DeepThink Pro",
    page_icon="🧠",
    layout="wide"
)

# Initialize session state with safe defaults
if "messages" not in st.session_state:
    st.session_state.messages = []

if "model" not in st.session_state:
    st.session_state.model = None

if "system_prompt" not in st.session_state:
    st.session_state.system_prompt = PRESET_PROMPTS["Deep Thinker Pro"]

if "selected_preset" not in st.session_state:
    st.session_state.selected_preset = "Deep Thinker Pro"

# Custom CSS for better UI
st.markdown("""
<style>
    .stChatMessage {
        padding: 1rem;
        border-radius: 10px;
        margin-bottom: 1rem;
    }
    .thinking-bubble {
        background-color: #f0f8ff;
        padding: 1rem;
        border-radius: 10px;
        border-left: 4px solid #4a90e2;
        margin: 1rem 0;
    }
    .analysis-box {
        background-color: #fff8e1;
        padding: 1rem;
        border-radius: 10px;
        border-left: 4px solid #ffb300;
        margin: 1rem 0;
    }
    .source-tag {
        display: inline-block;
        background-color: #e3f2fd;
        color: #1565c0;
        padding: 0.2rem 0.5rem;
        border-radius: 12px;
        font-size: 0.8rem;
        margin: 0.2rem;
    }
    .search-result {
        background-color: #f8f9fa;
        border-left: 3px solid #4285f4;
        padding: 10px;
        margin: 5px 0;
        border-radius: 5px;
    }
</style>
""", unsafe_allow_html=True)

# Download function
def download_model():
    MODEL_DIR.mkdir(exist_ok=True)
    
    if MODEL_PATH.exists():
        return True
    
    st.warning("⚠️ Model not found. Downloading...")
    try:
        response = requests.get(MODEL_URL, stream=True, timeout=30)
        response.raise_for_status()
        
        total_size = int(response.headers.get('content-length', 0))
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        downloaded = 0
        with open(MODEL_PATH, 'wb') as f:
            for chunk in response.iter_content(chunk_size=8192):
                if chunk:
                    f.write(chunk)
                    downloaded += len(chunk)
                    if total_size > 0:
                        progress = downloaded / total_size
                        progress_bar.progress(progress)
                        status_text.text(f"Downloading: {downloaded / (1024**2):.1f} MB")
        
        progress_bar.empty()
        status_text.empty()
        
        if MODEL_PATH.exists():
            file_size = MODEL_PATH.stat().st_size / (1024**3)
            st.success(f"✅ Model downloaded: {file_size:.2f} GB")
            return True
    except Exception as e:
        st.error(f"Download failed: {str(e)}")
    return False

@st.cache_resource(show_spinner=False)
def load_model():
    from ctransformers import AutoModelForCausalLM
    
    if not MODEL_PATH.exists():
        if not download_model():
            raise Exception("Model download failed")
    
    return AutoModelForCausalLM.from_pretrained(
        str(MODEL_DIR),
        model_file=MODEL_PATH.name,
        model_type="llama",
        context_length=4096,
        gpu_layers=0,
        threads=8
    )

# Enhanced search functions with complete implementations

def search_wikipedia(query: str) -> List[Dict]:
    """Enhanced Wikipedia search with better parsing."""
    try:
        params = {
            'action': 'query',
            'format': 'json',
            'list': 'search',
            'srsearch': query,
            'srlimit': 5,
            'srprop': 'size|wordcount|timestamp',
            'utf8': 1
        }
        response = requests.get(SEARCH_TOOLS["Wikipedia"]["endpoint"], params=params, timeout=8)
        response.raise_for_status()
        data = response.json()
        
        results = []
        for item in data.get('query', {}).get('search', []):
            # Get detailed page info
            params2 = {
                'action': 'query',
                'format': 'json',
                'prop': 'extracts|info|categories',
                'inprop': 'url',
                'exintro': 1,
                'explaintext': 1,
                'pageids': item['pageid']
            }
            response2 = requests.get(SEARCH_TOOLS["Wikipedia"]["endpoint"], params=params2, timeout=8)
            if response2.status_code == 200:
                page_data = response2.json()
                pages = page_data.get('query', {}).get('pages', {})
                for page_info in pages.values():
                    extract = page_info.get('extract', '')
                    if extract:
                        # Clean the extract
                        extract = re.sub(r'\n+', ' ', extract)
                        extract = re.sub(r'\s+', ' ', extract)
                        
                        results.append({
                            'title': page_info.get('title', ''),
                            'summary': extract[:500] + ('...' if len(extract) > 500 else ''),
                            'url': page_info.get('fullurl', ''),
                            'categories': [cat['title'] for cat in page_info.get('categories', [])[:3]],
                            'wordcount': page_info.get('wordcount', 0),
                            'source': 'Wikipedia',
                            'timestamp': page_info.get('touched', ''),
                            'relevance': item.get('score', 0)
                        })
        
        return sorted(results, key=lambda x: x['relevance'], reverse=True) if results else []
    except Exception as e:
        return []

def search_duckduckgo(query: str) -> Dict:
    """Enhanced DuckDuckGo search with better parsing."""
    try:
        params = {
            'q': query,
            'format': 'json',
            'no_html': 1,
            'skip_disambig': 1,
            't': 'streamlit_app'
        }
        response = requests.get(SEARCH_TOOLS["DuckDuckGo"]["endpoint"], params=params, timeout=8)
        response.raise_for_status()
        data = response.json()
        
        results = {
            'abstract': data.get('AbstractText', ''),
            'abstract_source': data.get('AbstractSource', ''),
            'abstract_url': data.get('AbstractURL', ''),
            'answer': data.get('Answer', ''),
            'answer_type': data.get('AnswerType', ''),
            'definition': data.get('Definition', ''),
            'definition_source': data.get('DefinitionSource', ''),
            'definition_url': data.get('DefinitionURL', ''),
            'image': data.get('Image', ''),
            'categories': [topic.get('Name', '') for topic in data.get('Categories', [])[:3]],
            'related_topics': [],
            'source': 'DuckDuckGo'
        }
        
        # Process related topics
        for topic in data.get('RelatedTopics', []):
            if 'Text' in topic:
                results['related_topics'].append(topic['Text'])
            elif isinstance(topic, dict) and 'Topics' in topic:
                for subtopic in topic['Topics'][:2]:
                    if 'Text' in subtopic:
                        results['related_topics'].append(subtopic['Text'])
        
        # Clean and filter empty values
        cleaned = {}
        for key, value in results.items():
            if isinstance(value, str) and value.strip():
                cleaned[key] = value.strip()
            elif isinstance(value, list) and value:
                filtered = [v.strip() for v in value if v and v.strip()]
                if filtered:
                    cleaned[key] = filtered
        
        return cleaned if cleaned else {}
    except Exception:
        return {}

def search_duckduckgo_news(query: str) -> List[Dict]:
    """Search DuckDuckGo for news."""
    try:
        params = {
            'q': query + ' news',
            'format': 'json',
            'no_html': 1,
            't': 'streamlit_app'
        }
        response = requests.get(SEARCH_TOOLS["DuckDuckGo"]["endpoint"], params=params, timeout=8)
        response.raise_for_status()
        data = response.json()
        
        news_results = []
        # DuckDuckGo news is in RelatedTopics
        for topic in data.get('RelatedTopics', []):
            if isinstance(topic, dict) and 'FirstURL' in topic and 'Text' in topic:
                # Check if it looks like a news article
                if any(keyword in topic['Text'].lower() for keyword in ['news', 'update', 'report', 'latest']):
                    news_results.append({
                        'title': topic['Text'].split(' - ')[0] if ' - ' in topic['Text'] else topic['Text'],
                        'summary': topic['Text'],
                        'url': topic['FirstURL'],
                        'source': 'DuckDuckGo News'
                    })
        
        return news_results[:5]
    except Exception:
        return []

def search_wikidata(query: str) -> List[Dict]:
    """Search Wikidata for structured data."""
    try:
        # First search for entities
        params = {
            'action': 'wbsearchentities',
            'format': 'json',
            'language': 'en',
            'search': query,
            'limit': 5
        }
        response = requests.get(SEARCH_TOOLS["Wikidata"]["endpoint"], params=params, timeout=10)
        response.raise_for_status()
        data = response.json()
        
        results = []
        for entity in data.get('search', []):
            # Get entity details
            entity_id = entity.get('id')
            if entity_id:
                params2 = {
                    'action': 'wbgetentities',
                    'format': 'json',
                    'ids': entity_id,
                    'languages': 'en',
                    'props': 'labels|descriptions|claims'
                }
                response2 = requests.get(SEARCH_TOOLS["Wikidata"]["endpoint"], params=params2, timeout=10)
                if response2.status_code == 200:
                    entity_data = response2.json()
                    entity_info = entity_data.get('entities', {}).get(entity_id, {})
                    
                    # Extract key information
                    labels = entity_info.get('labels', {}).get('en', {}).get('value', '')
                    description = entity_info.get('descriptions', {}).get('en', {}).get('value', '')
                    
                    # Extract some claims
                    claims = entity_info.get('claims', {})
                    properties = {}
                    
                    # Common properties
                    prop_mapping = {
                        'P31': 'instance_of',
                        'P21': 'gender',
                        'P569': 'date_of_birth',
                        'P570': 'date_of_death',
                        'P27': 'country',
                        'P17': 'country'
                    }
                    
                    for prop_id, prop_name in prop_mapping.items():
                        if prop_id in claims:
                            claim = claims[prop_id][0]
                            if 'mainsnak' in claim and 'datavalue' in claim['mainsnak']:
                                value = claim['mainsnak']['datavalue'].get('value', {})
                                if isinstance(value, dict) and 'id' in value:
                                    properties[prop_name] = f"Q{value['id']}"
                                elif isinstance(value, str):
                                    properties[prop_name] = value
                    
                    results.append({
                        'id': entity_id,
                        'label': labels,
                        'description': description,
                        'properties': properties,
                        'url': f"https://www.wikidata.org/wiki/{entity_id}",
                        'source': 'Wikidata'
                    })
        
        return results
    except Exception:
        return []

def search_arxiv(query: str) -> List[Dict]:
    """Enhanced ArXiv search."""
    try:
        params = {
            'search_query': f'all:{query}',
            'start': 0,
            'max_results': 5,
            'sortBy': 'relevance',
            'sortOrder': 'descending'
        }
        response = requests.get(SEARCH_TOOLS["ArXiv"]["endpoint"], params=params, timeout=10)
        response.raise_for_status()
        
        root = ET.fromstring(response.content)
        
        papers = []
        for entry in root.findall('{http://www.w3.org/2005/Atom}entry'):
            title_elem = entry.find('{http://www.w3.org/2005/Atom}title')
            summary_elem = entry.find('{http://www.w3.org/2005/Atom}summary')
            
            if title_elem is not None and summary_elem is not None:
                title = title_elem.text.strip() if title_elem.text else ''
                summary = summary_elem.text.strip() if summary_elem.text else ''
                
                if title and summary:
                    # Get authors
                    authors = []
                    for author in entry.findall('{http://www.w3.org/2005/Atom}author'):
                        name_elem = author.find('{http://www.w3.org/2005/Atom}name')
                        if name_elem is not None and name_elem.text:
                            authors.append(name_elem.text.strip())
                    
                    # Get published date
                    published_elem = entry.find('{http://www.w3.org/2005/Atom}published')
                    published = published_elem.text[:10] if published_elem is not None and published_elem.text else ''
                    
                    # Get categories
                    categories = []
                    for category in entry.findall('{http://www.w3.org/2005/Atom}category'):
                        if 'term' in category.attrib:
                            categories.append(category.attrib['term'])
                    
                    papers.append({
                        'title': title,
                        'summary': summary[:400] + '...' if len(summary) > 400 else summary,
                        'authors': authors[:3],
                        'published': published,
                        'categories': categories[:3],
                        'url': f"https://arxiv.org/abs/{entry.find('{http://www.w3.org/2005/Atom}id').text.split('/')[-1] if entry.find('{http://www.w3.org/2005/Atom}id') is not None else ''}",
                        'source': 'ArXiv',
                        'relevance': 1.0
                    })
        
        return papers
    except Exception:
        return []

def search_pubmed(query: str) -> List[Dict]:
    """Search PubMed for medical research."""
    try:
        # Search for articles
        params = {
            'db': 'pubmed',
            'term': query,
            'retmode': 'json',
            'retmax': 5,
            'sort': 'relevance'
        }
        response = requests.get(SEARCH_TOOLS["PubMed"]["endpoint"], params=params, timeout=10)
        response.raise_for_status()
        data = response.json()
        
        article_ids = data.get('esearchresult', {}).get('idlist', [])
        
        if not article_ids:
            return []
        
        # Get article details
        fetch_params = {
            'db': 'pubmed',
            'id': ','.join(article_ids),
            'retmode': 'xml'
        }
        fetch_url = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/efetch.fcgi"
        response2 = requests.get(fetch_url, params=fetch_params, timeout=10)
        
        if response2.status_code != 200:
            return []
        
        # Parse XML
        root = ET.fromstring(response2.content)
        
        articles = []
        for article in root.findall('.//PubmedArticle'):
            # Extract title
            title_elem = article.find('.//ArticleTitle')
            title = title_elem.text if title_elem is not None else ''
            
            # Extract abstract
            abstract_elem = article.find('.//AbstractText')
            abstract = abstract_elem.text if abstract_elem is not None else ''
            
            # Extract authors
            authors = []
            for author in article.findall('.//Author'):
                last_name = author.find('LastName')
                fore_name = author.find('ForeName')
                if last_name is not None and last_name.text:
                    author_name = last_name.text
                    if fore_name is not None and fore_name.text:
                        author_name = f"{fore_name.text} {author_name}"
                    authors.append(author_name)
            
            # Extract publication date
            pub_date_elem = article.find('.//PubDate/Year')
            pub_date = pub_date_elem.text if pub_date_elem is not None else ''
            
            # Extract journal
            journal_elem = article.find('.//Journal/Title')
            journal = journal_elem.text if journal_elem is not None else ''
            
            if title:
                articles.append({
                    'title': title,
                    'abstract': abstract[:500] + '...' if len(abstract) > 500 else abstract,
                    'authors': authors[:3],
                    'journal': journal,
                    'publication_date': pub_date,
                    'pmid': article.find('.//PMID').text if article.find('.//PMID') is not None else '',
                    'url': f"https://pubmed.ncbi.nlm.nih.gov/{article.find('.//PMID').text if article.find('.//PMID') is not None else ''}",
                    'source': 'PubMed'
                })
        
        return articles
    except Exception:
        return []

def search_openlibrary(query: str) -> List[Dict]:
    """Search OpenLibrary for books."""
    try:
        params = {
            'q': query,
            'limit': 5,
            'mode': 'everything'
        }
        response = requests.get(SEARCH_TOOLS["OpenLibrary"]["endpoint"], params=params, timeout=10)
        response.raise_for_status()
        data = response.json()
        
        books = []
        for doc in data.get('docs', []):
            title = doc.get('title', '')
            author = doc.get('author_name', [''])[0] if doc.get('author_name') else ''
            publish_year = doc.get('first_publish_year', '')
            
            if title:
                books.append({
                    'title': title,
                    'author': author,
                    'publish_year': publish_year,
                    'isbn': doc.get('isbn', [''])[0] if doc.get('isbn') else '',
                    'cover_id': doc.get('cover_i', ''),
                    'url': f"https://openlibrary.org{doc.get('key', '')}",
                    'source': 'OpenLibrary'
                })
        
        return books
    except Exception:
        return []

def search_dictionary(query: str) -> Dict:
    """Search for word definitions."""
    try:
        # Clean query for single word
        words = re.findall(r'\b[a-zA-Z]+\b', query)
        if not words:
            return {}
        
        word = words[0].lower()
        response = requests.get(f"{SEARCH_TOOLS['Dictionary']['endpoint']}{word}", timeout=8)
        
        if response.status_code == 200:
            data = response.json()
            if data:
                result = {
                    'word': word,
                    'phonetic': '',
                    'meanings': [],
                    'source': 'Dictionary API'
                }
                
                for entry in data[:1]:
                    if 'phonetic' in entry:
                        result['phonetic'] = entry['phonetic']
                    
                    for meaning in entry.get('meanings', []):
                        part_of_speech = meaning.get('partOfSpeech', '')
                        for definition in meaning.get('definitions', [])[:2]:
                            result['meanings'].append({
                                'part_of_speech': part_of_speech,
                                'definition': definition.get('definition', ''),
                                'example': definition.get('example', '')
                            })
                
                return result
        return {}
    except Exception:
        return {}

def search_countries(query: str) -> List[Dict]:
    """Search for country information."""
    try:
        # Try by name
        response = requests.get(f"{SEARCH_TOOLS['Countries']['endpoint']}name/{query}", timeout=8)
        
        if response.status_code != 200:
            # Try by partial name
            response = requests.get(f"{SEARCH_TOOLS['Countries']['endpoint']}name/{query}?fullText=false", timeout=8)
        
        if response.status_code == 200:
            data = response.json()
            countries = []
            
            for country in data[:3]:
                country_info = {
                    'name': country.get('name', {}).get('common', ''),
                    'official_name': country.get('name', {}).get('official', ''),
                    'capital': country.get('capital', [''])[0] if country.get('capital') else '',
                    'region': country.get('region', ''),
                    'population': country.get('population', 0),
                    'area': country.get('area', 0),
                    'languages': list(country.get('languages', {}).values()) if country.get('languages') else [],
                    'currency': list(country.get('currencies', {}).keys()) if country.get('currencies') else [],
                    'flag': country.get('flags', {}).get('png', ''),
                    'source': 'REST Countries'
                }
                countries.append(country_info)
            
            return countries
        return []
    except Exception:
        return []

def search_quotes(query: str) -> List[Dict]:
    """Search for famous quotes."""
    try:
        # Search quotes by keyword
        params = {
            'query': query,
            'limit': 5,
            'maxLength': 100
        }
        response = requests.get(f"{SEARCH_TOOLS['Quotes']['endpoint']}search/quotes", params=params, timeout=8)
        response.raise_for_status()
        data = response.json()
        
        quotes = []
        for quote in data.get('results', []):
            quotes.append({
                'content': quote.get('content', ''),
                'author': quote.get('author', ''),
                'tags': quote.get('tags', [])[:3],
                'source': 'Quotable'
            })
        
        return quotes
    except Exception:
        return []

def search_github(query: str) -> List[Dict]:
    """Search GitHub repositories."""
    try:
        params = {
            'q': query,
            'sort': 'stars',
            'order': 'desc',
            'per_page': 5
        }
        headers = {
            'Accept': 'application/vnd.github.v3+json'
        }
        response = requests.get(
            SEARCH_TOOLS['GitHub']['endpoint'],
            params=params,
            headers=headers,
            timeout=10
        )
        response.raise_for_status()
        data = response.json()
        
        repos = []
        for repo in data.get('items', []):
            repos.append({
                'name': repo.get('name', ''),
                'full_name': repo.get('full_name', ''),
                'description': repo.get('description', ''),
                'stars': repo.get('stargazers_count', 0),
                'forks': repo.get('forks_count', 0),
                'language': repo.get('language', ''),
                'url': repo.get('html_url', ''),
                'updated_at': repo.get('updated_at', ''),
                'source': 'GitHub'
            })
        
        return repos
    except Exception:
        return []

def search_stackoverflow(query: str) -> List[Dict]:
    """Search Stack Overflow questions."""
    try:
        params = {
            'order': 'desc',
            'sort': 'relevance',
            'q': query,
            'site': 'stackoverflow',
            'pagesize': 5
        }
        response = requests.get(SEARCH_TOOLS['StackOverflow']['endpoint'], params=params, timeout=10)
        response.raise_for_status()
        data = response.json()
        
        questions = []
        for question in data.get('items', []):
            questions.append({
                'title': question.get('title', ''),
                'score': question.get('score', 0),
                'answer_count': question.get('answer_count', 0),
                'view_count': question.get('view_count', 0),
                'tags': question.get('tags', [])[:3],
                'is_answered': question.get('is_answered', False),
                'link': question.get('link', ''),
                'creation_date': datetime.fromtimestamp(question.get('creation_date', 0)).strftime('%Y-%m-%d'),
                'source': 'StackOverflow'
            })
        
        return questions
    except Exception:
        return []

def search_geocoding(query: str) -> List[Dict]:
    """Geocoding search with Nominatim."""
    try:
        params = {
            'q': query,
            'format': 'json',
            'limit': 3,
            'addressdetails': 1
        }
        headers = {
            'User-Agent': 'DeepThinkPro-App/1.0'
        }
        response = requests.get(
            SEARCH_TOOLS['Geocoding']['endpoint'],
            params=params,
            headers=headers,
            timeout=10
        )
        response.raise_for_status()
        data = response.json()
        
        locations = []
        for location in data[:3]:
            locations.append({
                'name': location.get('display_name', ''),
                'latitude': location.get('lat', ''),
                'longitude': location.get('lon', ''),
                'type': location.get('type', ''),
                'importance': location.get('importance', 0),
                'address': location.get('address', {}),
                'source': 'Nominatim'
            })
        
        return locations
    except Exception:
        return []

def search_weather(query: str) -> Dict:
    """Weather information search."""
    try:
        # Try to get location from query
        location = query
        
        # If query doesn't look like a location, try geocoding first
        if not any(word in query.lower() for word in ['weather', 'temperature', 'forecast', 'city', 'town']):
            geocoding_results = search_geocoding(query)
            if geocoding_results:
                location = geocoding_results[0]['name'].split(',')[0]
        
        params = {
            'format': 'j1',
            'lang': 'en'
        }
        response = requests.get(f"{SEARCH_TOOLS['Weather']['endpoint']}{location}", params=params, timeout=10)
        
        if response.status_code == 200:
            data = response.json()
            
            current = data.get('current_condition', [{}])[0]
            weather = data.get('weather', [{}])[0]
            
            result = {
                'location': location,
                'temperature': current.get('temp_C', ''),
                'feels_like': current.get('FeelsLikeC', ''),
                'description': current.get('weatherDesc', [{}])[0].get('value', ''),
                'humidity': current.get('humidity', ''),
                'wind_speed': current.get('windspeedKmph', ''),
                'wind_direction': current.get('winddir16Point', ''),
                'precipitation': current.get('precipMM', ''),
                'pressure': current.get('pressure', ''),
                'visibility': current.get('visibility', ''),
                'sunrise': weather.get('astronomy', [{}])[0].get('sunrise', ''),
                'sunset': weather.get('astronomy', [{}])[0].get('sunset', ''),
                'source': 'wttr.in'
            }
            
            return {k: v for k, v in result.items() if v}
        return {}
    except Exception:
        return {}

def search_air_quality(query: str) -> List[Dict]:
    """Air quality data search."""
    try:
        # Extract location from query
        location = query.split()[-1] if len(query.split()) > 1 else query
        
        params = {
            'limit': 3,
            'page': 1,
            'offset': 0,
            'sort': 'desc',
            'radius': 1000,
            'order_by': 'lastUpdated',
            'dumpRaw': 'false'
        }
        
        # Try by city name
        response = requests.get(
            f"{SEARCH_TOOLS['Air Quality']['endpoint']}latest",
            params={**params, 'city': location},
            timeout=10
        )
        
        if response.status_code != 200 or not response.json().get('results'):
            # Try by country
            response = requests.get(
                f"{SEARCH_TOOLS['Air Quality']['endpoint']}latest",
                params={**params, 'country': location},
                timeout=10
            )
        
        if response.status_code == 200:
            data = response.json()
            measurements = []
            
            for result in data.get('results', [])[:3]:
                for measurement in result.get('measurements', [])[:2]:
                    measurements.append({
                        'location': result.get('location', ''),
                        'city': result.get('city', ''),
                        'country': result.get('country', ''),
                        'parameter': measurement.get('parameter', ''),
                        'value': measurement.get('value', 0),
                        'unit': measurement.get('unit', ''),
                        'last_updated': measurement.get('lastUpdated', ''),
                        'source': 'OpenAQ'
                    })
            
            return measurements
        return []
    except Exception:
        return []

# Function mapping for search tools
SEARCH_FUNCTIONS = {
    'Wikipedia': search_wikipedia,
    'DuckDuckGo': search_duckduckgo,
    'DuckDuckGo News': search_duckduckgo_news,
    'Wikidata': search_wikidata,
    'ArXiv': search_arxiv,
    'PubMed': search_pubmed,
    'OpenLibrary': search_openlibrary,
    'Dictionary': search_dictionary,
    'Countries': search_countries,
    'Quotes': search_quotes,
    'GitHub': search_github,
    'StackOverflow': search_stackoverflow,
    'Geocoding': search_geocoding,
    'Weather': search_weather,
    'Air Quality': search_air_quality
}

def smart_source_selector(query: str) -> List[str]:
    """Intelligently select which sources to search based on query."""
    query_lower = query.lower()
    
    selected_sources = []
    
    # Always include these for general knowledge
    selected_sources.append('Wikipedia')
    selected_sources.append('DuckDuckGo')
    
    # Check for specific patterns
    if any(word in query_lower for word in ['news', 'latest', 'update', 'current', 'recent']):
        selected_sources.append('DuckDuckGo News')
    
    if any(word in query_lower for word in ['data', 'structure', 'entity', 'property', 'fact']):
        selected_sources.append('Wikidata')
    
    if any(word in query_lower for word in ['research', 'paper', 'study', 'scientific', 'academic']):
        selected_sources.append('ArXiv')
        selected_sources.append('PubMed')
    
    if any(word in query_lower for word in ['medical', 'health', 'disease', 'medicine', 'hospital']):
        selected_sources.append('PubMed')
    
    if any(word in query_lower for word in ['book', 'author', 'novel', 'literature', 'publish']):
        selected_sources.append('OpenLibrary')
    
    if any(word in query_lower for word in ['define', 'definition', 'meaning', 'word', 'dictionary']):
        selected_sources.append('Dictionary')
    
    if any(word in query_lower for word in ['country', 'capital', 'population', 'flag', 'nation']):
        selected_sources.append('Countries')
    
    if any(word in query_lower for word in ['quote', 'say', 'wisdom', 'inspiration', 'famous']):
        selected_sources.append('Quotes')
    
    if any(word in query_lower for word in ['code', 'github', 'repository', 'programming', 'software']):
        selected_sources.append('GitHub')
        selected_sources.append('StackOverflow')
    
    if any(word in query_lower for word in ['location', 'map', 'where is', 'coordinates', 'place']):
        selected_sources.append('Geocoding')
    
    if any(word in query_lower for word in ['weather', 'temperature', 'forecast', 'rain', 'sunny']):
        selected_sources.append('Weather')
    
    if any(word in query_lower for word in ['air', 'pollution', 'quality', 'aqi', 'pm2.5']):
        selected_sources.append('Air Quality')
    
    # Remove duplicates and limit to 8 sources max
    selected_sources = list(dict.fromkeys(selected_sources))[:8]
    
    return selected_sources

def perform_intelligent_search(query: str) -> Dict[str, Any]:
    """Perform parallel search on intelligently selected sources."""
    selected_sources = smart_source_selector(query)
    
    results = {}
    with concurrent.futures.ThreadPoolExecutor(max_workers=len(selected_sources)) as executor:
        future_to_source = {}
        for source in selected_sources:
            if source in SEARCH_FUNCTIONS:
                future = executor.submit(SEARCH_FUNCTIONS[source], query)
                future_to_source[future] = source
        
        for future in concurrent.futures.as_completed(future_to_source):
            source_name = future_to_source[future]
            try:
                data = future.result(timeout=10)
                if data:
                    results[source_name] = data
            except Exception:
                continue
    
    return results

def analyze_search_results(query: str, results: Dict) -> Dict:
    """Analyze search results to extract key insights."""
    analysis = {
        'key_facts': [],
        'conflicting_info': [],
        'knowledge_gaps': [],
        'source_quality': {},
        'main_themes': []
    }
    
    # Extract key facts from each source
    for source, data in results.items():
        if source == 'Wikipedia' and isinstance(data, list):
            for item in data[:2]:
                if 'summary' in item:
                    analysis['key_facts'].append({
                        'fact': item['summary'][:200],
                        'source': source,
                        'title': item.get('title', '')
                    })
        
        elif source == 'DuckDuckGo' and isinstance(data, dict):
            if data.get('answer'):
                analysis['key_facts'].append({
                    'fact': data['answer'],
                    'source': source,
                    'type': 'direct_answer'
                })
            if data.get('abstract'):
                analysis['key_facts'].append({
                    'fact': data['abstract'][:200],
                    'source': source,
                    'type': 'abstract'
                })
        
        elif source == 'ArXiv' and isinstance(data, list):
            for paper in data[:1]:
                analysis['key_facts'].append({
                    'fact': f"Research paper: {paper.get('title', '')}",
                    'source': source,
                    'type': 'scientific'
                })
        
        elif source == 'PubMed' and isinstance(data, list):
            for article in data[:1]:
                analysis['key_facts'].append({
                    'fact': f"Medical research: {article.get('title', '')}",
                    'source': source,
                    'type': 'medical'
                })
    
    # Identify potential knowledge gaps
    query_terms = query.lower().split()
    found_terms = []
    for fact in analysis['key_facts']:
        fact_text = fact['fact'].lower()
        for term in query_terms:
            if term in fact_text:
                found_terms.append(term)
    
    missing_terms = [term for term in query_terms if term not in found_terms]
    if missing_terms:
        analysis['knowledge_gaps'].append(f"Missing information about: {', '.join(missing_terms[:3])}")
    
    # Assess source quality
    for source in results:
        if source == 'Wikipedia':
            analysis['source_quality'][source] = {'reliability': 'high', 'coverage': 'broad'}
        elif source == 'ArXiv':
            analysis['source_quality'][source] = {'reliability': 'high', 'coverage': 'specialized'}
        elif source == 'PubMed':
            analysis['source_quality'][source] = {'reliability': 'high', 'coverage': 'medical'}
        elif source == 'DuckDuckGo':
            analysis['source_quality'][source] = {'reliability': 'medium', 'coverage': 'general'}
    
    return analysis

def create_thinking_prompt(query: str, messages: List[Dict], system_prompt: str, 
                          search_results: Dict, search_analysis: Dict) -> str:
    """Create an enhanced prompt that encourages deep thinking."""
    
    # Build search context
    search_context = "RELEVANT INFORMATION FOUND:\n\n"
    
    for source, data in search_results.items():
        search_context += f"=== {source.upper()} ===\n"
        
        if isinstance(data, list):
            for item in data[:2]:
                if isinstance(item, dict):
                    if 'title' in item:
                        search_context += f"Title: {item['title']}\n"
                    if 'summary' in item:
                        search_context += f"Summary: {item['summary']}\n"
                    if 'answer' in item:
                        search_context += f"Answer: {item['answer']}\n"
                    if 'abstract' in item:
                        search_context += f"Abstract: {item['abstract'][:300]}...\n"
                    search_context += "\n"
        
        elif isinstance(data, dict):
            for key, value in data.items():
                if key not in ['source', 'type', 'image'] and value:
                    if isinstance(value, list):
                        search_context += f"{key}: {', '.join(str(v) for v in value[:3])}\n"
                    else:
                        search_context += f"{key}: {value}\n"
            search_context += "\n"
    
    # Add analysis insights
    if search_analysis['key_facts']:
        search_context += "ANALYSIS INSIGHTS:\n"
        search_context += "• Key facts identified from sources\n"
        if search_analysis['knowledge_gaps']:
            search_context += f"• Knowledge gaps: {search_analysis['knowledge_gaps'][0]}\n"
        if search_analysis['source_quality']:
            search_context += "• Source reliability assessed\n"
    
    # Build conversation history
    conversation = ""
    for msg in messages[-4:]:
        if msg["role"] == "user":
            conversation += f"User: {msg['content']}\n"
        elif msg["role"] == "assistant":
            conversation += f"Assistant: {msg['content']}\n"
    
    # Final prompt
    prompt = f"""<|system|>
{system_prompt}

CURRENT DATE: {datetime.now().strftime('%B %d, %Y')}

USER'S QUESTION: {query}

{search_context}

CONVERSATION CONTEXT:
{conversation}

THINKING INSTRUCTIONS:
1. First, verify the key information from sources
2. Identify the most reliable facts
3. Consider historical context if relevant
4. Think about why this matters
5. Connect to broader themes or concepts
6. Identify what's still unknown or debated
7. Formulate a comprehensive yet concise answer
8. End with thought-provoking questions or further reading suggestions

IMPORTANT: Show your reasoning process. Be precise about what's well-established vs. what's uncertain.</s>

<|user|>
{query}</s>

<|assistant|>
"""
    
    return prompt

def generate_thoughtful_response(model, prompt: str, max_tokens: int = 768, temperature: float = 0.7) -> str:
    """Generate response with thinking emphasis."""
    
    response = model(
        prompt,
        max_new_tokens=max_tokens,
        temperature=temperature,
        top_p=0.9,
        repetition_penalty=1.1,
        stop=["</s>", "<|user|>", "\n\nUser:", "### END", "Sources:"]
    )
    
    # Clean up response
    response = response.strip()
    
    # Ensure it doesn't cut off mid-thought
    if response.count('.') < 2:
        # If response seems incomplete, try to extend it
        extended = model(
            prompt + response,
            max_new_tokens=200,
            temperature=temperature,
            top_p=0.9
        )
        response = response + " " + extended.strip()
    
    return response

# Sidebar
with st.sidebar:
    st.header("🎭 Thinking Persona")
    
    # Safely get index for selectbox
    preset_keys = list(PRESET_PROMPTS.keys())
    current_preset = st.session_state.selected_preset
    
    # Ensure current preset is valid
    if current_preset not in preset_keys:
        current_preset = "Deep Thinker Pro"
        st.session_state.selected_preset = current_preset
    
    index = preset_keys.index(current_preset)
    
    persona = st.selectbox(
        "Select AI Persona:",
        options=preset_keys,
        index=index
    )
    
    if persona != st.session_state.selected_preset:
        st.session_state.selected_preset = persona
        st.session_state.system_prompt = PRESET_PROMPTS[persona]
    
    st.divider()
    
    st.header("⚡ Thinking Parameters")
    
    thinking_mode = st.radio(
        "Thinking Mode:",
        ["Analytical", "Creative", "Critical", "Balanced"],
        index=3
    )
    
    research_depth = st.select_slider(
        "Research Depth:",
        options=["Quick Scan", "Moderate", "Deep Dive", "Exhaustive"],
        value="Moderate"
    )
    
    temperature = st.slider(
        "Creativity Level:",
        0.1, 1.5, 0.7, 0.1,
        help="Lower = more factual, Higher = more creative"
    )
    
    st.divider()
    
    st.header("🔧 Tools")
    
    auto_search = st.toggle("Auto-Research", value=True)
    show_thinking = st.toggle("Show Thinking Process", value=True)
    
    col1, col2 = st.columns(2)
    with col1:
        if st.button("🔄 New Chat", use_container_width=True):
            st.session_state.messages = []
            st.rerun()
    with col2:
        if st.button("🧠 Reset AI", use_container_width=True):
            st.session_state.system_prompt = PRESET_PROMPTS["Deep Thinker Pro"]
            st.session_state.selected_preset = "Deep Thinker Pro"
            st.rerun()
    
    st.divider()
    st.caption("DeepThink Pro v1.2")
    st.caption("Advanced thinking AI with 15+ search sources")

# Main interface
col1, col2, col3 = st.columns([3, 1, 1])
with col1:
    st.title("🧠 DeepThink Pro")
with col2:
    if auto_search:
        st.success("🔍 Auto-Research ON")
with col3:
    if show_thinking:
        st.info("💭 Showing Thoughts")

# Display current persona
with st.expander("🤖 Active Persona", expanded=False):
    st.write(st.session_state.selected_preset)
    st.caption(st.session_state.system_prompt[:300] + "...")

# Load model
if st.session_state.model is None:
    with st.spinner("🚀 Loading AI Brain..."):
        try:
            st.session_state.model = load_model()
            st.success("✅ AI Ready for Deep Thinking!")
        except Exception as e:
            st.error(f"❌ Failed to load: {str(e)}")
            st.stop()

# Display chat
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        
        # Show source tags if available
        if "metadata" in message and "sources" in message["metadata"]:
            st.markdown("<div style='margin-top: 10px;'>", unsafe_allow_html=True)
            for source in message["metadata"]["sources"]:
                icon = SEARCH_TOOLS.get(source, {}).get('icon', '📌')
                st.markdown(f'<span class="source-tag">{icon} {source}</span>', unsafe_allow_html=True)
            st.markdown("</div>", unsafe_allow_html=True)

# Chat input
if prompt := st.chat_input("Ask me anything..."):
    # Add user message
    st.session_state.messages.append({"role": "user", "content": prompt})
    
    with st.chat_message("user"):
        st.markdown(prompt)
    
    # Prepare assistant response
    with st.chat_message("assistant"):
        # Step 1: Show thinking
        thinking_placeholder = st.empty()
        
        if show_thinking:
            thinking_placeholder.markdown("""
            <div class="thinking-bubble">
            <strong>💭 Initial Analysis:</strong><br>
            1. Parsing question structure and intent<br>
            2. Identifying key concepts and entities<br>
            3. Determining appropriate research approach<br>
            4. Preparing search strategy...
            </div>
            """, unsafe_allow_html=True)
        
        # Step 2: Intelligent Search
        search_results = {}
        search_analysis = {}
        
        if auto_search:
            if show_thinking:
                thinking_placeholder.markdown("""
                <div class="thinking-bubble">
                <strong>🔍 Smart Research:</strong><br>
                • Analyzing query type and selecting optimal sources<br>
                • Conducting parallel searches across selected databases<br>
                • Evaluating source reliability and relevance...
                </div>
                """, unsafe_allow_html=True)
            
            with st.spinner("🔍 Conducting intelligent research..."):
                search_results = perform_intelligent_search(prompt)
                
                if search_results:
                    search_analysis = analyze_search_results(prompt, search_results)
                    
                    # Display search summary
                    with st.expander("📊 Research Summary", expanded=False):
                        sources_used = len(search_results)
                        st.write(f"**Sources consulted:** {sources_used}")
                        
                        for source, data in search_results.items():
                            st.subheader(f"{SEARCH_TOOLS.get(source, {}).get('icon', '📌')} {source}")
                            
                            if isinstance(data, list):
                                for item in data[:2]:
                                    if isinstance(item, dict):
                                        with st.container():
                                            if 'title' in item:
                                                st.write(f"**{item['title']}**")
                                            if 'summary' in item:
                                                st.write(item['summary'])
                                            if 'authors' in item:
                                                st.write(f"*Authors:* {', '.join(item['authors'][:3])}")
                                            st.divider()
                            elif isinstance(data, dict):
                                for key, value in data.items():
                                    if key not in ['source', 'type', 'image'] and value:
                                        if isinstance(value, list):
                                            st.write(f"**{key.title()}:** {', '.join(str(v) for v in value[:3])}")
                                        else:
                                            st.write(f"**{key.title()}:** {value}")
        
        # Step 3: Generate thoughtful response
        if show_thinking:
            thinking_placeholder.markdown("""
            <div class="thinking-bubble">
            <strong>🤔 Deep Synthesis:</strong><br>
            • Integrating information from multiple sources<br>
            • Applying critical thinking and analysis<br>
            • Formulating comprehensive response<br>
            • Preparing insights and recommendations...
            </div>
            """, unsafe_allow_html=True)
        
        with st.spinner("🧠 Engaging deep thinking process..."):
            # Create enhanced prompt
            enhanced_prompt = create_thinking_prompt(
                prompt, 
                st.session_state.messages,
                st.session_state.system_prompt,
                search_results,
                search_analysis
            )
            
            # Adjust tokens based on research depth
            token_map = {
                "Quick Scan": 512,
                "Moderate": 768,
                "Deep Dive": 1024,
                "Exhaustive": 1536
            }
            tokens = token_map.get(research_depth, 768)
            
            # Generate response
            response = generate_thoughtful_response(
                st.session_state.model,
                enhanced_prompt,
                max_tokens=tokens,
                temperature=temperature
            )
        
        # Clear thinking placeholders
        thinking_placeholder.empty()
        
        # Display response
        st.markdown(response)
        
        # Add analysis box for deep thinking
        if thinking_mode != "Quick Scan" and search_results:
            st.markdown(f"""
            <div class="analysis-box">
            <strong>📈 Analysis Summary:</strong><br>
            • Information synthesized from {len(search_results)} sources<br>
            • Key themes identified<br>
            • Reliability assessment completed<br>
            • Knowledge gaps noted for further research
            </div>
            """, unsafe_allow_html=True)
        
        # Store message with metadata
        metadata = {
            "sources": list(search_results.keys()) if search_results else [],
            "thinking_mode": thinking_mode,
            "research_depth": research_depth,
            "timestamp": datetime.now().isoformat()
        }
        
        st.session_state.messages.append({
            "role": "assistant", 
            "content": response,
            "metadata": metadata
        })

# Add quick questions examples
if not st.session_state.messages:
    st.markdown("### 💡 Try asking about:")
    
    col1, col2 = st.columns(2)
    with col1:
        if st.button("Historical figure analysis", use_container_width=True):
            st.session_state.messages.append({"role": "user", "content": "Who was Napoleon Bonaparte and what was his impact on Europe?"})
            st.rerun()
        if st.button("Scientific concept", use_container_width=True):
            st.session_state.messages.append({"role": "user", "content": "Explain quantum entanglement in simple terms"})
            st.rerun()
        if st.button("Weather forecast", use_container_width=True):
            st.session_state.messages.append({"role": "user", "content": "What's the weather like in New York?"})
            st.rerun()
        if st.button("Code example", use_container_width=True):
            st.session_state.messages.append({"role": "user", "content": "Show me popular Python machine learning repositories on GitHub"})
            st.rerun()
    
    with col2:
        if st.button("Current events", use_container_width=True):
            st.session_state.messages.append({"role": "user", "content": "What are the main challenges facing renewable energy adoption today?"})
            st.rerun()
        if st.button("Philosophical question", use_container_width=True):
            st.session_state.messages.append({"role": "user", "content": "What is the meaning of consciousness according to different philosophical traditions?"})
            st.rerun()
        if st.button("Medical research", use_container_width=True):
            st.session_state.messages.append({"role": "user", "content": "What are the latest findings about Alzheimer's disease?"})
            st.rerun()
        if st.button("Country information", use_container_width=True):
            st.session_state.messages.append({"role": "user", "content": "Tell me about Japan's culture and economy"})
            st.rerun()
