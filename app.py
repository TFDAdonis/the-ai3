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

if "sandbox_mode" not in st.session_state:
    st.session_state.sandbox_mode = False

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
    .warning-box {
        background-color: #fff3cd;
        border: 1px solid #ffc107;
        padding: 1rem;
        border-radius: 10px;
        margin: 1rem 0;
    }
</style>
""", unsafe_allow_html=True)

# Safe download function with better error handling
def safe_download_model():
    MODEL_DIR.mkdir(exist_ok=True)
    
    if MODEL_PATH.exists():
        return True
    
    with st.spinner("⚠️ Model not found. Checking for local copy..."):
        try:
            # First check if we can reach the URL without downloading
            head_response = requests.head(MODEL_URL, timeout=5)
            if head_response.status_code == 200:
                st.info("Model found online. Starting download...")
                
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
            else:
                st.warning("⚠️ Cannot download model from external source. Running in offline mode.")
                st.session_state.sandbox_mode = True
                return False
                
        except Exception as e:
            st.warning(f"⚠️ Download failed: {str(e)[:100]}... Running in offline mode.")
            st.session_state.sandbox_mode = True
            return False
    return False

@st.cache_resource(show_spinner=False)
def safe_load_model():
    try:
        from ctransformers import AutoModelForCausalLM
        
        if not MODEL_PATH.exists():
            if not safe_download_model():
                # Create a dummy model for offline mode
                class DummyModel:
                    def __call__(self, prompt, **kwargs):
                        return "I'm running in offline/sandbox mode. External searches are disabled due to Hugging Face Space restrictions. Please ask me general questions based on my internal knowledge."
                
                return DummyModel()
        
        return AutoModelForCausalLM.from_pretrained(
            str(MODEL_DIR),
            model_file=MODEL_PATH.name,
            model_type="llama",
            context_length=2048,  # Reduced for stability
            gpu_layers=0,
            threads=4  # Reduced for stability
        )
    except Exception as e:
        st.error(f"Model loading error: {str(e)[:200]}")
        # Return dummy model
        class DummyModel:
            def __call__(self, prompt, **kwargs):
                return "Model initialization failed. Running in basic mode."
        
        return DummyModel()

# SAFE SEARCH FUNCTIONS - All wrapped in try-except
def safe_search_wikipedia(query: str) -> List[Dict]:
    """Safe Wikipedia search with fallback."""
    if st.session_state.sandbox_mode:
        return [{
            'title': 'Wikipedia (Sandbox Mode)',
            'summary': f'Information about "{query}" would normally appear here, but external searches are disabled in this environment.',
            'source': 'Wikipedia',
            'sandbox': True
        }]
    
    try:
        params = {
            'action': 'query',
            'format': 'json',
            'list': 'search',
            'srsearch': query,
            'srlimit': 3,
            'utf8': 1
        }
        response = requests.get(SEARCH_TOOLS["Wikipedia"]["endpoint"], params=params, timeout=5)
        response.raise_for_status()
        data = response.json()
        
        results = []
        for item in data.get('query', {}).get('search', [])[:2]:
            results.append({
                'title': item.get('title', ''),
                'summary': item.get('snippet', '') + '...',
                'source': 'Wikipedia'
            })
        
        return results if results else []
    except Exception:
        return [{
            'title': 'Wikipedia Search Failed',
            'summary': f'Could not retrieve Wikipedia results for "{query}". External API might be blocked.',
            'source': 'Wikipedia',
            'error': True
        }]

def safe_search_duckduckgo(query: str) -> Dict:
    """Safe DuckDuckGo search."""
    if st.session_state.sandbox_mode:
        return {
            'abstract': f'Web search for "{query}" is disabled in sandbox mode.',
            'answer': 'Try asking general knowledge questions instead.',
            'source': 'DuckDuckGo',
            'sandbox': True
        }
    
    try:
        params = {
            'q': query,
            'format': 'json',
            'no_html': 1,
            't': 'streamlit_app'
        }
        response = requests.get(SEARCH_TOOLS["DuckDuckGo"]["endpoint"], params=params, timeout=5)
        response.raise_for_status()
        data = response.json()
        
        result = {
            'abstract': data.get('AbstractText', ''),
            'answer': data.get('Answer', ''),
            'source': 'DuckDuckGo'
        }
        
        # Clean empty values
        return {k: v for k, v in result.items() if v}
    except Exception:
        return {
            'error': f'DuckDuckGo search failed for "{query}".',
            'source': 'DuckDuckGo'
        }

def safe_search_duckduckgo_news(query: str) -> List[Dict]:
    """Safe news search."""
    if st.session_state.sandbox_mode:
        return [{
            'title': 'News Search (Sandbox Mode)',
            'summary': f'News about "{query}" would appear here in online mode.',
            'source': 'DuckDuckGo News',
            'sandbox': True
        }]
    
    try:
        return safe_search_duckduckgo(query + ' news').get('related', [])
    except Exception:
        return []

def safe_search_wikidata(query: str) -> List[Dict]:
    """Safe Wikidata search."""
    if st.session_state.sandbox_mode:
        return [{
            'title': 'Structured Data (Sandbox)',
            'summary': f'Structured data about "{query}" unavailable in offline mode.',
            'source': 'Wikidata',
            'sandbox': True
        }]
    
    try:
        params = {
            'action': 'wbsearchentities',
            'format': 'json',
            'language': 'en',
            'search': query,
            'limit': 2
        }
        response = requests.get(SEARCH_TOOLS["Wikidata"]["endpoint"], params=params, timeout=5)
        response.raise_for_status()
        data = response.json()
        
        return [{
            'title': item.get('label', ''),
            'description': item.get('description', ''),
            'source': 'Wikidata'
        } for item in data.get('search', [])[:2]]
    except Exception:
        return []

def safe_search_arxiv(query: str) -> List[Dict]:
    """Safe arXiv search."""
    if st.session_state.sandbox_mode:
        return [{
            'title': 'Research Papers (Sandbox)',
            'summary': f'Academic papers about "{query}" would appear here in online mode.',
            'source': 'ArXiv',
            'sandbox': True
        }]
    
    try:
        params = {
            'search_query': f'all:{query}',
            'start': 0,
            'max_results': 2,
            'sortBy': 'relevance'
        }
        response = requests.get(SEARCH_TOOLS["ArXiv"]["endpoint"], params=params, timeout=5)
        response.raise_for_status()
        
        root = ET.fromstring(response.content)
        papers = []
        
        for entry in root.findall('{http://www.w3.org/2005/Atom}entry')[:2]:
            title_elem = entry.find('{http://www.w3.org/2005/Atom}title')
            summary_elem = entry.find('{http://www.w3.org/2005/Atom}summary')
            
            if title_elem is not None:
                papers.append({
                    'title': title_elem.text.strip() if title_elem.text else '',
                    'summary': summary_elem.text[:300] + '...' if summary_elem is not None and summary_elem.text else '',
                    'source': 'ArXiv'
                })
        
        return papers
    except Exception:
        return []

def safe_search_pubmed(query: str) -> List[Dict]:
    """Safe PubMed search."""
    if st.session_state.sandbox_mode:
        return [{
            'title': 'Medical Research (Sandbox)',
            'summary': f'Medical research about "{query}" unavailable in offline mode.',
            'source': 'PubMed',
            'sandbox': True
        }]
    
    try:
        params = {
            'db': 'pubmed',
            'term': query,
            'retmode': 'json',
            'retmax': 2
        }
        response = requests.get(SEARCH_TOOLS["PubMed"]["endpoint"], params=params, timeout=5)
        response.raise_for_status()
        
        return [{
            'title': f'PubMed result for "{query}"',
            'summary': 'Medical research data would appear here in online mode.',
            'source': 'PubMed'
        }]
    except Exception:
        return []

def safe_search_openlibrary(query: str) -> List[Dict]:
    """Safe OpenLibrary search."""
    if st.session_state.sandbox_mode:
        return [{
            'title': 'Books (Sandbox)',
            'summary': f'Books about "{query}" would appear here in online mode.',
            'source': 'OpenLibrary',
            'sandbox': True
        }]
    
    try:
        params = {'q': query, 'limit': 2}
        response = requests.get(SEARCH_TOOLS["OpenLibrary"]["endpoint"], params=params, timeout=5)
        response.raise_for_status()
        
        return [{
            'title': doc.get('title', ''),
            'author': doc.get('author_name', [''])[0] if doc.get('author_name') else '',
            'source': 'OpenLibrary'
        } for doc in response.json().get('docs', [])[:2]]
    except Exception:
        return []

def safe_search_dictionary(query: str) -> Dict:
    """Safe dictionary search."""
    if st.session_state.sandbox_mode:
        return {
            'word': query.split()[0] if query.split() else 'word',
            'definition': f'Definition would appear here in online mode.',
            'source': 'Dictionary',
            'sandbox': True
        }
    
    try:
        word = query.split()[0] if query.split() else query
        response = requests.get(f"{SEARCH_TOOLS['Dictionary']['endpoint']}{word}", timeout=5)
        
        if response.status_code == 200:
            data = response.json()
            if data and isinstance(data, list):
                first = data[0]
                return {
                    'word': first.get('word', ''),
                    'definition': first.get('meanings', [{}])[0].get('definitions', [{}])[0].get('definition', ''),
                    'source': 'Dictionary'
                }
        return {}
    except Exception:
        return {}

def safe_search_countries(query: str) -> List[Dict]:
    """Safe country search."""
    if st.session_state.sandbox_mode:
        return [{
            'name': 'Country Data (Sandbox)',
            'summary': f'Information about "{query}" would appear here in online mode.',
            'source': 'Countries',
            'sandbox': True
        }]
    
    try:
        response = requests.get(f"{SEARCH_TOOLS['Countries']['endpoint']}name/{query}", timeout=5)
        if response.status_code == 200:
            return [{
                'name': country.get('name', {}).get('common', ''),
                'capital': country.get('capital', [''])[0],
                'source': 'Countries'
            } for country in response.json()[:2]]
        return []
    except Exception:
        return []

def safe_search_quotes(query: str) -> List[Dict]:
    """Safe quotes search."""
    if st.session_state.sandbox_mode:
        return [{
            'content': f'Quotes about "{query}" would appear here in online mode.',
            'author': 'Various',
            'source': 'Quotes',
            'sandbox': True
        }]
    
    try:
        params = {'query': query, 'limit': 2}
        response = requests.get(f"{SEARCH_TOOLS['Quotes']['endpoint']}search/quotes", params=params, timeout=5)
        response.raise_for_status()
        
        return [{
            'content': quote.get('content', ''),
            'author': quote.get('author', ''),
            'source': 'Quotes'
        } for quote in response.json().get('results', [])[:2]]
    except Exception:
        return []

# Function mapping for safe search tools
SAFE_SEARCH_FUNCTIONS = {
    'Wikipedia': safe_search_wikipedia,
    'DuckDuckGo': safe_search_duckduckgo,
    'DuckDuckGo News': safe_search_duckduckgo_news,
    'Wikidata': safe_search_wikidata,
    'ArXiv': safe_search_arxiv,
    'PubMed': safe_search_pubmed,
    'OpenLibrary': safe_search_openlibrary,
    'Dictionary': safe_search_dictionary,
    'Countries': safe_search_countries,
    'Quotes': safe_search_quotes
}

def safe_perform_search(query: str) -> Dict[str, Any]:
    """Perform parallel search safely with timeout."""
    if st.session_state.sandbox_mode:
        return {
            'System': [{
                'title': 'Sandbox Mode Active',
                'summary': 'Running in offline/sandbox mode due to Hugging Face restrictions.',
                'source': 'System'
            }]
        }
    
    results = {}
    selected_sources = list(SAFE_SEARCH_FUNCTIONS.keys())[:4]  # Limit to 4 sources max
    
    with concurrent.futures.ThreadPoolExecutor(max_workers=3) as executor:
        future_to_source = {}
        for source in selected_sources:
            future = executor.submit(SAFE_SEARCH_FUNCTIONS[source], query)
            future_to_source[future] = source
        
        for future in concurrent.futures.as_completed(future_to_source):
            source_name = future_to_source[future]
            try:
                data = future.result(timeout=8)
                if data:
                    results[source_name] = data
            except Exception:
                continue
    
    return results if results else {
        'System': [{
            'title': 'No External Sources Available',
            'summary': 'All search APIs are currently unreachable. Running in offline mode.',
            'source': 'System'
        }]
    }

# Modified main functions to use safe versions
def create_thinking_prompt(query: str, messages: List[Dict], system_prompt: str, 
                          search_results: Dict, sandbox_mode: bool = False) -> str:
    """Create an enhanced prompt that encourages deep thinking."""
    
    # Build search context
    search_context = ""
    if search_results:
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
                        search_context += "\n"
            
            elif isinstance(data, dict):
                for key, value in data.items():
                    if key not in ['source', 'type', 'image'] and value:
                        if isinstance(value, list):
                            search_context += f"{key}: {', '.join(str(v) for v in value[:3])}\n"
                        else:
                            search_context += f"{key}: {value}\n"
                search_context += "\n"
    
    # Add sandbox notice if applicable
    if sandbox_mode:
        search_context += "\n⚠️ NOTE: Running in sandbox/offline mode. External searches are limited.\n"
    
    # Build conversation history
    conversation = ""
    for msg in messages[-3:]:  # Last 3 messages only
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
1. First, analyze the question carefully
2. Consider what you know about the topic
3. Think about different perspectives
4. Formulate a comprehensive yet concise answer
5. Acknowledge limitations if any
6. End with thoughtful insights

IMPORTANT: Be honest about what you know vs. what you don't know.</s>

<|user|>
{query}</s>

<|assistant|>
"""
    
    return prompt

def generate_thoughtful_response(model, prompt: str, max_tokens: int = 512, temperature: float = 0.7) -> str:
    """Generate response with thinking emphasis."""
    try:
        response = model(
            prompt,
            max_new_tokens=max_tokens,
            temperature=temperature,
            top_p=0.9,
            repetition_penalty=1.1,
            stop=["</s>", "<|user|>", "\n\nUser:", "### END"]
        )
        
        return response.strip()
    except Exception as e:
        return f"I encountered an error while generating a response: {str(e)[:100]}. Please try again."

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
        options=["Quick Scan", "Moderate", "Deep Dive"],
        value="Moderate"
    )
    
    temperature = st.slider(
        "Creativity Level:",
        0.1, 1.0, 0.7, 0.1,
        help="Lower = more factual, Higher = more creative"
    )
    
    st.divider()
    
    st.header("🔧 Tools")
    
    auto_search = st.toggle("Auto-Research", value=True)
    show_thinking = st.toggle("Show Thinking Process", value=True)
    
    # Sandbox mode indicator
    if st.session_state.sandbox_mode:
        st.warning("⚠️ Sandbox Mode Active")
        st.caption("External searches limited due to environment restrictions")
    
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
    st.caption("Safe mode for Hugging Face Spaces")

# Main interface
col1, col2 = st.columns([3, 1])
with col1:
    st.title("🧠 DeepThink Pro")
with col2:
    if st.session_state.sandbox_mode:
        st.warning("🔒 Sandbox Mode")

# Display current persona
with st.expander("🤖 Active Persona", expanded=False):
    st.write(st.session_state.selected_preset)
    st.caption(st.session_state.system_prompt[:200] + "...")

# Display sandbox warning if active
if st.session_state.sandbox_mode:
    st.markdown("""
    <div class="warning-box">
    <strong>⚠️ Running in Sandbox Mode</strong><br>
    This Hugging Face Space has limited internet access. Some features like external web searches 
    may not work properly. The AI will respond using its internal knowledge only.
    </div>
    """, unsafe_allow_html=True)

# Load model safely
if st.session_state.model is None:
    with st.spinner("🚀 Initializing AI Brain..."):
        try:
            st.session_state.model = safe_load_model()
            if st.session_state.sandbox_mode:
                st.info("✅ AI Ready (Offline Mode)")
            else:
                st.success("✅ AI Ready for Deep Thinking!")
        except Exception as e:
            st.error(f"❌ Initialization failed: {str(e)[:100]}")
            st.info("Running in basic text generation mode...")

# Display chat
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

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
            3. Preparing response strategy...
            </div>
            """, unsafe_allow_html=True)
        
        # Step 2: Safe Search
        search_results = {}
        
        if auto_search:
            if show_thinking:
                thinking_placeholder.markdown("""
                <div class="thinking-bubble">
                <strong>🔍 Smart Research:</strong><br>
                • Checking available information sources<br>
                • Gathering relevant data safely...
                </div>
                """, unsafe_allow_html=True)
            
            with st.spinner("🔍 Conducting safe research..."):
                search_results = safe_perform_search(prompt)
        
        # Step 3: Generate thoughtful response
        if show_thinking:
            thinking_placeholder.markdown("""
            <div class="thinking-bubble">
            <strong>🤔 Deep Synthesis:</strong><br>
            • Integrating information<br>
            • Applying critical thinking<br>
            • Formulating comprehensive response...
            </div>
            """, unsafe_allow_html=True)
        
        with st.spinner("🧠 Thinking deeply..."):
            # Create enhanced prompt
            enhanced_prompt = create_thinking_prompt(
                prompt, 
                st.session_state.messages,
                st.session_state.system_prompt,
                search_results,
                st.session_state.sandbox_mode
            )
            
            # Adjust tokens based on research depth
            token_map = {
                "Quick Scan": 384,
                "Moderate": 512,
                "Deep Dive": 768
            }
            tokens = token_map.get(research_depth, 512)
            
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
        
        # Store message with metadata
        metadata = {
            "sources": list(search_results.keys()) if search_results else [],
            "thinking_mode": thinking_mode,
            "research_depth": research_depth,
            "sandbox_mode": st.session_state.sandbox_mode
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
        if st.button("Historical figure", use_container_width=True):
            st.session_state.messages.append({"role": "user", "content": "Who was Napoleon Bonaparte?"})
            st.rerun()
        if st.button("Scientific concept", use_container_width=True):
            st.session_state.messages.append({"role": "user", "content": "Explain quantum entanglement"})
            st.rerun()
        if st.button("Philosophical question", use_container_width=True):
            st.session_state.messages.append({"role": "user", "content": "What is consciousness?"})
            st.rerun()
    
    with col2:
        if st.button("Current events", use_container_width=True):
            st.session_state.messages.append({"role": "user", "content": "What are renewable energy sources?"})
            st.rerun()
        if st.button("Creative writing", use_container_width=True):
            st.session_state.messages.append({"role": "user", "content": "Write a short story about a robot learning to paint"})
            st.rerun()
        if st.button("General knowledge", use_container_width=True):
            st.session_state.messages.append({"role": "user", "content": "Tell me about Japan's culture"})
            st.rerun()

# Add footer
st.divider()
st.caption("💡 Tip: This app runs safely on Hugging Face Spaces with limited external access. For full features, run locally.")
