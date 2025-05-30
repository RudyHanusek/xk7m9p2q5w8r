import streamlit as st
import pandas as pd
import requests
from bs4 import BeautifulSoup
import re
import time
import unicodedata
from datetime import datetime
import openai
import anthropic
import google.generativeai as genai
from typing import List, Dict, Tuple

# Konfigurace stránky
st.set_page_config(
    page_title="AI Verification Tool",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS pro lepší vzhled
st.markdown("""
<style>
    .main-header {
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        padding: 2rem;
        border-radius: 10px;
        color: white;
        text-align: center;
        margin-bottom: 2rem;
    }
    .status-card {
        background: #f8f9fa;
        padding: 1rem;
        border-radius: 8px;
        border-left: 4px solid #007bff;
        margin: 1rem 0;
    }
    .success-card {
        background: #d4edda;
        padding: 1rem;
        border-radius: 8px;
        border-left: 4px solid #28a745;
        margin: 1rem 0;
    }
    .error-card {
        background: #f8d7da;
        padding: 1rem;
        border-radius: 8px;
        border-left: 4px solid #dc3545;
        margin: 1rem 0;
    }
</style>
""", unsafe_allow_html=True)

# Hlavičku aplikace
st.markdown("""
<div class="main-header">
    <h1>🔍 AI Verification Tool</h1>
    <p>Ověření zobrazení značky v AI - Komplexní analýza</p>
</div>
""", unsafe_allow_html=True)

# Sidebar s návodem
with st.sidebar:
    st.header("📋 Návod použití")
    st.markdown("""
    1. **Zadejte údaje** o značce a doméně
    2. **Klikněte** na "Spustit analýzu"  
    3. **Počkejte** na dokončení (několik minut)
    4. **Prohlédněte** si výsledky
    5. **Stáhněte** data jako CSV
    """)
    
    st.markdown("---")
    st.info("🔧 API klíče jsou předkonfigurovány")

# Hlavní formulář
col1, col2 = st.columns(2)

with col1:
    brand = st.text_input("🏷️ Brand", value="taste", help="Název značky k hledání")
    domena = st.text_input("🌐 Doména", value="taste.cz", help="Webová doména (bez https://)")

with col2:
    zeme = st.selectbox(
        "🌍 Země",
        ["Česká republika", "Slovensko", "Polsko", "Německo", "Rakousko", "Maďarsko"],
        help="Země pro doporučení společností"
    )

# Funkce pro web scraping
def scrape_website(domain: str) -> str:
    """Stáhne a zpracuje obsah webové stránky"""
    urls_to_try = [f"https://{domain}", f"http://{domain}"]
    
    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
    }
    
    for url in urls_to_try:
        try:
            response = requests.get(url, headers=headers, timeout=30, allow_redirects=True)
            if response.status_code == 200:
                soup = BeautifulSoup(response.content, 'html.parser')
                
                # Odstranění nepotřebných elementů
                for script in soup(["script", "style", "nav", "footer", "header"]):
                    script.decompose()
                
                # Extrakce textu
                text = soup.get_text(separator='\n', strip=True)
                text = re.sub(r'\s+', ' ', text)
                text = re.sub(r'\n+', '\n', text)
                
                # Omezení délky
                MAX_LENGTH = 15000
                if len(text) > MAX_LENGTH:
                    text = text[:MAX_LENGTH] + "... (text byl zkrácen)"
                
                return text
                
        except Exception as e:
            st.error(f"Chyba při načítání {url}: {str(e)}")
            continue
    
    return f"CHYBA: Nepodařilo se načíst stránku {domain}"

# AI API funkce
def query_openai(prompt: str, api_key: str) -> str:
    """Dotaz na OpenAI ChatGPT"""
    try:
        client = openai.OpenAI(api_key=api_key)
        response = client.chat.completions.create(
            model="gpt-4",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.7,
            max_tokens=2048
        )
        return response.choices[0].message.content
    except Exception as e:
        return f"❌ Chyba: {str(e)}"

def query_claude(prompt: str, api_key: str) -> str:
    """Dotaz na Claude AI"""
    try:
        client = anthropic.Anthropic(api_key=api_key)
        response = client.messages.create(
            model="claude-3-opus-20240229",
            max_tokens=2048,
            temperature=0.7,
            messages=[{"role": "user", "content": prompt}]
        )
        return response.content[0].text
    except Exception as e:
        return f"❌ Chyba: {str(e)}"

def query_gemini(prompt: str, api_key: str) -> str:
    """Dotaz na Google Gemini"""
    try:
        genai.configure(api_key=api_key)
        model = genai.GenerativeModel('gemini-1.5-pro-latest')
        response = model.generate_content(prompt)
        return response.text
    except Exception as e:
        return f"❌ Chyba: {str(e)}"

def extract_business_areas(content: str, api_key: str) -> List[str]:
    """Extrakce oblastí podnikání z obsahu webu pomocí AI"""
    prompt = f"""Analyzuj následující text z webové stránky a identifikuj maximálně 5 klíčových oblastí podnikání nebo témat, kterým se tato stránka/společnost věnuje. Každou oblast uveď na nový řádek. Pokud nelze identifikovat žádné smysluplné oblasti, odpověz "Nebylo možné identifikovat oblasti".

Text:
{content}"""
    
    response = query_gemini(prompt, api_key)
    
    if "❌ Chyba:" in response:
        return [f"Chyba AI při generování oblastí: {response}"]
    
    if "nebylo možné identifikovat" in response.lower():
        return ["Nebylo možné identifikovat oblasti z textu"]
    
    areas = [area.strip() for area in response.split('\n') if area.strip()]
    return areas[:5]  # Maximálně 5 oblastí

def intelligent_check(text: str, search_term: str) -> str:
    """Inteligentní kontrola výskytu termínu v textu"""
    if not text or not search_term:
        return "N/A"
    
    # Normalizace textu (odstranění diakritiky)
    normalized_text = unicodedata.normalize('NFD', text.lower())
    normalized_text = ''.join(c for c in normalized_text if unicodedata.category(c) != 'Mn')
    
    normalized_search = unicodedata.normalize('NFD', search_term.lower())
    normalized_search = ''.join(c for c in normalized_search if unicodedata.category(c) != 'Mn')
    
    return "✅ Ano" if normalized_search in normalized_text else "❌ Ne"

# Načtení API klíčů ze secrets
try:
    openai_key = st.secrets["api_keys"]["openai"]
    anthropic_key = st.secrets["api_keys"]["anthropic"] 
    gemini_key = st.secrets["api_keys"]["gemini"]
except KeyError as e:
    st.error(f"❌ Chybějící API klíč v konfiguraci: {e}")
    st.stop()

# Hlavní spouštěcí tlačítko
if st.button("🚀 Spustit analýzu", type="primary", use_container_width=True):
    
    if not domena:
        st.error("⚠️ Vyplňte prosím doménu!")
        st.stop()
    
    # Inicializace session state pro výsledky
    if 'analysis_results' not in st.session_state:
        st.session_state.analysis_results = {}
    
    # Progress tracking
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    # Krok 1: Web scraping
    status_text.markdown('<div class="status-card">🕸️ <strong>Krok 1/4:</strong> Stahuji obsah webu...</div>', unsafe_allow_html=True)
    progress_bar.progress(10)
    
    website_content = scrape_website(domena)
    
    if website_content.startswith("CHYBA:"):
        st.markdown(f'<div class="error-card">{website_content}</div>', unsafe_allow_html=True)
        st.stop()
    
    # Krok 2: Extrakce oblastí
    status_text.markdown('<div class="status-card">🧠 <strong>Krok 2/4:</strong> Analyzuji oblasti podnikání...</div>', unsafe_allow_html=True)
    progress_bar.progress(25)
    
    business_areas = extract_business_areas(website_content, gemini_key)
    
    if not business_areas or (len(business_areas) == 1 and ("Chyba" in business_areas[0] or "nebylo možné" in business_areas[0].lower())):
        st.markdown('<div class="error-card">❌ Nepodařilo se identifikovat oblasti podnikání</div>', unsafe_allow_html=True)
        st.write("Extrahované oblasti:", business_areas)
        st.stop()
    
    st.markdown('<div class="success-card">✅ <strong>Identifikované oblasti:</strong><br>' + '<br>'.join([f"• {area}" for area in business_areas]) + '</div>', unsafe_allow_html=True)
    
    # Krok 3: Dotazování AI modelů
    status_text.markdown('<div class="status-card">🤖 <strong>Krok 3/4:</strong> Dotazuji AI modely...</div>', unsafe_allow_html=True)
    
    all_responses = []
    analysis_results = []
    
    total_queries = len(business_areas) * 3
    current_query = 0
    
    for area in business_areas:
        clean_area = area.strip()
        if not clean_area or "Chyba" in clean_area or "nebylo možné" in clean_area.lower():
            continue
        
        query = f'Jaké společnosti z oblasti "{clean_area}" doporučuješ v zemi "{zeme}"?'
        
        # ChatGPT
        current_query += 1
        progress_bar.progress(25 + (current_query / total_queries) * 50)
        status_text.markdown(f'<div class="status-card">🤖 Dotazuji ChatGPT ({current_query}/{total_queries})...</div>', unsafe_allow_html=True)
        
        gpt_response = query_openai(query, openai_key)
        all_responses.append({
            "Dotaz + AI": f"{query} (ChatGPT)",
            "Odpověď AI": gpt_response
        })
        
        # Claude AI
        current_query += 1
        progress_bar.progress(25 + (current_query / total_queries) * 50)
        status_text.markdown(f'<div class="status-card">🧠 Dotazuji Claude AI ({current_query}/{total_queries})...</div>', unsafe_allow_html=True)
        
        claude_response = query_claude(query, anthropic_key)
        all_responses.append({
            "Dotaz + AI": f"{query} (Claude AI)",
            "Odpověď AI": claude_response
        })
        
        # Gemini
        current_query += 1
        progress_bar.progress(25 + (current_query / total_queries) * 50)
        status_text.markdown(f'<div class="status-card">✨ Dotazuji Gemini ({current_query}/{total_queries})...</div>', unsafe_allow_html=True)
        
        gemini_response = query_gemini(query, gemini_key)
        all_responses.append({
            "Dotaz + AI": f"{query} (Gemini)",
            "Odpověď AI": gemini_response
        })
        
        # Analýza zmínek pro každé AI
        for ai_name, response in [("ChatGPT", gpt_response), ("Claude AI", claude_response), ("Gemini", gemini_response)]:
            brand_match = intelligent_check(response, brand)
            domain_match = intelligent_check(response, domena)
            
            analysis_results.append({
                "Oblast": clean_area,
                "AI": ai_name,
                "Brand": brand_match,
                "Doména": domain_match
            })
        
        time.sleep(1)  # Rate limiting
    
    # Krok 4: Finalizace
    status_text.markdown('<div class="status-card">📊 <strong>Krok 4/4:</strong> Finalizuji výsledky...</div>', unsafe_allow_html=True)
    progress_bar.progress(100)
    
    # Uložení do session state
    st.session_state.analysis_results = {
        'responses': all_responses,
        'analysis': analysis_results,
        'metadata': {
            'brand': brand,
            'domain': domena,
            'country': zeme,
            'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            'areas_found': len(business_areas)
        }
    }
    
    status_text.markdown('<div class="success-card">✅ <strong>Analýza dokončena!</strong></div>', unsafe_allow_html=True)
    time.sleep(1)
    progress_bar.empty()
    status_text.empty()

# Zobrazení výsledků (pokud existují)
if 'analysis_results' in st.session_state and st.session_state.analysis_results:
    results = st.session_state.analysis_results
    
    st.markdown("---")
    st.header("📊 Výsledky analýzy")
    
    # Metadata
    metadata = results['metadata']
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Brand", metadata['brand'])
    with col2:
        st.metric("Doména", metadata['domain'])
    with col3:
        st.metric("Země", metadata['country'])
    with col4:
        st.metric("Nalezené oblasti", metadata['areas_found'])
    
    # Tabs pro výsledky
    tab1, tab2, tab3 = st.tabs(["📋 Souhrn odpovědí AI", "📊 Analýza zmínek", "📈 Statistiky"])
    
    with tab1:
        st.subheader("Všechny odpovědi AI modelů")
        responses_df = pd.DataFrame(results['responses'])
        st.dataframe(responses_df, use_container_width=True, height=600)
        
        # CSV download pro odpovědi
        csv_responses = responses_df.to_csv(index=False, encoding='utf-8')
        st.download_button(
            label="💾 Stáhnout odpovědi jako CSV",
            data=csv_responses,
            file_name=f"ai_responses_{metadata['brand']}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
            mime="text/csv"
        )
    
    with tab2:
        st.subheader("Analýza zmínek značky a domény")
        analysis_df = pd.DataFrame(results['analysis'])
        
        # Barevné formátování
        def highlight_results(val):
            if val == "✅ Ano":
                return 'background-color: #d4edda; color: #155724'
            elif val == "❌ Ne":
                return 'background-color: #f8d7da; color: #721c24'
            return ''
        
        styled_df = analysis_df.style.applymap(highlight_results, subset=['Brand', 'Doména'])
        st.dataframe(styled_df, use_container_width=True)
        
        # CSV download pro analýzu
        csv_analysis = analysis_df.to_csv(index=False, encoding='utf-8')
        st.download_button(
            label="💾 Stáhnout analýzu jako CSV",
            data=csv_analysis,
            file_name=f"ai_analysis_{metadata['brand']}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
            mime="text/csv"
        )
    
    with tab3:
        st.subheader("Statistiky zmínek")
        
        # Statistiky pro brand
        brand_stats = analysis_df['Brand'].value_counts()
        col1, col2 = st.columns(2)
        
        with col1:
            st.write("**Zmínky brandu:**")
            if "✅ Ano" in brand_stats:
                st.success(f"✅ Zmíněn: {brand_stats['✅ Ano']}x")
            if "❌ Ne" in brand_stats:
                st.error(f"❌ Nezmíněn: {brand_stats['❌ Ne']}x")
        
        with col2:
            st.write("**Zmínky domény:**")
            domain_stats = analysis_df['Doména'].value_counts()
            if "✅ Ano" in domain_stats:
                st.success(f"✅ Zmíněna: {domain_stats['✅ Ano']}x")
            if "❌ Ne" in domain_stats:
                st.error(f"❌ Nezmíněna: {domain_stats['❌ Ne']}x")
        
        # Statistiky po AI
        st.write("**Úspěšnost podle AI modelů:**")
        ai_stats = analysis_df.groupby('AI').agg({
            'Brand': lambda x: (x == "✅ Ano").sum(),
            'Doména': lambda x: (x == "✅ Ano").sum()
        })
        ai_stats.columns = ['Brand zmínky', 'Doména zmínky']
        st.dataframe(ai_stats)



# Footer
st.markdown("---")
st.markdown("*🔍 AI Verification Tool - Vytvořeno pro analýzu zmínek značky v AI odpovědích*")
