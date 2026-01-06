import streamlit as st
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from newspaper import Article
from collections import Counter


st.set_page_config(
    page_title="FinSentrix",
    layout="wide"
)

MODEL_PATH = "Models/FinBERT_trained_weighted"

@st.cache_resource
def load_model():
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
    model = AutoModelForSequenceClassification.from_pretrained(MODEL_PATH)
    model.eval()
    return tokenizer, model

tokenizer, model = load_model()
id2label = {0: "Negative", 1: "Neutral", 2: "Positive"}

sentiment_to_market = {
    "Positive": ("📈 Bullish", ", Stock Price May Increase 🚀"),
    "Negative": ("📉 Bearish", ", Stock Price May Decrease 👇"),
    "Neutral": ("➖ Sideways", ", No Strong Movement ↕️")
}

STOCK_OPTIONS = {
    # US
    "AAPL": "NASDAQ:AAPL",
    "AMZN": "NASDAQ:AMZN",
    "GOOGL": "NASDAQ:GOOGL",
    "JNJ": "NYSE:JNJ",
    "JPM": "NYSE:JPM",
    "KO": "NYSE:KO",
    "META": "NASDAQ:META",
    "MSFT": "NASDAQ:MSFT",
    "NFLX": "NASDAQ:NFLX",
    "NVDA": "NASDAQ:NVDA",
    "TSLA": "NASDAQ:TSLA",
    "XOM": "NYSE:XOM",
    
    # Malaysia
    "CIMB": "MYX:CIMB",
    "GENTING": "MYX:GENTING",
    "MAXIS": "MYX:MAXIS",
    "MAYBANK": "MYX:MAYBANK",
    "PBBANK": "MYX:PBBANK",
    "PCHEM": "MYX:PCHEM",
    "TM": "MYX:TM",
    "TNB": "MYX:TNB",

    # Others
    "ALIBABA": "NYSE:BABA",
    "NESTLE": "SIX:NESN",
    "SAMSUNG": "KRX:005930",
    "SHELL": "LSE:SHEL",
    "SIEMENS": "XETRA:SIE",
    "TOYOTA": "TSE:7203",
    "TENCENT": "HKEX:0700", 
}

TICKER_TO_TAG = {v: k for k, v in STOCK_OPTIONS.items()}

ENTITY_TO_TICKER = {
    # US Stock
    "google": "NASDAQ:GOOGL",
    "alphabet": "NASDAQ:GOOGL",
    "apple": "NASDAQ:AAPL",
    "microsoft": "NASDAQ:MSFT",
    "amazon": "NASDAQ:AMZN",
    "tesla": "NASDAQ:TSLA",
    "netflix": "NASDAQ:NFLX",
    "nvidia": "NASDAQ:NVDA",
    "meta": "NASDAQ:META",
    "jpmorgan": "NYSE:JPM",
    "coca-cola": "NYSE:KO",
    "exxonmobil": "NYSE:XOM",
    "johnson & johnson": "NYSE:JNJ",
    # Malaysia Stock
    "cimb": "MYX:CIMB",
    "public bank": "MYX:PBBANK",
    "maybank": "MYX:MAYBANK",
    "tenaga nasional": "MYX:TNB",
    "petronas chemicals": "MYX:PCHEM",
    "maxis": "MYX:MAXIS",
    "genting": "MYX:GENTING",
    "telekom malaysia": "MYX:TM",
    # Others
    "toyota": "TSE:7203",
    "samsung": "KRX:005930",
    "alibaba": "NYSE:BABA",
    "tencent": "HKEX:0700",
    "nestle": "SIX:NESN",
    "shell": "LSE:SHEL",
    "siemens": "XETRA:SIE"
}


if "stock_tag" not in st.session_state:
    st.session_state["stock_tag"] = ["AAPL"]
if "auto_detected_stock" not in st.session_state:
    st.session_state["auto_detected_stock"] = None


# FUNCTIONS
@st.cache_data(show_spinner=False)
def extract_article_text(url):
    try:
        article = Article(url)
        article.download()
        article.parse()
        return article.text
    except:
        return None

def detect_stock_from_text(text):
    text_lower = text.lower()
    first_paragraph = text_lower.split("\n")[0]

    # Check First paragraph
    for keyword, ticker in ENTITY_TO_TICKER.items():
        if keyword in first_paragraph:
            return ticker

    # Frequency-based detection
    keyword_counts = {}
    for keyword, ticker in ENTITY_TO_TICKER.items():
        count = text_lower.count(keyword)
        if count > 0:
            keyword_counts[ticker] = keyword_counts.get(ticker, 0) + count

    if not keyword_counts:
        return None

    return max(keyword_counts, key=keyword_counts.get)

def predict_sentiment(text):
    inputs = tokenizer(
        text,
        padding=True,
        truncation=True,
        max_length=128,
        return_tensors="pt"
    )
    with torch.no_grad():
        logits = model(**inputs).logits
        probs = torch.softmax(logits, dim=1).squeeze().tolist()
        pred_id = torch.argmax(logits, dim=1).item()
    return id2label[pred_id], max(probs)

def show_tradingview_chart(ticker):
    st.components.v1.html(
        f"""
        <iframe src="https://s.tradingview.com/widgetembed/?symbol={ticker}&interval=D&theme=light&style=1"
        width="100%" height="500" frameborder="0"></iframe>
        """,
        height=500
    )


# UI
st.image("System_logo/Logo_1.png", width=460)
st.title("📊 Blue-Chip Stock Sentiment Analyzer")
st.write("Analyze financial news sentiment and visualize market trends using **FinBERT**.")
with st.expander("How to use this system"):
    st.write("1. Choose input source (News article or Social media text)")
    st.write("2. Select a blue-chip stock (optional)")
    st.write("3. Provide article URL or paste text")
    st.write("4. Click **Analyze** to view sentiment and market signal")


mode = st.radio(
    "**Choose Input Source**",
    ["📰 News Article", "💬 Text Input (X/Reddit/StockTwits)"],
    horizontal=True
)

selected_stock = st.selectbox(
    "🏷️ **Select Blue-Chip Stock (Optional)**",
    options=list(STOCK_OPTIONS.keys()),
    index=list(STOCK_OPTIONS.keys()).index(
        st.session_state["auto_detected_stock"]
        if st.session_state["auto_detected_stock"]
        else "AAPL"
    )
)
selected_ticker = STOCK_OPTIONS[selected_stock]


if mode == "📰 News Article":
    url = st.text_input("🔗 **Enter Financial News URL**")
    if st.button("🔍 Analyze News"):
        if not url:
            st.error("Please enter a valid URL.")
        else:
            with st.spinner("Extracting article..."):
                article_text = extract_article_text(url)

            if article_text is None or len(article_text.strip()) < 50:
                st.error("Failed to extract article text. Try another URL.")
            else:
                
                auto_ticker = detect_stock_from_text(article_text)
                if auto_ticker:
                    detected_tag = TICKER_TO_TAG.get(auto_ticker)
                    if detected_tag:
                        st.session_state["auto_detected_stock"] = detected_tag
                        selected_ticker = auto_ticker
                        st.success(f"🧠 Auto-detected stock from article: {auto_ticker}")

                
                with st.spinner("Analyzing sentiment with FinBERT..."):
                    sentiment, confidence = predict_sentiment(article_text)
                trend_label, trend_desc = sentiment_to_market[sentiment]

                
                tab1, tab2, tab3 = st.tabs(["🧠 Sentiment", "📈 Stock Chart", "📘 Explanation"])
                with tab1:
                    st.subheader("Sentiment Prediction")
                    color_map = {"Positive": "green", "Negative": "red", "Neutral": "orange"}
                    st.markdown(
                        f"<h2 style='color:{color_map[sentiment]}'>{sentiment}</h2>",
                        unsafe_allow_html=True
                    )

                    confidence_pct = int(confidence * 100)
                    st.markdown(f"**Confidence Level:** {confidence_pct}%")
                    st.progress(confidence)  

                    if confidence >= 0.8:
                        st.success("✅ High confidence — strong sentiment signal")
                    elif confidence >= 0.6:
                        st.warning("⚠ Medium confidence — interpret with care")
                    else:
                        st.error("⚠ Low confidence — interpret with caution")

                    st.subheader("Market Trend Signal")
                    st.markdown(f"### {trend_label} {trend_desc}")

                with tab2:
                    st.subheader("Stock Price Chart")
                    show_tradingview_chart(selected_ticker)

                with tab3:
                    st.markdown("""
                    ### System Explanation
                    - Financial news is extracted from the provided URL
                    - A fine-tuned **FinBERT** model performs sentiment classification with confidence
                    - It is a deep learning model specifically for Financial Sentiment
                    - Stock entities are automatically detected from the article
                    - The detected stock dynamically updates the visualization
                    - Market sentiment is mapped to expected stock movement

                    **Technologies Used**
                    - FinBERT (Transformer-based NLP)
                    - TradingView (Market Visualization)
                    """)

else:
    st.write("💬 Enter post content (X/Reddit/StockTwits)")
    post_content = st.text_area("**Paste your content here**", height=200)
    if st.button("🔍 Analyze Post"):
        if not post_content or len(post_content.strip()) < 5:
            st.error("Please enter some text to analyze.")
        else:
            
            auto_ticker = detect_stock_from_text(post_content)
            if auto_ticker:
                detected_tag = TICKER_TO_TAG.get(auto_ticker)
                if detected_tag:
                    st.session_state["auto_detected_stock"] = detected_tag
                    selected_ticker = auto_ticker
                    st.success(f"🧠 Auto-detected stock from content: {auto_ticker}")

            
            with st.spinner("Analyzing sentiment with FinBERT..."):
                sentiment, confidence = predict_sentiment(post_content)
            trend_label, trend_desc = sentiment_to_market[sentiment]
           
            tab1, tab2, tab3 = st.tabs(["🧠 Sentiment", "📈 Stock Chart", "📘 Explanation"])
            with tab1:
                st.subheader("Sentiment Prediction")
                color_map = {"Positive": "green", "Negative": "red", "Neutral": "orange"}
                st.markdown(
                    f"<h2 style='color:{color_map[sentiment]}'>{sentiment}</h2>",
                    unsafe_allow_html=True
                )
                
                confidence_pct = int(confidence * 100)
                st.markdown(f"**Confidence Level:** {confidence_pct}%")
                st.progress(confidence) 
               
                if confidence >= 0.8:
                    st.success("✅ High confidence — strong sentiment signal")
                elif confidence >= 0.6:
                    st.warning("⚠ Medium confidence — interpret with care")
                else:
                    st.error("⚠ Low confidence — interpret with caution")
               
                st.subheader("Market Trend Signal")
                st.markdown(f"### {trend_label} {trend_desc}")

            with tab2:
                st.subheader("Stock Price Chart")
                show_tradingview_chart(selected_ticker)

            with tab3:
                st.markdown("""
                ### System Explanation
                - User content is analyzed directly
                - Fine-tuned **FinBERT** predicts sentiment with confidence
                - Stock entities are automatically detected from the text
                - Market sentiment is mapped to expected stock movement
                            
                **Technologies Used**
                - FinBERT (Transformer-based NLP)
                - TradingView (Market Visualization)
                """)
                
st.markdown("---")
st.caption("Final Year Project | Financial Sentiment Analysis & Market Trend Prediction")
