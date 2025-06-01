import streamlit as st
from literature import retrieve_literature, embed_documents
from knowledge_extraction import extract_knowledge_and_build_graph, extracted_data, build_langchain_graph
from gap_detection import detect_gaps
from idea_generation import generate_research_ideas
from paper_storage import PaperStorage
import os
import json
import base64

# Set page config
st.set_page_config(
    page_title="Research Gap Analysis",
    page_icon="🔍",
    layout="wide"
)

# Initialize session state
if 'current_topic' not in st.session_state:
    st.session_state.current_topic = None
if 'gaps' not in st.session_state:
    st.session_state.gaps = None
if 'ideas' not in st.session_state:
    st.session_state.ideas = None
if 'current_paper' not in st.session_state:
    st.session_state.current_paper = None

# Custom CSS
st.markdown("""
    <style>
    .main {
        padding: 2rem;
    }
    .stButton>button {
        width: 100%;
        margin-top: 1rem;
    }
    .paper-card {
        padding: 1.5rem;
        border-radius: 0.5rem;
        margin-bottom: 1.5rem;
        background-color: #f0f2f6;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }
    .topic-header {
        color: #1f77b4;
        font-size: 1.5rem;
        margin-bottom: 1rem;
        font-weight: bold;
    }
    .gap-card {
        background-color: #e6f3ff;
        padding: 1.5rem;
        border-radius: 0.5rem;
        margin-bottom: 1.5rem;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }
    .idea-card {
        background-color: #f0f7ff;
        padding: 1.5rem;
        border-radius: 0.5rem;
        margin-bottom: 1.5rem;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }
    .paper-title {
        font-size: 1.2rem;
        font-weight: bold;
        color: #1f77b4;
        margin-bottom: 0.5rem;
    }
    .paper-content {
        font-size: 1rem;
        line-height: 1.5;
        color: #333;
    }
    .download-button {
        background-color: #4CAF50;
        color: white;
        padding: 0.5rem 1rem;
        border-radius: 0.3rem;
        text-decoration: none;
        display: inline-block;
        margin-top: 1rem;
    }
    </style>
""", unsafe_allow_html=True)

def get_download_link(paper, filename):
    """Generate a download link for the paper"""
    paper_json = json.dumps(paper, indent=2)
    b64 = base64.b64encode(paper_json.encode()).decode()
    href = f'<a href="data:application/json;base64,{b64}" download="{filename}.json" class="download-button">Download Paper</a>'
    return href

def main():
    st.title("🔍 Research Gap Analysis")
    
    # Sidebar for saved papers
    with st.sidebar:
        st.header("Saved Papers")
        paper_storage = PaperStorage()
        topics = paper_storage.list_saved_topics()
        
        if topics:
            for topic, count in topics.items():
                with st.expander(f"{topic} ({count} papers)"):
                    papers = paper_storage.load_papers(topic)
                    for paper in papers:
                        if st.button(paper['title'], key=f"{topic}_{paper['id']}"):
                            st.session_state.current_topic = topic
                            st.session_state.current_paper = paper
        else:
            st.info("No saved papers yet. Search for papers to get started.")
    
    # Main content area
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.subheader("Enter Research Topic")
        topic = st.text_input("Research Topic", placeholder="e.g., deep learning in healthcare")
        domain = st.text_input("Domain", placeholder="e.g., healthcare, medical imaging")
        
        if st.button("Search Papers"):
            if topic:
                with st.spinner("Retrieving and analyzing papers..."):
                    # Retrieve papers with domain-specific filtering
                    papers = retrieve_literature(topic, domain)
                    
                    if not papers:
                        st.error("No relevant papers found. Try adjusting your search terms.")
                        return
                    
                    st.success(f"Found {len(papers)} relevant papers")
                    
                    # Create embeddings
                    vectorstore, embeddings = embed_documents(papers)
                    
                    # Extract knowledge and build graph
                    graph = extract_knowledge_and_build_graph(papers, domain, embeddings)
                    
                    # Build LangChain graph
                    langchain_graph = build_langchain_graph(papers, extracted_data)
                    
                    # Detect gaps
                    gaps = detect_gaps(graph, langchain_graph)
                    st.session_state.gaps = gaps
                    
                    # Generate ideas
                    ideas = generate_research_ideas(gaps, domain)
                    st.session_state.ideas = ideas
                    
                    # Save papers
                    paper_storage.save_papers(papers, topic)
                    st.session_state.current_topic = topic
            else:
                st.warning("Please enter a research topic")
    
    # Center column for gaps and ideas
    st.markdown("---")
    
    # Display gaps
    if st.session_state.gaps:
        st.subheader("Identified Research Gaps")
        for gap in st.session_state.gaps:
            st.markdown(f"""
                <div class="gap-card">
                    <h4>🔍 {gap}</h4>
                </div>
            """, unsafe_allow_html=True)
    
    # Display ideas
    if st.session_state.ideas:
        st.subheader("Generated Research Ideas")
        for idea in st.session_state.ideas:
            st.markdown(f"""
                <div class="idea-card">
                    <h4>💡 {idea['gap']}</h4>
                    <p>{idea['idea']}</p>
                </div>
            """, unsafe_allow_html=True)
    
    # Display current paper if selected
    if st.session_state.current_paper:
        st.markdown("---")
        st.subheader("Selected Paper")
        paper = st.session_state.current_paper
        
        # Create a card for the paper
        st.markdown(f"""
            <div class="paper-card">
                <div class="paper-title">{paper['title']}</div>
                <div class="paper-content">{paper['content']}</div>
            </div>
        """, unsafe_allow_html=True)
        
        # Add download button
        st.markdown(get_download_link(paper, f"paper_{paper['id']}"), unsafe_allow_html=True)

if __name__ == "__main__":
    main() 