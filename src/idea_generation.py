from typing import List, Dict, Any

def generate_research_ideas(gaps: List[str], domain: str):
    """Generate research ideas based on detected gaps"""
    ideas = []
    from llm import GroqLLM
    llm = GroqLLM(model_name="llama3-70b-8192", temperature=0.7, max_tokens=2048)
    for gap in gaps:
        prompt = f"""Generate a novel research idea in the domain of {domain} using the concept: {gap}.
        Include:
        1. A clear title for the research
        2. Brief problem statement
        3. Proposed methodology
        4. Expected impact
        
        Be specific and focused.
        """
        idea = llm(prompt)
        print("\n\U0001F50D Research Gap:", gap)
        print("\U0001F4A1 Research Idea:", idea)
        ideas.append({"gap": gap, "idea": idea})
    return ideas 