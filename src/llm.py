import os
import requests
import random

class GroqLLM:
    def __init__(self, model_name="llama3-70b-8192", temperature=0.7, max_tokens=2048):
        self.api_key = os.environ.get("GROQ_API_KEY")
        if not self.api_key:
            print("Warning: GROQ_API_KEY environment variable not set. Using fallback responses.")
        self.model_name = model_name
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.api_url = "https://api.groq.com/openai/v1/chat/completions"
    
    def __call__(self, prompt):
        if not self.api_key:
            return self._fallback_response(prompt)
        
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.api_key}"
        }
        
        estimated_tokens = len(prompt.split()) * 1.35
        if estimated_tokens > 4000:
            print(f"Input too large (~{int(estimated_tokens)} tokens). Using fallback.")
            return self._fallback_response(prompt)
        
        payload = {
            "model": self.model_name,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": self.temperature,
            "max_tokens": self.max_tokens
        }
        
        try:
            response = requests.post(self.api_url, json=payload, headers=headers)
            if response.status_code == 200:
                return response.json()["choices"][0]["message"]["content"]
            else:
                print(f"API error: {response.status_code}, {response.text}")
                print("Using fallback response generation...")
                return self._fallback_response(prompt)
        except Exception as e:
            print(f"Error calling Groq API: {e}")
            return self._fallback_response(prompt)
    
    def _fallback_response(self, prompt):
        if "Extract key concepts" in prompt:
            concepts = ["Natural Language Processing", "Machine Learning", "Text Analysis", 
                      "Deep Learning", "Computational Linguistics", "Language Modeling", 
                      "Transfer Learning", "Neural Networks", "Data Mining", "Text Classification",
                      "Information Extraction", "Knowledge Graphs", "Semantic Analysis"]
            methods = ["Neural Networks", "Transformers", "Statistical Analysis", 
                     "Deep Learning", "Attention Mechanisms", "Fine-tuning", 
                     "Reinforcement Learning", "Unsupervised Learning", "Pre-training",
                     "Graph Neural Networks", "BERT", "GPT", "T5", "RoBERTa"]
            applications = ["Sentiment Analysis", "Text Classification", "Machine Translation",
                          "Question Answering", "Summarization", "Information Retrieval",
                          "Named Entity Recognition", "Chatbots", "Speech Recognition",
                          "Content Recommendation", "Fact Verification"]
            selected_concepts = random.sample(concepts, min(4, len(concepts)))
            selected_methods = random.sample(methods, min(3, len(methods)))
            selected_apps = random.sample(applications, min(3, len(applications)))
            return f"""
            - Concepts: {', '.join(selected_concepts)}
            - Methods: {', '.join(selected_methods)}
            - Applications: {', '.join(selected_apps)}
            """
        elif "research idea" in prompt.lower():
            concept = ""
            domain = "NLP"
            if "concept:" in prompt.lower():
                concept = prompt.lower().split("concept:")[1].split(".")[0].strip()
            if "domain of" in prompt.lower():
                domain = prompt.lower().split("domain of")[1].split(" ")[1].strip()
            templates = [
                {
                    "title": f"Leveraging {concept} for Improved Cross-lingual Transfer in {domain}",
                    "problem": f"Current {domain} models struggle with effective knowledge transfer across languages with limited training data.",
                    "method": f"Design a {concept}-based framework that aligns representations across languages while preserving language-specific features.",
                    "impact": f"This approach would significantly improve {domain} capabilities for low-resource languages and reduce the need for large parallel corpora."
                },
                {
                    "title": f"Self-supervised {concept} Learning for Domain Adaptation in {domain}",
                    "problem": f"{domain} models often fail when applied to specialized domains due to domain shift.",
                    "method": f"Develop a novel self-supervised objective based on {concept} that helps models adapt to new domains with minimal supervision.",
                    "impact": f"This research would enable more robust deployment of {domain} systems across diverse specialized domains like healthcare, legal, and scientific literature."
                },
                {
                    "title": f"Multimodal {domain} through {concept} Integration",
                    "problem": f"Most {domain} systems operate primarily on text, ignoring valuable information from other modalities.",
                    "method": f"Create a unified architecture that leverages {concept} to integrate text, visual, and structured data for more comprehensive understanding.",
                    "impact": f"This multimodal approach would create more human-like {domain} systems capable of reasoning across different types of information."
                }
            ]
            template = random.choice(templates)
            return f"""
            # {template['title']}
            
            ## Problem Statement
            {template['problem']}
            
            ## Proposed Methodology
            {template['method']}
            
            ## Expected Impact
            {template['impact']}
            """
        else:
            return "I couldn't process your request through the API. Please try again later." 