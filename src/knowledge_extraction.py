import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from typing import List, Dict, Any

# Global variable to store extracted data
extracted_data = {
    "concepts": set(),
    "methods": set(),
    "applications": set(),
    "domains": set()
}

def calculate_similarity(concept1, concept2, embeddings=None):
    """Calculate semantic similarity between two concepts"""
    if embeddings is None:
        c1 = concept1.lower()
        c2 = concept2.lower()
        words1 = set(c1.split())
        words2 = set(c2.split())
        if not words1 or not words2:
            return 0
        intersection = len(words1.intersection(words2))
        union = len(words1.union(words2))
        return intersection / union if union > 0 else 0
    else:
        try:
            emb1 = embeddings.embed_query(concept1)
            emb2 = embeddings.embed_query(concept2)
            emb1_np = np.array(emb1).reshape(1, -1)
            emb2_np = np.array(emb2).reshape(1, -1)
            sim = cosine_similarity(emb1_np, emb2_np)[0][0]
            return float(sim)
        except Exception as e:
            print(f"Error calculating similarity with embeddings: {e}")
            return calculate_similarity(concept1, concept2)

def are_related(concept, method, docs, window_size=100):
    """Check if concept and method are related in the documents"""
    concept_lower = concept.lower()
    method_lower = method.lower()
    for doc in docs:
        content = doc.page_content.lower()
        if concept_lower in content and method_lower in content:
            concept_pos = content.find(concept_lower)
            method_pos = content.find(method_lower)
            if abs(concept_pos - method_pos) < window_size:
                return True
    return False

def extract_knowledge_and_build_graph(docs, domain_input, embeddings=None):
    global extracted_data
    neo4j_graph = None
    try:
        from neo4j_graph import EnhancedNeo4jGraph
        neo4j_graph = EnhancedNeo4jGraph(
            url="bolt://localhost:7687",
            username="neo4j",
            password="langchain123"  # replace with your actual password
        )
    except Exception as e:
        print(f"Could not connect to Neo4j: {e}")
    graph_data = {
        "nodes": [],
        "relationships": []
    }
    domains = [d.strip() for d in domain_input.split(',')]
    for domain in domains:
        extracted_data["domains"].add(domain)
        if neo4j_graph and neo4j_graph.is_connected:
            neo4j_graph.create_node("Domain", {"name": domain})
        else:
            graph_data["nodes"].append({"label": "Domain", "properties": {"name": domain}})
    paper_concepts = {}
    for idx, doc in enumerate(docs):
        paper_id = doc.metadata.get("id", f"paper_{idx}")
        paper_title = doc.metadata.get("title", f"Untitled Paper {idx}")
        if neo4j_graph and neo4j_graph.is_connected:
            neo4j_graph.create_node("Paper", {"id": paper_id, "title": paper_title})
        else:
            graph_data["nodes"].append({"label": "Paper", "properties": {"id": paper_id, "title": paper_title}})
        content = doc.page_content
        if len(content) > 5000:
            content = content[:4800] + "..."
        from llm import GroqLLM
        llm = GroqLLM(model_name="llama3-70b-8192", temperature=0.7, max_tokens=2048)
        llm_prompt = f"""
        Extract key concepts, methods, and applications from this abstract:

        {content}

        Output format:
        - Concepts: [list of key concepts separated by commas]
        - Methods: [list of methodologies separated by commas]
        - Applications: [list of applications separated by commas]
        """
        response = llm(llm_prompt)
        print(f"\nExtracted from document {idx+1} ({paper_title}):", response)
        concepts = []
        methods = []
        applications = []
        for line in response.split('\n'):
            line = line.strip()
            if line.startswith("- Concepts:"):
                concepts = [c.strip() for c in line.replace("- Concepts:", "").split(",") if c.strip()]
                extracted_data["concepts"].update(concepts)
                paper_concepts[paper_id] = concepts
            elif line.startswith("- Methods:"):
                methods = [m.strip() for m in line.replace("- Methods:", "").split(",") if m.strip()]
                extracted_data["methods"].update(methods)
            elif line.startswith("- Applications:"):
                applications = [a.strip() for a in line.replace("- Applications:", "").split(",") if a.strip()]
                extracted_data["applications"].update(applications)
        if neo4j_graph and neo4j_graph.is_connected:
            for concept in concepts:
                neo4j_graph.create_node("Concept", {"name": concept})
                neo4j_graph.create_relationship(
                    "Concept", {"name": concept},
                    "APPEARS_IN",
                    "Paper", {"id": paper_id},
                    {"confidence": 1.0}
                )
                for domain in domains:
                    neo4j_graph.create_relationship(
                        "Concept", {"name": concept},
                        "BELONGS_TO",
                        "Domain", {"name": domain},
                        {"relevance": 0.8}
                    )
            for method in methods:
                neo4j_graph.create_node("Method", {"name": method})
                neo4j_graph.create_relationship(
                    "Method", {"name": method},
                    "USED_IN",
                    "Paper", {"id": paper_id},
                    {"confidence": 1.0}
                )
            for app in applications:
                neo4j_graph.create_node("Application", {"name": app})
                neo4j_graph.create_relationship(
                    "Application", {"name": app},
                    "DESCRIBED_IN",
                    "Paper", {"id": paper_id},
                    {"confidence": 1.0}
                )
                for concept in concepts:
                    neo4j_graph.create_relationship(
                        "Application", {"name": app},
                        "APPLIES",
                        "Concept", {"name": concept},
                        {"strength": 0.9}
                    )
                for method in methods:
                    neo4j_graph.create_relationship(
                        "Application", {"name": app},
                        "USES",
                        "Method", {"name": method},
                        {"strength": 0.9}
                    )
            for concept in concepts:
                for method in methods:
                    neo4j_graph.create_relationship(
                        "Concept", {"name": concept},
                        "IMPLEMENTED_BY",
                        "Method", {"name": method},
                        {"confidence": 0.8}
                    )
        else:
            for concept in concepts:
                graph_data["nodes"].append({"label": "Concept", "properties": {"name": concept}})
                graph_data["relationships"].append({
                    "source": concept,
                    "target": paper_id,
                    "type": "APPEARS_IN"
                })
                for domain in domains:
                    graph_data["relationships"].append({
                        "source": concept,
                        "target": domain,
                        "type": "BELONGS_TO"
                    })
            for method in methods:
                graph_data["nodes"].append({"label": "Method", "properties": {"name": method}})
                graph_data["relationships"].append({
                    "source": method,
                    "target": paper_id,
                    "type": "USED_IN"
                })
            for app in applications:
                graph_data["nodes"].append({"label": "Application", "properties": {"name": app}})
                graph_data["relationships"].append({
                    "source": app,
                    "target": paper_id,
                    "type": "DESCRIBED_IN"
                })
                for concept in concepts:
                    graph_data["relationships"].append({
                        "source": app,
                        "target": concept,
                        "type": "APPLIES"
                    })
                for method in methods:
                    graph_data["relationships"].append({
                        "source": app,
                        "target": method,
                        "type": "USES"
                    })
            for concept in concepts:
                for method in methods:
                    graph_data["relationships"].append({
                        "source": concept,
                        "target": method,
                        "type": "IMPLEMENTED_BY"
                    })
    if neo4j_graph and neo4j_graph.is_connected:
        concepts_list = list(extracted_data["concepts"])
        for i, concept1 in enumerate(concepts_list):
            for concept2 in concepts_list[i+1:]:
                similarity = calculate_similarity(concept1, concept2, embeddings)
                if similarity > 0.6:
                    neo4j_graph.create_relationship(
                        "Concept", {"name": concept1},
                        "RELATED_TO",
                        "Concept", {"name": concept2},
                        {"similarity": similarity}
                    )
    else:
        concepts_list = list(extracted_data["concepts"])
        for i, concept1 in enumerate(concepts_list):
            for concept2 in concepts_list[i+1:]:
                similarity = calculate_similarity(concept1, concept2, embeddings)
                if similarity > 0.6:
                    graph_data["relationships"].append({
                        "source": concept1,
                        "target": concept2,
                        "type": "RELATED_TO",
                        "properties": {"similarity": similarity}
                    })
    concept_pairs = set()
    for paper_id, concepts in paper_concepts.items():
        for i, concept1 in enumerate(concepts):
            for concept2 in concepts[i+1:]:
                if (concept1, concept2) not in concept_pairs and (concept2, concept1) not in concept_pairs:
                    concept_pairs.add((concept1, concept2))
                    if neo4j_graph and neo4j_graph.is_connected:
                        neo4j_graph.create_relationship(
                            "Concept", {"name": concept1},
                            "CO_OCCURS_WITH",
                            "Concept", {"name": concept2},
                            {"count": 1}
                        )
                    else:
                        graph_data["relationships"].append({
                            "source": concept1,
                            "target": concept2,
                            "type": "CO_OCCURS_WITH",
                            "properties": {"count": 1}
                        })
    if neo4j_graph and neo4j_graph.is_connected:
        print(f"\nGraph Statistics:")
        print(f"- Concepts: {neo4j_graph.get_node_count('Concept')}")
        print(f"- Methods: {neo4j_graph.get_node_count('Method')}")
        print(f"- Applications: {neo4j_graph.get_node_count('Application')}")
        print(f"- Papers: {neo4j_graph.get_node_count('Paper')}")
        print(f"- Total relationships: {neo4j_graph.get_relationship_count()}")
        return neo4j_graph
    else:
        print(f"\nIn-Memory Graph Statistics:")
        print(f"- Concepts: {len([n for n in graph_data['nodes'] if n.get('label') == 'Concept'])}")
        print(f"- Methods: {len([n for n in graph_data['nodes'] if n.get('label') == 'Method'])}")
        print(f"- Applications: {len([n for n in graph_data['nodes'] if n.get('label') == 'Application'])}")
        print(f"- Papers: {len([n for n in graph_data['nodes'] if n.get('label') == 'Paper'])}")
        print(f"- Total relationships: {len(graph_data['relationships'])}")
        return graph_data

def build_langchain_graph(docs, extracted_data):
    """Build a LangChain knowledge graph from extracted data"""
    nodes = []
    edges = []
    
    # Add nodes for concepts, methods, applications, domains, papers
    for concept in extracted_data["concepts"]:
        nodes.append({"id": f"concept-{concept}", "type": "Concept", "properties": {"name": concept}})
    
    for method in extracted_data["methods"]:
        nodes.append({"id": f"method-{method}", "type": "Method", "properties": {"name": method}})
    
    for app in extracted_data["applications"]:
        nodes.append({"id": f"app-{app}", "type": "Application", "properties": {"name": app}})
    
    for domain in extracted_data["domains"]:
        nodes.append({"id": f"domain-{domain}", "type": "Domain", "properties": {"name": domain}})
    
    for i, doc in enumerate(docs):
        paper_id = doc.metadata.get("id", f"paper_{i}")
        paper_title = doc.metadata.get("title", f"Untitled Paper {i}")
        nodes.append({"id": f"paper-{paper_id}", "type": "Paper", "properties": {"id": paper_id, "title": paper_title}})
    
    # Add edges based on relationship heuristics
    # For concept-method relationships, check co-occurrence in documents
    for concept in extracted_data["concepts"]:
        for method in extracted_data["methods"]:
            if are_related(concept, method, docs):
                edges.append({
                    "source": f"concept-{concept}",
                    "target": f"method-{method}",
                    "type": "IMPLEMENTED_BY"
                })
    
    # For concept-concept relationships, check similarity
    concepts = list(extracted_data["concepts"])
    for i, concept1 in enumerate(concepts):
        for concept2 in concepts[i+1:]:
            similarity = calculate_similarity(concept1, concept2)
            if similarity > 0.6:  # Threshold
                edges.append({
                    "source": f"concept-{concept1}",
                    "target": f"concept-{concept2}",
                    "type": "RELATED_TO",
                    "properties": {"similarity": similarity}
                })
    
    # For application-method relationships
    for app in extracted_data["applications"]:
        for method in extracted_data["methods"]:
            if are_related(app, method, docs):
                edges.append({
                    "source": f"app-{app}",
                    "target": f"method-{method}",
                    "type": "USES"
                })
    
    # For concept-domain relationships
    for concept in extracted_data["concepts"]:
        for domain in extracted_data["domains"]:
            edges.append({
                "source": f"concept-{concept}",
                "target": f"domain-{domain}",
                "type": "BELONGS_TO"
            })
    
    # Create LangChain knowledge graph
    knowledge_graph = {"nodes": nodes, "edges": edges}
    return knowledge_graph 