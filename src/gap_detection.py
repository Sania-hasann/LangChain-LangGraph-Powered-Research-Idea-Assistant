from typing import List, Dict, Any
from knowledge_extraction import extracted_data
from llm import GroqLLM

def analyze_langchain_graph(graph):
    """Analyze LangChain knowledge graph to find research gaps"""
    if not graph:
        return []
    
    gaps = []
    
    # Analyze node connectivity
    node_connectivity = {}
    for edge in graph['edges']:
        source = edge["source"]
        target = edge["target"]
        
        # Count connections for each node
        if source not in node_connectivity:
            node_connectivity[source] = 0
        if target not in node_connectivity:
            node_connectivity[target] = 0
        
        node_connectivity[source] += 1
        node_connectivity[target] += 1
    
    # Find concepts with few connections
    isolated_concepts = []
    for node in graph['nodes']:
        node_id = node["id"]
        if node["type"] == "Concept" and node_connectivity.get(node_id, 0) < 2:
            isolated_concepts.append(node["properties"]["name"])
    
    # Limit to top 3 isolated concepts
    if isolated_concepts:
        gaps.extend(isolated_concepts[:3])
    
    # Find methods not connected to applications
    unused_methods = []
    method_nodes = [node for node in graph['nodes'] if node["type"] == "Method"]
    for method_node in method_nodes:
        method_id = method_node["id"]
        method_name = method_node["properties"]["name"]
        
        # Check if method is used by any application
        is_used = False
        for edge in graph['edges']:
            if edge["target"] == method_id and edge["type"] == "USES":
                is_used = True
                break
        
        if not is_used:
            unused_methods.append(method_name)
    
    # Add top 2 unused methods to gaps
    if unused_methods:
        gaps.extend(unused_methods[:2])
    
    return gaps

def detect_gaps_neo4j(graph):
    """Detect research gaps using Neo4j graph analysis"""
    if not hasattr(graph, 'query'):
        return []
    
    gaps = []
    
    try:
        # 1. Find concepts with few connections
        isolated_concepts_query = """
        MATCH (c:Concept)
        OPTIONAL MATCH (c)-[r]-()
        WITH c, count(r) as rel_count
        WHERE rel_count < 3
        RETURN c.name AS gap_topic, rel_count
        ORDER BY rel_count
        LIMIT 3
        """
        
        # 2. Find concepts rarely used in applications
        underutilized_concepts_query = """
        MATCH (c:Concept)
        OPTIONAL MATCH (a:Application)-[:APPLIES]->(c)
        WITH c, count(a) as app_count
        WHERE app_count < 2
        RETURN c.name AS gap_topic, app_count
        ORDER BY app_count
        LIMIT 3
        """
        
        # 3. Find method pairs that are never used together
        method_combination_query = """
        MATCH (m1:Method), (m2:Method)
        WHERE m1.name < m2.name
        OPTIONAL MATCH (a:Application)-[:USES]->(m1), (a)-[:USES]->(m2)
        WITH m1, m2, count(a) as combo_count
        WHERE combo_count = 0
        RETURN m1.name + ' with ' + m2.name AS gap_topic
        LIMIT 3
        """
        
        # 4. Find concepts with high mutual information but no direct relationship
        related_concepts_query = """
        MATCH (c1:Concept), (c2:Concept)
        WHERE c1.name < c2.name
        OPTIONAL MATCH (c1)-[r:RELATED_TO|CO_OCCURS_WITH]->(c2)
        WITH c1, c2, count(r) as rel_count
        MATCH (p1:Paper)<-[:APPEARS_IN]-(c1), (p2:Paper)<-[:APPEARS_IN]-(c2)
        WHERE p1 = p2
        WITH c1, c2, rel_count, count(p1) as co_papers
        WHERE rel_count = 0 AND co_papers > 0
        RETURN c1.name + ' integrated with ' + c2.name AS gap_topic, co_papers
        ORDER BY co_papers DESC
        LIMIT 3
        """
        
        # 5. Find methods not used with specific concepts (fixed query)
        concept_method_query = """
        MATCH (c:Concept), (m:Method)
        WHERE NOT (c)-[:IMPLEMENTED_BY]->(m)
        AND EXISTS((c)<-[:APPLIES]-())
        AND EXISTS((m)<-[:USES]-())
        RETURN c.name + ' using ' + m.name AS gap_topic
        LIMIT 3
        """
        
        # Execute queries and collect results
        all_queries = [
            isolated_concepts_query,
            underutilized_concepts_query,
            method_combination_query,
            related_concepts_query,
            concept_method_query
        ]
        
        for query in all_queries:
            results = graph.query(query)
            for record in results:
                if "gap_topic" in record and record["gap_topic"]:
                    gaps.append(record["gap_topic"])
        
        return gaps
    except Exception as e:
        print(f"Error detecting gaps with Neo4j: {e}")
        return []

def detect_gaps_in_memory(graph_data):
    """Detect research gaps using in-memory graph analysis"""
    if not isinstance(graph_data, dict):
        return []
    gaps = []
    concept_connections = {}
    for node in graph_data["nodes"]:
        if node.get("label") == "Concept":
            concept_name = node["properties"]["name"]
            concept_connections[concept_name] = 0
    for rel in graph_data["relationships"]:
        source = rel["source"]
        target = rel["target"]
        if source in concept_connections:
            concept_connections[source] += 1
        if target in concept_connections:
            concept_connections[target] += 1
    isolated_concepts = sorted(concept_connections.items(), key=lambda x: x[1])[:3]
    for concept, count in isolated_concepts:
        if count < 3:
            gaps.append(concept)
    method_used = {}
    for node in graph_data["nodes"]:
        if node.get("label") == "Method":
            method_name = node["properties"]["name"]
            method_used[method_name] = False
    for rel in graph_data["relationships"]:
        if rel["type"] == "USES" and rel["target"] in method_used:
            method_used[rel["target"]] = True
    for method, used in method_used.items():
        if not used:
            gaps.append(f"{method} applications")
    concept_pairs = set()
    for rel in graph_data["relationships"]:
        if rel["type"] in ["RELATED_TO", "CO_OCCURS_WITH"]:
            pair = tuple(sorted([rel["source"], rel["target"]]))
            concept_pairs.add(pair)
    concepts_by_paper = {}
    for rel in graph_data["relationships"]:
        if rel["type"] == "APPEARS_IN":
            concept = rel["source"]
            paper = rel["target"]
            if paper not in concepts_by_paper:
                concepts_by_paper[paper] = []
            concepts_by_paper[paper].append(concept)
    potential_pairs = []
    for paper, concepts in concepts_by_paper.items():
        for i, c1 in enumerate(concepts):
            for c2 in concepts[i+1:]:
                pair = tuple(sorted([c1, c2]))
                if pair not in concept_pairs:
                    potential_pairs.append(pair)
    pair_counts = {}
    for pair in potential_pairs:
        if pair not in pair_counts:
            pair_counts[pair] = 0
        pair_counts[pair] += 1
    top_pairs = sorted(pair_counts.items(), key=lambda x: x[1], reverse=True)[:2]
    for (c1, c2), count in top_pairs:
        gaps.append(f"{c1} integrated with {c2}")
    return gaps

def find_topic_gaps(topics, extracted_data):
    """Identify potential research gaps by comparing topic model terms with extracted concepts"""
    if not topics:
        return []
        
    # Extract terms from topics
    topic_terms = set()
    for topic_id, topic in topics.items():
        # Extract terms from topic
        try:
            # Format: "0.029*"term1" + 0.028*"term2" + ..."
            terms = []
            for term_part in topic.split('+'):
                parts = term_part.strip().split('*')
                if len(parts) >= 2:
                    term = parts[1].strip().replace('"', '')
                    terms.append(term)
            topic_terms.update(terms)
        except Exception as e:
            print(f"Error parsing topic: {e}")
    
    # Find terms in topics not covered in extracted concepts
    # Filter out common words that might not be meaningful research areas
    common_words = {"the", "and", "is", "in", "to", "of", "a", "for", "data", "model", "using", "learning"}
    missing_concepts = topic_terms - extracted_data["concepts"] - common_words
    
    # Return as potential research gaps
    return list(missing_concepts)

def detect_gaps(graph, lc_graph=None, topics=None):
    """Detect research gaps using all available methods"""
    all_gaps = []
    
    if hasattr(graph, 'query'):
        print("Detecting gaps using Neo4j graph analysis...")
        neo4j_gaps = detect_gaps_neo4j(graph)
        if neo4j_gaps:
            print(f"Found {len(neo4j_gaps)} gaps using Neo4j analysis")
            all_gaps.extend(neo4j_gaps)
    elif isinstance(graph, dict):
        print("Detecting gaps using in-memory graph analysis...")
        memory_gaps = detect_gaps_in_memory(graph)
        if memory_gaps:
            print(f"Found {len(memory_gaps)} gaps using in-memory graph analysis")
            all_gaps.extend(memory_gaps)
    
    if lc_graph:
        print("Detecting gaps using LangChain graph analysis...")
        lc_gaps = analyze_langchain_graph(lc_graph)
        if lc_gaps:
            print(f"Found {len(lc_gaps)} gaps using LangChain graph analysis")
            all_gaps.extend(lc_gaps)
    
    if topics:
        print("Detecting gaps using topic modeling...")
        topic_gaps = find_topic_gaps(topics, extracted_data)
        if topic_gaps:
            print(f"Found {len(topic_gaps)} gaps using topic modeling")
            all_gaps.extend(topic_gaps)
    
    if not all_gaps:
        print("No gaps found through graph analysis. Falling back to LLM...")
        concepts_str = ", ".join(list(extracted_data["concepts"])[:10])
        methods_str = ", ".join(list(extracted_data["methods"])[:10])
        applications_str = ", ".join(list(extracted_data["applications"])[:10])
        
        gap_prompt = f"""
        Based on the following concepts, methods, and applications,
        suggest 3 potential research gaps or underexplored areas:
        
        Concepts: {concepts_str}
        Methods: {methods_str}
        Applications: {applications_str}
        
        Output format:
        1. [Gap 1 name]
        2. [Gap 2 name]
        3. [Gap 3 name]
        """
        
        try:
            llm = GroqLLM(model_name="llama3-70b-8192", temperature=0.7, max_tokens=2048)
            response = llm(gap_prompt)
            for line in response.split('\n'):
                line = line.strip()
                if line.startswith('1.') or line.startswith('2.') or line.startswith('3.'):
                    gap = line[2:].strip()
                    if gap:
                        all_gaps.append(gap)
            print(f"Found {len(all_gaps)} gaps using LLM")
        except Exception as e:
            print(f"Error getting gaps from LLM: {e}")
    
    unique_gaps = list(dict.fromkeys(all_gaps))
    
    if not unique_gaps:
        print("No gaps found. Using predefined research gaps.")
        return [
            "Multimodal Learning in Low-resource Settings",
            "Interpretable Neural Language Models",
            "Cross-domain Knowledge Transfer"
        ]
    
    return unique_gaps[:5] 