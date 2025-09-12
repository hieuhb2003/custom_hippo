def calculate_similarity(vec1, vec2):
    """Tính toán cosine similarity giữa hai vector."""
    vec1 = np.squeeze(vec1)
    vec2 = np.squeeze(vec2)
    if vec1.ndim == 0 or vec2.ndim == 0 or vec1.shape[0] != vec2.shape[0]:
        return 0.0
    dot_product = np.dot(vec1, vec2)
    norm_vec1 = np.linalg.norm(vec1)
    norm_vec2 = np.linalg.norm(vec2)
    if norm_vec1 == 0 or norm_vec2 == 0:
        return 0.0
    return dot_product / (norm_vec1 * norm_vec2)

def retrieve_entities_and_similarities(
    entity_query_embedding: np.ndarray,
    entities_vdb,
    top_k_entities: int
) -> tuple[list, dict, dict]:
    """
    Truy xuất các thực thể liên quan, tính toán điểm tương đồng và trả về top K thực thể.

    Hàm này sẽ ưu tiên sử dụng dữ liệu đã tính toán trước (precomputed) để tăng tốc độ,
    nếu không có sẽ fallback về truy vấn VDB trực tiếp.

    Args:
        entity_query_embedding: Vector embedding của câu query.
        entities_vdb: Vector database của các thực thể.
        top_k_entities: Số lượng thực thể hàng đầu cần lấy.

    Returns:
        Một tuple chứa:
        - top_entities (list): Danh sách các (entity_id, score) hàng đầu.
        - entity_similarities (dict): Dictionary chứa điểm của tất cả thực thể tìm được.
        - entity_info (dict): Dictionary chứa thông tin (tên, mô tả) của các thực thể.
    """
    entity_similarities = {}
    entity_info = {}


    entities_results = entities_vdb.query(entity_query_embedding, top_k=1000000)
    for result in entities_results:
        if entity_id := result.get("__id__"):
            entity_info[entity_id] = {"name": result.get("entity_name", ""), "description": result.get("description", "")}
            entity_similarities[entity_id] = float(result.get("score", result.get("distance", 0.0)))
    logger.info(f"Retrieved {len(entity_similarities)} entities using VDB query.")


    sorted_entities = sorted(entity_similarities.items(), key=lambda x: x[1], reverse=True)
    top_entities = sorted_entities[:top_k_entities]
    top_entities_similarities = {eid: score for eid, score in top_entities}
    logger.info(f"Selected top {len(top_entities)} entities from {len(sorted_entities)} candidates.")
    
    return top_entities, entity_similarities, entity_info, top_entities_similarities

def retrieve_edges_and_similarities(
    entity_query_embedding: np.ndarray,
    relationships_vdb,
    top_k_edges: int
) -> tuple[list, dict, dict]:
    """
    Truy xuất các quan hệ (cạnh) liên quan, tính toán điểm tương đồng và trả về top K.

    Args:
        entity_query_embedding: Vector embedding của câu query (dùng chung với entity).
        relationships_vdb: Vector database của các quan hệ.
        top_k_edges: Số lượng quan hệ hàng đầu cần lấy.
        use_precomputed_data: Flag để sử dụng dữ liệu tính toán trước.
        predefined_candidates: Các quan hệ được cung cấp sẵn.

    Returns:
        Một tuple chứa:
        - top_edges (list): Danh sách các (edge_id, score) hàng đầu.
        - edge_similarities (dict): Dictionary chứa điểm của tất cả quan hệ tìm được.
        - edge_info (dict): Dictionary chứa thông tin (src, tgt, mô tả) của các quan hệ.
    """
    edge_similarities = {}
    edge_info = {}
    

    relationships_results = relationships_vdb.query(entity_query_embedding, top_k=1000000)
    for result in relationships_results:
        if edge_id := result.get("__id__"):
            edge_info[edge_id] = {"src_id": result.get("src_id", ""), "tgt_id": result.get("tgt_id", ""), "description": result.get("description", "")}
            edge_similarities[edge_id] = float(result.get("score", result.get("distance", 0.0)))
    logger.info(f"Retrieved {len(edge_similarities)} edges using VDB query.")


    # Sắp xếp và lấy top K
    sorted_edges = sorted(edge_similarities.items(), key=lambda x: x[1], reverse=True)
    top_edges = sorted_edges[:top_k_edges]
    top_edges_similarities = {eid: score for eid, score in top_edges}
    logger.info(f"Selected top {len(top_edges)} edges from {len(sorted_edges)} candidates.")
    
    return top_edges, edge_similarities, edge_info, top_edges_similarities

def retrieve_and_score_chunks(
    query: str,
    query_embedding: np.ndarray,
    model,
    top_entities: list,
    top_edges: list,
    entity_info: dict,
    edge_info: dict,
    full_graph,
    text_chunks_db,
    chunks_vdb,
) -> tuple[dict, dict, dict, dict]:
    """
    Từ các thực thể và quan hệ hàng đầu, tìm các chunk liên quan và tính điểm cho chúng.

    Args:
        query: Câu query gốc.
        query_embedding: Vector embedding của câu query (dùng cho chunk).
        embedding_func: Hàm để tạo embedding.
        top_entities: Danh sách các thực thể hàng đầu.
        top_edges: Danh sách các quan hệ hàng đầu.
        entity_info: Thông tin về các thực thể.
        edge_info: Thông tin về các quan hệ.
        full_graph: Instance của knowledge graph.
        text_chunks_db: DB chứa nội dung các chunk.
        chunks_vdb: Vector DB của các chunk.

    Returns:
        Một tuple chứa:
        - chunk_sim_scores (dict): Điểm tương đồng của mỗi chunk với query.
        - chunk_to_entities (dict): Mapping từ chunk_id ra danh sách entity_id.
        - chunk_to_edges (dict): Mapping từ chunk_id ra danh sách edge_id.
        - chunk_contents (dict): Nội dung text của các chunk.
    """
    # Bước 5 & 6: Lấy các chunk chứa top entities và top edges
    all_unique_chunk_ids = set()
    chunks_by_entity = defaultdict(list)
    chunks_by_edge = defaultdict(list)

    for entity_id, _ in top_entities:
        try:
            entity_name = entity_info[entity_id]["name"]
            
            if full_graph.has_node(entity_name) and "source_id" in full_graph.nodes[entity_name]:
                entity_chunks = full_graph.nodes[entity_name]["source_id"].split("<SEP>")
                chunks_by_entity[entity_id] = entity_chunks
                all_unique_chunk_ids.update(entity_chunks)
        except Exception as e:
            logger.error(f"Error getting chunks for entity {entity_id}: {e}")

    for edge_id, _ in top_edges:
        try:
            if edge_id in edge_info:
                src_id, tgt_id = edge_info[edge_id]["src_id"], edge_info[edge_id]["tgt_id"]
                if full_graph.has_edge(src_id, tgt_id) and "source_id" in full_graph.edges[src_id, tgt_id]:
                    edge_data = full_graph.edges[src_id, tgt_id]            
                    edge_chunks = [c for c in edge_data.get("source_id", "").split("<SEP>") if c]
                    chunks_by_edge[edge_id] = edge_chunks
                    all_unique_chunk_ids.update(edge_chunks)
        except Exception as e:
            logger.error(f"Error getting chunks for edge {edge_id}: {e}")
    
    if not all_unique_chunk_ids:
        logger.warning("No chunks found containing top entities or edges.")
        return {}, {}, {}, {}
    
    logger.info(f"Found {len(all_unique_chunk_ids)} unique chunks from entities and edges.")
    chunk_ids_list = list(all_unique_chunk_ids)

    # Bước 8: Lấy nội dung các chunk
    chunk_contents = {}
    try:
        chunk_contents = {cid: text_chunks_db[cid]['content'] for cid in chunk_ids_list}
        # chunk_contents = {chunk_id: data['content'] for chunk_id, data in zip(chunk_ids_list, chunk_data_list) if data and 'content' in data}
        logger.info(f"Retrieved contents for {len(chunk_contents)} chunks.")
    except Exception as e:
        logger.error(f"Error fetching chunk contents: {e}")
        return {}, {}, {}, {}

    # Bước 9: Tính toán điểm tương đồng của chunk với query
    chunk_sim_scores = {}
    try: # Ưu tiên query VDB
        vdb_results = chunks_vdb.query(query_embedding, top_k=len(chunk_ids_list), filter_lambda=lambda x: x["__id__"] in chunk_ids_list)
        for result in vdb_results:
            if chunk_id := result.get("__id__"):
                chunk_sim_scores[chunk_id] = float(result.get("score", result.get("distance", 0.0)))
        logger.info(f"Retrieved similarity for {len(chunk_sim_scores)} chunks using VDB.")
    except Exception as e:
        logger.error(f"Error querying chunks VDB, will calculate manually: {e}")

    # Tính toán similarity cho các chunk còn thiếu
    missing_chunks = [cid for cid in chunk_ids_list if cid not in chunk_sim_scores and cid in chunk_contents]
    if missing_chunks:
        logger.info(f"Calculating similarity for {len(missing_chunks)} missing chunks manually.")
        try:
            missing_embeddings =  [model.encode(chunk_contents[cid])['dense_vecs'] for cid in missing_chunks]
            for i, chunk_id in enumerate(missing_chunks):
                chunk_sim_scores[chunk_id] = calculate_similarity(query_embedding, missing_embeddings[i])
        except Exception as e:
            logger.error(f"Error calculating missing chunk similarities: {e}")

    # Bước 10: Tạo mappings
    chunk_to_entities = defaultdict(list)
    for entity_id, chunks in chunks_by_entity.items():
        for chunk_id in chunks:
            if chunk_id in all_unique_chunk_ids:
                chunk_to_entities[chunk_id].append(entity_id)

    chunk_to_edges = defaultdict(list)
    for edge_id, chunks in chunks_by_edge.items():
        for chunk_id in chunks:
            if chunk_id in all_unique_chunk_ids:
                chunk_to_edges[chunk_id].append(edge_id)

    return chunk_sim_scores, chunk_to_entities, chunk_to_edges, chunk_contents
# ==============================================================================

def process_chunk_id_results(chunk_ids, chunk_id_to_doc_id, doc_id_to_doc_content, doc_content_to_doc_idx, top_k=100) -> list[str]:
    """Xử lý kết quả chunk IDs thành corpus IDs."""
    results = []
    seen_corpus_ids = set()
    for chunk_id in chunk_ids:
        if len(results) >= top_k:
            break
        doc_id = chunk_id_to_doc_id.get(chunk_id)
        if doc_id:
            doc_content = doc_id_to_doc_content.get(doc_id)
            if doc_content:
                corpus_id = doc_content_to_doc_idx.get(doc_content)
                if corpus_id and corpus_id not in seen_corpus_ids:
                    results.append(corpus_id)
                    seen_corpus_ids.add(corpus_id)
    return results

# ==============================================================================
# HÀM RETRIEVAL COMPOSITE CHÍNH
# ==============================================================================

def enhanced_chunk_retrieval_direct(
    query: str,
    full_graph: Any,  # BaseGraphStorage
    entities_vdb: Any,  # BaseVectorStorage
    relationships_vdb: Any,  # BaseVectorStorage
    text_chunks_db: Any,  # BaseKVStorage
    chunks_vdb: Any,  # BaseVectorStorage
    model,
    chunk_id_to_doc_id: Dict[str, str],
    doc_id_to_doc_content: Dict[str, str],
    doc_content_to_doc_idx: Dict[str, str],
    top_k_entities: int = 100,
    top_k_edges: int = 100,
    top_k_chunks: int = 100,
    a: float = 0.3,            # NEW: weight
    b: float = 0.5,            # NEW: weight
    c: float = 0.5,            # NEW: weight
    relevance_threshold: float = 0.7,  # NEW: align with async
    method: str | None = None           # NEW: align with async
) -> List[str]:  # FIX: return type is list[str] (corpus_ids)
    """
    Synchronous version aligned with async variant. If method is None,
    falls back to weighted sum on normalized scores.
    """
    query_embedding = model.encode(query)["dense_vecs"]

    # --- entities
    top_entities, entity_similarities, entity_info, top_entities_similarities = retrieve_entities_and_similarities(
        entity_query_embedding=query_embedding,
        entities_vdb=entities_vdb,
        top_k_entities=top_k_entities
    )
    if not top_entities:
        logger.warning("No top entities found. Returning empty result.")
        return []

    # --- edges
    top_edges, edge_similarities, edge_info, top_edges_similarities = retrieve_edges_and_similarities(
        entity_query_embedding=query_embedding,
        relationships_vdb=relationships_vdb,
        top_k_edges=top_k_edges
    )

    # --- chunks
    chunk_sim_scores, chunk_to_entities, chunk_to_edges, chunk_contents = retrieve_and_score_chunks(
        query=query,
        query_embedding=query_embedding,
        model=model,
        top_entities=top_entities,
        top_edges=top_edges,
        entity_info=entity_info,
        edge_info=edge_info,
        full_graph=full_graph,
        text_chunks_db=text_chunks_db,
        chunks_vdb=chunks_vdb,
    )
    if not chunk_sim_scores:
        return []

    # mean scores per chunk from entity/edge
    chunk_entity_scores = {
        cid: (np.mean([entity_similarities.get(eid, 0.0) for eid in eids]) if eids else 0.0)
        for cid, eids in chunk_to_entities.items()
    }
    chunk_edge_scores = {
        cid: (np.mean([edge_similarities.get(rid, 0.0) for rid in rids]) if rids else 0.0)
        for cid, rids in chunk_to_edges.items()
    }

    def normalize_scores(scores: Dict[str, float]) -> Dict[str, float]:
        if not scores: return {}
        vals = list(scores.values())
        mn, mx = min(vals), max(vals)
        rng = mx - mn
        if rng == 0:
            return {k: 1.0 for k in scores}
        return {k: (v - mn) / rng for k, v in scores.items()}

    normalized_chunk_sim = normalize_scores(chunk_sim_scores)
    normalized_entity_scores = normalize_scores(chunk_entity_scores)
    normalized_edge_scores = normalize_scores(chunk_edge_scores)

    # optional: relevant count (same spirit as async)
    chunk_relevant_counts = defaultdict(int)
    for cid, eids in chunk_to_entities.items():
        chunk_relevant_counts[cid] += sum(1 for eid in eids if entity_similarities.get(eid, 0.0) > relevance_threshold)
    for cid, rids in chunk_to_edges.items():
        chunk_relevant_counts[cid] += sum(1 for rid in rids if edge_similarities.get(rid, 0.0) > relevance_threshold)

    final_scores: Dict[str, float] = {}
    all_unique_chunk_ids = list(chunk_sim_scores.keys())

    for chunk_id in all_unique_chunk_ids:
        chunk_score = normalized_chunk_sim.get(chunk_id, 0.0)
        entity_score = normalized_entity_scores.get(chunk_id, 0.0)
        edge_score = normalized_edge_scores.get(chunk_id, 0.0)

        if method == "max":
            ent_scores = [entity_similarities[eid] for eid in chunk_to_entities.get(chunk_id, [])
                          if entity_similarities.get(eid, 0.0) > relevance_threshold]
            rel_scores = [edge_similarities[rid] for rid in chunk_to_edges.get(chunk_id, [])
                          if edge_similarities.get(rid, 0.0) > relevance_threshold]
            max_ent = max(ent_scores) if ent_scores else 0.0
            max_rel = max(rel_scores) if rel_scores else 0.0
            # NOTE: dùng chunk_score đã chuẩn hoá để đồng nhất với async
            final_scores[chunk_id] = a * chunk_score + b * max_ent + c * max_rel

        elif method == "max_top_k":
            ent_raw = sorted(
                [entity_similarities[eid] for eid in chunk_to_entities.get(chunk_id, [])
                 if entity_similarities.get(eid, 0.0) > relevance_threshold],
                reverse=True
            )[:5]
            rel_raw = sorted(
                [edge_similarities[rid] for rid in chunk_to_edges.get(chunk_id, [])
                 if edge_similarities.get(rid, 0.0) > relevance_threshold],
                reverse=True
            )[:5]
            avg_ent = sum(ent_raw)/len(ent_raw) if ent_raw else 0.0
            avg_rel = sum(rel_raw)/len(rel_raw) if rel_raw else 0.0
            final_scores[chunk_id] = a * normalized_chunk_sim.get(chunk_id, 0.0) + b * avg_ent + c * avg_rel

        elif method == "new_function":
            base = a * chunk_score + b * entity_score + c * edge_score
            modifier = math.log1p(chunk_relevant_counts.get(chunk_id, 0))
            density_bonus_weight = 0.1
            final_scores[chunk_id] = base + density_bonus_weight * modifier

        else:
            # method is None → fallback (chuẩn hoá)
            final_scores[chunk_id] = a * chunk_score + b * entity_score + c * edge_score

    # chọn top_k_chunks và ánh xạ sang corpus_ids
    sorted_chunks = sorted(final_scores.items(), key=lambda x: x[1], reverse=True)[:top_k_chunks]
    top_chunk_ids = [cid for cid, _ in sorted_chunks]
    return process_chunk_id_results(top_chunk_ids, chunk_id_to_doc_id, doc_id_to_doc_content, doc_content_to_doc_idx)