import numpy as np
import os
import json
import networkx as nx
from networkx import Graph
from tqdm import tqdm
import time
from scipy.special import softmax

from nano_vectordb import NanoVectorDB
from FlagEmbedding import BGEM3FlagModel

# SECTION 1: CÁC HÀM HELPER
# ... (Các hàm min_max_normalize, retrieve_entities, retrieve_chunks, run_ppr, process_chunk_id_results, create_query_subgraph, filter_subgraph_chunks giữ nguyên như file trước)
def min_max_normalize(x):
    """Chuẩn hóa mảng numpy về khoảng [0, 1]."""
    min_val = np.min(x)
    max_val = np.max(x)
    range_val = max_val - min_val
    if range_val == 0:
        return np.ones_like(x)
    return (x - min_val) / range_val

def retrieve_entities(entities_vdb, model, query, top_k, options) -> tuple[dict, dict]:
    """
    Lấy ra các entity liên quan và điểm của từng cặp (entity, chunk).
    1. Query một lượng lớn item.
    2. Sắp xếp, lấy top 100 items để xác định seed entities.
    3. Trả về cả điểm seed entities và điểm của từng cặp (entity, chunk).
    """
    # Bước 1: Query một lượng rất lớn để có cái nhìn tổng quan
    # CẢNH BÁO: top_k lớn có thể gây tốn bộ nhớ và chậm. 
    # Cân nhắc giảm xuống một con số hợp lý hơn nếu cần (ví dụ: 10000).
    # print("      Querying a large number of entities...")
    query_emb = model.encode(query)["dense_vecs"]
    all_results = entities_vdb.query(query_emb, top_k=1000000)
    
    # Bước 2: Tạo dict điểm cho từng cặp (entity, chunk) cụ thể
    entity_chunk_pair_scores = {
        (sample["entity_name"], sample["chunk_id"]): sample["__metrics__"]
        for sample in all_results
    }

    # Bước 3: Sắp xếp toàn bộ kết quả và chỉ lấy top 100 item đầu tiên để tính điểm seed entity
    all_results.sort(key=lambda x: x['__metrics__'], reverse=True)
    top_100_results = all_results[:100]

    # Bước 4: Tính điểm cho seed entity dựa trên top 100 item này
    unique_entities_list_score = {}
    for sample in top_100_results:
        entity_name = sample["entity_name"]
        if entity_name not in unique_entities_list_score:
            unique_entities_list_score[entity_name] = []
        unique_entities_list_score[entity_name].append(sample["__metrics__"])
    if options == "max":
        unique_entities = {entity: max(scores) for entity, scores in unique_entities_list_score.items()}
    else:  # average
        unique_entities = {entity: sum(scores) / len(scores) for entity, scores in unique_entities_list_score.items()}
    
    if not unique_entities:
        return {}, {}
        
    only_scores = np.array(list(unique_entities.values()))
    only_scores_min_max = min_max_normalize(only_scores).tolist()
    
    entities_name_to_score = sorted(zip(unique_entities.keys(), only_scores_min_max), key=lambda x: x[1], reverse=True)
    
    final_seed_entities = {k: v for k, v in entities_name_to_score[:top_k]}
    full_entities_to_score = {k: v for k, v in entities_name_to_score}
    return final_seed_entities, entity_chunk_pair_scores, full_entities_to_score

def retrieve_chunks(chunk_vdb, model, query, top_k) -> dict[str, int]:
    """Lấy ra các chunk liên quan từ vector database."""
    query_emb = model.encode(query)["dense_vecs"]
    result = chunk_vdb.query(query_emb, top_k=top_k)
    
    res_chunk_score = [(x["__id__"], x["__metrics__"]) for x in result]
    if not res_chunk_score:
        return {}
        
    only_scores = np.array([x[1] for x in res_chunk_score])
    only_scores_min_max = min_max_normalize(only_scores).tolist()
    
    return {x[0]: s for x, s in zip(res_chunk_score, only_scores_min_max)}

def run_ppr(graph, node_weight) -> list[str]:
    """Chạy thuật toán Personalized PageRank."""
    if graph.number_of_nodes() == 0 or np.sum(node_weight) == 0:
        return []

    dict_node_weight = {n: s for n, s in zip(list(graph.nodes), node_weight.tolist())}
    
    pr = nx.pagerank(graph, alpha=0.85, personalization=dict_node_weight, max_iter=500, weight="weight")
    
    pr_list_sorted = sorted(pr.items(), key=lambda x: x[1], reverse=True)
    
    return [node for node, score in pr_list_sorted if "chunk-" in node]

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

def create_query_subgraph(full_graph: Graph, seed_nodes: list, depth: int) -> tuple[Graph, list]:
    """
    Phiên bản tối ưu của create_query_subgraph.
    Xây dựng subgraph một cách thủ công để tránh chi phí của .subgraph().copy().
    """
    # Bước 1: Thu thập các node entity hiệu quả
    entity_nodes_in_subgraph = set(seed_nodes)
    
    # Sử dụng BFS thủ công thay vì ego_graph để kiểm soát tốt hơn
    queue = list(seed_nodes)
    visited = set(seed_nodes)
    
    for d in range(depth):
        level_size = len(queue)
        if level_size == 0:
            break
        for _ in range(level_size):
            current_node = queue.pop(0)
            if not full_graph.has_node(current_node):
                continue
            for neighbor in full_graph.neighbors(current_node):
                if "chunk-" not in str(neighbor) and neighbor not in visited:
                    visited.add(neighbor)
                    entity_nodes_in_subgraph.add(neighbor)
                    queue.append(neighbor)

    # Bước 2: Bắt đầu xây dựng subgraph mới từ đầu
    subgraph = nx.Graph()
    
    # Thêm các node entity vào trước
    subgraph.add_nodes_from(entity_nodes_in_subgraph)

    # Thêm thuộc tính và các cạnh entity-entity
    for entity_node in entity_nodes_in_subgraph:
        # DÒNG SỬA LỖI: Thêm kiểm tra sự tồn tại của node trong full_graph
        if full_graph.has_node(entity_node):
            # Sao chép thuộc tính node
            subgraph.nodes[entity_node].update(full_graph.nodes[entity_node])
            # Thêm các cạnh entity-entity
            for neighbor in full_graph.neighbors(entity_node):
                if neighbor in entity_nodes_in_subgraph and entity_node < neighbor:
                    subgraph.add_edge(entity_node, neighbor, **full_graph.get_edge_data(entity_node, neighbor))

    # Bước 3: Thu thập và thêm các node chunk cùng các cạnh tương ứng
    for entity_node in entity_nodes_in_subgraph:
        # DÒNG SỬA LỖI: Thêm kiểm tra sự tồn tại của node trong full_graph
        if full_graph.has_node(entity_node) and "source_id" in full_graph.nodes[entity_node]:
            chunks = full_graph.nodes[entity_node]["source_id"].split("<SEP>")
            for chunk in chunks:
                if not subgraph.has_node(chunk):
                    subgraph.add_node(chunk)
                subgraph.add_edge(entity_node, chunk)

    return subgraph, list(entity_nodes_in_subgraph)

def filter_subgraph_chunks(subgraph: Graph, entity_nodes: list, scored_chunks: dict, top_k_chunks: int) -> Graph:
    """Lọc subgraph để mỗi entity chỉ giữ lại kết nối tới top_k chunk có điểm cao nhất."""
    final_graph = nx.Graph()
    final_graph.add_nodes_from(entity_nodes)
    
    for u, v in subgraph.edges():
        if u in entity_nodes and v in entity_nodes:
            final_graph.add_edge(u, v, **subgraph.get_edge_data(u, v))

    for entity in entity_nodes:
        neighbor_chunks = [n for n in subgraph.neighbors(entity) if "chunk-" in str(n)]
        
        scored_neighbor_chunks = []
        for chunk in neighbor_chunks:
            if chunk in scored_chunks:
                scored_neighbor_chunks.append((chunk, scored_chunks[chunk]))
        
        scored_neighbor_chunks.sort(key=lambda x: x[1], reverse=True)
        
        top_chunks_for_entity = [chunk for chunk, score in scored_neighbor_chunks[:top_k_chunks]]
        
        for chunk in top_chunks_for_entity:
            final_graph.add_node(chunk) # Đảm bảo node chunk tồn tại
            final_graph.add_edge(entity, chunk, weight=1.0)
            
    return final_graph

# HÀM MỚI
def retrieve_relations(relations_vdb, model, query, top_k=100) -> dict[tuple, float]:
    """
    Lấy ra các quan hệ (cạnh) liên quan từ vector database.
    Điểm của cạnh là điểm max trong các quan hệ tìm được.
    """
    query_emb = model.encode(query)["dense_vecs"]
    results = relations_vdb.query(query_emb, top_k=1000000)
    unique_edges_scores = {}
    for sample in results:
        edge = tuple(sorted((sample["src_id"], sample["tgt_id"])))
        if edge not in unique_edges_scores:
            unique_edges_scores[edge] = []
        unique_edges_scores[edge].append(sample["__metrics__"])
    final_edge_scores = {edge: max(scores) for edge, scores in unique_edges_scores.items()}
    return final_edge_scores

# HÀM MỚI
def assign_custom_edge_weights(graph: Graph, relation_scores: dict, entity_chunk_pair_scores: dict, all_scored_chunks=None, alpha: float = 0.5) -> Graph:
    """
    Gán trọng số tùy chỉnh cho các cạnh trong subgraph.
    - Cạnh Entity-Chunk giờ sẽ lấy điểm từ cặp (entity, chunk) cụ thể.
    """
    ee_edges, ec_edges = [], []
    ee_raw_scores, ec_raw_scores = [], []
    for u, v in graph.edges():
        u_is_entity = "chunk-" not in str(u)
        v_is_entity = "chunk-" not in str(v)

        # Trường hợp cạnh Entity-Entity (giữ nguyên)
        if u_is_entity and v_is_entity:
            edge_key = tuple(sorted((u, v)))
            if edge_key in relation_scores:
                ee_edges.append((u, v))
                ee_raw_scores.append(relation_scores[edge_key])
        
        # Trường hợp cạnh Entity-Chunk (logic mới)
        if all_scored_chunks is None:
            entity_node, chunk_node = (u, v) if u_is_entity else (v, u)
            ec_key = (entity_node, chunk_node)
            if ec_key in entity_chunk_pair_scores:
                ec_edges.append((u, v))
                ec_raw_scores.append(entity_chunk_pair_scores[ec_key])

        else:
            entity_node, chunk_node = (u, v) if u_is_entity else (v, u)
            ec_key = (entity_node, chunk_node)
            
            score_pair = entity_chunk_pair_scores.get(ec_key, 0)
            score_chunk = all_scored_chunks.get(chunk_node, 0)
            
            # Kết hợp điểm: Trọng số hóa sự liên quan của cặp (E,C) và sự liên quan trực tiếp (Q,C)
            hybrid_score = alpha * score_pair + score_chunk
                        
            ec_edges.append((u, v))
            ec_raw_scores.append(hybrid_score)

    # Phần tính softmax và gán trọng số giữ nguyên
    if ee_edges:
        ee_weights = softmax(np.array(ee_raw_scores))
        for i, edge in enumerate(ee_edges):
            graph.edges[edge]['weight'] = ee_weights[i]
    if ec_edges:
        ec_weights = softmax(np.array(ec_raw_scores))
        for i, edge in enumerate(ec_edges):
            graph.edges[edge]['weight'] = ec_weights[i]
            
    return graph
# SECTION 2: LUỒNG CHẠY CHÍNH VỚI LOGIC SUBGRAPH (ĐÃ CẬP NHẬT)

def assign_local_softmax_weights(
    graph: Graph, 
    relation_scores: dict, 
    entity_chunk_pair_scores: dict, 
    all_scored_chunks: dict, 
    alpha: float = 0.5, 
    temperature: float = 1.0, 
    eps: float = 1e-6
) -> Graph:
    """
    Gán trọng số cạnh bằng phương pháp Softmax Local cho từng node entity.

    Với mỗi node entity, hàm này sẽ tính toán một phân phối xác suất trên tất cả 
    các cạnh đi ra của nó (bao gồm cả cạnh tới entity khác và tới chunk). 
    Điều này buộc các node lân cận phải "cạnh tranh" trực tiếp với nhau để nhận 
    được "dòng chảy" xác suất từ PageRank.

    Args:
        graph (Graph): Subgraph đầu vào.
        relation_scores (dict): Dict chứa điểm của các cặp (entity, entity).
        entity_chunk_pair_scores (dict): Dict chứa điểm của các cặp (entity, chunk).
        all_scored_chunks (dict): Dict chứa điểm truy vấn trực tiếp của các chunk.
        alpha (float): Trọng số để kết hợp điểm pair và điểm chunk (hybrid score).
        temperature (float): Tham số điều chỉnh độ "sắc" của softmax. 
                             Temperature thấp -> sắc nét hơn, cao -> mềm mại hơn.
        eps (float): Một giá trị rất nhỏ để tránh lỗi chia cho 0.

    Returns:
        Graph: Đồ thị đã được gán trọng số cạnh mới.
    """
    # Lấy danh sách các node là entity
    entity_nodes = [n for n in graph.nodes() if "chunk-" not in str(n)]

    # Duyệt qua từng entity để tính toán trọng số cho các cạnh đi ra
    for u in entity_nodes:
        neighbors = list(graph.neighbors(u))
        if not neighbors:
            continue

        raw_scores = []
        edges_to_weight = []

        # Thu thập điểm thô cho tất cả các cạnh đi ra từ node `u`
        for v in neighbors:
            edge = (u, v)
            # Trường hợp cạnh Entity-Entity
            if "chunk-" not in str(v):
                key = tuple(sorted(edge))
                score = relation_scores.get(key, 0.0)
                raw_scores.append(score)
                edges_to_weight.append(edge)
            # Trường hợp cạnh Entity-Chunk (sử dụng hybrid score)
            else:
                score_pair = entity_chunk_pair_scores.get(edge, 0.0)
                score_chunk = all_scored_chunks.get(v, 0.0)
                hybrid_score = (alpha * score_pair) + ((1 - alpha) * score_chunk)
                raw_scores.append(hybrid_score)
                edges_to_weight.append(edge)

        # Áp dụng temperature và softmax nếu có điểm để tính
        if raw_scores:
            scores_array = np.array(raw_scores) / max(temperature, eps)
            softmax_weights = softmax(scores_array)

            # Gán trọng số đã được chuẩn hóa lại cho các cạnh
            for weight, (src, dst) in zip(softmax_weights, edges_to_weight):
                graph[src][dst]['weight'] = float(weight)

    # Đảm bảo tất cả các cạnh đều có trọng số, tránh lỗi trong PageRank
    # Gán trọng số rất nhỏ cho các cạnh chưa được xử lý (ví dụ: cạnh giữa 2 chunk nếu có)
    for u, v in graph.edges():
        if 'weight' not in graph.edges[u, v]:
            graph.edges[u, v]['weight'] = eps
            
    return graph

def run_single_query_subgraph_ppr(full_graph, entities_vdb, relations_vdb, chunks_vdb, query, model, 
                                  chunk_id_to_doc_id, doc_id_to_doc_content, doc_content_to_doc_idx,
                                  top_k_entities, subgraph_depth, top_k_chunks_per_entity, passage_node_weight, number_chunk, options="max", weight_method="global", alpha=0.5):
    """Chạy toàn bộ quy trình PPR trên subgraph cho một query duy nhất."""
    # === THAY ĐỔI CÁCH GỌI HÀM ===
    # Giờ đây hàm trả về 2 giá trị
    if options == "max":
        seed_entities_to_score, entity_chunk_pair_scores, all_retrieved_entities_to_score = retrieve_entities(entities_vdb, model, query, top_k=top_k_entities, options="max")
    else:   # average
        seed_entities_to_score, entity_chunk_pair_scores, all_retrieved_entities_to_score = retrieve_entities(entities_vdb, model, query, top_k=top_k_entities, options="average")
    
    retrieved_relations_to_score = retrieve_relations(relations_vdb, model, query, top_k=200)
    all_scored_chunks = retrieve_chunks(chunks_vdb, model, query, top_k=number_chunk)
    seed_nodes = list(seed_entities_to_score.keys())
    
    unfiltered_subgraph, entity_nodes = create_query_subgraph(full_graph, seed_nodes, depth=subgraph_depth)
    final_subgraph = filter_subgraph_chunks(unfiltered_subgraph, entity_nodes, all_scored_chunks, top_k_chunks=top_k_chunks_per_entity)
    
    if final_subgraph.number_of_nodes() == 0:
        return []

    # === THAY ĐỔI CÁCH GỌI HÀM ===
    # Truyền dict điểm của cặp (entity, chunk) thay vì điểm trung bình của entity
    if weight_method == "local":
        weighted_subgraph = assign_local_softmax_weights(final_subgraph, retrieved_relations_to_score, entity_chunk_pair_scores, all_scored_chunks=all_scored_chunks, alpha=alpha, temperature=1.0)
    else:  # global
        weighted_subgraph = assign_custom_edge_weights(final_subgraph, retrieved_relations_to_score, entity_chunk_pair_scores, all_scored_chunks=all_scored_chunks, alpha=alpha)

    # Phần còn lại giữ nguyên
    node_weights = np.zeros(weighted_subgraph.number_of_nodes())
    node_name_to_id = {v: k for k, v in enumerate(list(weighted_subgraph.nodes()))}

    for node_name, node_id in node_name_to_id.items():
        if node_name in all_retrieved_entities_to_score:
            node_weights[node_id] = all_retrieved_entities_to_score[node_name]
        elif node_name in all_scored_chunks:
            node_weights[node_id] = all_scored_chunks[node_name] * passage_node_weight
            
    docs_results = run_ppr(weighted_subgraph, node_weights)
    return process_chunk_id_results(docs_results, chunk_id_to_doc_id, doc_id_to_doc_content, doc_content_to_doc_idx)


def run_query_page_rank(working_dir_1, query_path, corpus_path, db_name, model_path,
                        top_k_entities, subgraph_depth, top_k_chunks_per_entity, passage_node_weight=0.5, options="max", weight_method="global", alpha=0.5):
    """Hàm tổng hợp, điều phối toàn bộ quá trình."""
    print("Initializing models and databases...")
    embedding_dim = 1024
    entities_vdb = NanoVectorDB(embedding_dim, storage_file=os.path.join(working_dir_1, f"vdb_entities_{db_name}.json"))
    chunks_vdb = NanoVectorDB(embedding_dim, storage_file=os.path.join(working_dir_1, f"vdb_chunks_{db_name}.json"))
    # === THÊM VDB CHO RELATIONS ===
    relations_vdb = NanoVectorDB(embedding_dim, storage_file=os.path.join(working_dir_1, f"vdb_relationships_{db_name}.json"))
    
    model = BGEM3FlagModel(model_path, use_fp16=False, devices= "cuda:1", pooling_method="cls")

    print("Loading graph and mapping files...")
    graph_path = os.path.join(working_dir_1, "graph_chunk_entity_relation.graphml")
    full_graph = nx.read_graphml(graph_path)
    
    chunk_kv_path = os.path.join(working_dir_1, "kv_store_text_chunks.json")
    with open(chunk_kv_path, "r", encoding="utf-8") as f:
        chunk_kv = json.load(f)
    number_chunk = len(chunk_kv)
    full_graph.add_nodes_from(list(chunk_kv.keys()))

    doc_kv_path = os.path.join(working_dir_1, "kv_store_full_docs.json")
    with open(doc_kv_path, "r", encoding="utf-8") as f:
        doc_kv = json.load(f)
    doc_id_to_doc_content = {k: v["content"] for k, v in doc_kv.items()}
    chunk_id_to_doc_id = {k: v["full_doc_id"] for k, v in chunk_kv.items()}

    with open(corpus_path, "r", encoding="utf-8") as f:
        corpus = json.load(f)
    doc_content_to_doc_idx = {v: k for k, v in corpus.items()}

    with open(query_path, "r", encoding="utf-8") as f:
        queries = json.load(f)

    
    results_dict = {"PPR_method": {}}
    for query_id, query in tqdm(list(queries.items()), desc="Running Queries on Subgraphs..."):
        results = run_single_query_subgraph_ppr(full_graph, entities_vdb, relations_vdb, chunks_vdb, query, model,
                                                chunk_id_to_doc_id, doc_id_to_doc_content, doc_content_to_doc_idx,
                                                top_k_entities, subgraph_depth, top_k_chunks_per_entity,
                                                passage_node_weight, number_chunk, options=options, weight_method=weight_method, alpha=alpha)
        results_dict["PPR_method"][query_id] = results

    return results_dict


# SECTION 3: ĐIỂM BẮT ĐẦU CHẠY SCRIPT
if __name__ == "__main__":
    # WORKING_DIR_1 = "/home/hungpv/projects/TN/LIGHTRAG/new_musique_en_without_embedding"
    # QUERY_PATH = "/home/hungpv/projects/TN/data/data_musique/dev_queries_vi.json"
    # CORPUS_PATH = "/home/hungpv/projects/TN/data/data_musique/filter_corpus_en.json"
    # MODEL_PATH = "BAAI/bge-m3"
    # DB_NAME = "bge-m3-cls"
    # OUTPUT_DIR = "//home/hungpv/projects/ppr_weight/sub_graph_weight_assign_softmax_global" # Thư mục lưu kết quả

    # # Tham số cố định cho thuật toán
    # TOP_K_ENTITIES = 10
    # TOP_K_CHUNKS_PER_ENTITY = 10
    
    # # --- PHẦN 2: ĐỊNH NGHĨA KHÔNG GIAN TÌM KIẾM (GRID SEARCH) ---
    
    # top_k_chunk_to_search = [10]  # Có thể thêm các giá trị khác nếu muốn thử nghiệm

    # top_k_entity_to_search = [10]  # Có thể thêm các giá trị khác nếu muốn thử nghiệm
    # # Danh sách các giá trị `depth` cần thử nghiệm
    # depths_to_search = [1] 
    # alpha_to_search = [0.2,0.3,0.5,0.7]
    # # Danh sách các giá trị `weight` của passage cần thử nghiệm
    # weights_to_search = [0, 0.05, 0.1]
    # options = ['max', 'average']
    # weights_options = ['global']
    # # --- PHẦN 3: THỰC THI VÒNG LẶP GRID SEARCH ---
    
    # print("🚀 Starting Grid Search for Subgraph PPR...")
    # for top_k_entities in top_k_entity_to_search:
    # # Vòng lặp ngoài: duyệt qua các giá trị depth
    #     for depth in depths_to_search:
    #         # Vòng lặp trong: duyệt qua các giá trị weight
    #         for weight in weights_to_search:
    #             for top_k_chunks in top_k_chunk_to_search:
    #                 for opt in options:
    #                     for weight_method in weights_options:
    #                         for alpha in alpha_to_search:
    #                             print("-" * 60)
    #                             print(f"🧪 Running experiment with: SUBGRAPH_DEPTH = {depth}, PASSAGE_NODE_WEIGHT = {weight}")
                                
    #                             # Tạo tên file output động để không bị ghi đè
    #                             output_filename = f"test_musique_subgraph_{weight_method}_softmax_top_entity_{top_k_entities}_depth_{depth}_weight{weight}_top_chunk_per_entity_{top_k_chunks}_option_{opt}_wetght_of_entity_in_ec_{alpha}.json"
    #                             output_path = os.path.join(OUTPUT_DIR, output_filename)

    #                             # Gọi hàm xử lý chính với các tham số hiện tại
    #                             results = run_query_page_rank(
    #                                 working_dir_1=WORKING_DIR_1,
    #                                 query_path=QUERY_PATH,
    #                                 corpus_path=CORPUS_PATH,
    #                                 db_name=DB_NAME,
    #                                 model_path=MODEL_PATH,
    #                                 top_k_entities=top_k_entities,
    #                                 top_k_chunks_per_entity=top_k_chunks,
    #                                 # Sử dụng các giá trị từ vòng lặp
    #                                 subgraph_depth=depth,
    #                                 passage_node_weight=weight,
    #                                 options=opt,
    #                                 weight_method=weight_method,
    #                                 alpha=alpha
    #                             )

    #                             # Lưu kết quả của lần chạy này
    #                             print(f"💾 Saving results for (depth={depth}, weight={weight}) to {output_path}")
    #                             with open(output_path, "w", encoding="utf-8") as f:
    #                                 json.dump(results, f, ensure_ascii=False, indent=4)
    #                             print("✅ Done with this run.")

    # print("-" * 60)
    # print("🎉 Grid Search finished successfully!")

    # WORKING_DIR_1 = "/home/hungpv/projects/TN/LIGHTRAG/new_nq_en_without_embedding"
    # QUERY_PATH = "/home/hungpv/projects/TN/data/data_nq/dev_queries_vi.json"
    # CORPUS_PATH = "/home/hungpv/projects/TN/data/data_nq/filter_corpus_en.json"
    # MODEL_PATH = "BAAI/bge-m3"
    # DB_NAME = "bge-m3-cls"
    # OUTPUT_DIR = "//home/hungpv/projects/ppr_weight/sub_graph_weight_assign_softmax_global" # Thư mục lưu kết quả

    # # Tham số cố định cho thuật toán
    # TOP_K_ENTITIES = 10
    # TOP_K_CHUNKS_PER_ENTITY = 10
    
    # # --- PHẦN 2: ĐỊNH NGHĨA KHÔNG GIAN TÌM KIẾM (GRID SEARCH) ---
    
    # TOP_K_ENTITIES = 10
    # TOP_K_CHUNKS_PER_ENTITY = 10
    
    # # --- PHẦN 2: ĐỊNH NGHĨA KHÔNG GIAN TÌM KIẾM (GRID SEARCH) ---
    
    # top_k_chunk_to_search = [10]  # Có thể thêm các giá trị khác nếu muốn thử nghiệm

    # top_k_entity_to_search = [10]  # Có thể thêm các giá trị khác nếu muốn thử nghiệm
    # # Danh sách các giá trị `depth` cần thử nghiệm
    # depths_to_search = [1] 
    
    # # Danh sách các giá trị `weight` của passage cần thử nghiệm
    # weights_to_search = [0, 0.05, 0.1]
    # options = ['max', 'average']
    # weights_options = ['global']
    # alpha_to_search = [0.2,0.3,0.5,0.7]
    # # --- PHẦN 3: THỰC THI VÒNG LẶP GRID SEARCH ---
    
    # print("🚀 Starting Grid Search for Subgraph PPR...")
    # for top_k_entities in top_k_entity_to_search:
    # # Vòng lặp ngoài: duyệt qua các giá trị depth
    #     for depth in depths_to_search:
    #         # Vòng lặp trong: duyệt qua các giá trị weight
    #         for weight in weights_to_search:
    #             for top_k_chunks in top_k_chunk_to_search:
    #                 for opt in options:
    #                     for weight_method in weights_options:
    #                         for alpha in alpha_to_search:
    #                             print("-" * 60)
    #                             print(f"🧪 Running experiment with: SUBGRAPH_DEPTH = {depth}, PASSAGE_NODE_WEIGHT = {weight}")
                                
    #                             # Tạo tên file output động để không bị ghi đè
    #                             # output_filename = f"test_popqa_subgraph_top_entity_{top_k_entities}_depth_{depth}_weight{weight}.json"
    #                             output_filename = f"test_nq_subgraph_{weight_method}_softmax_top_entity_{top_k_entities}_depth_{depth}_weight{weight}_top_chunk_per_entity_{top_k_chunks}_option_{opt}_wetght_of_entity_in_ec_{alpha}.json"
    #                             # output_filename = f"test_nq_subgraph_top_entity_{top_k_entities}_depth_{depth}_weight{weight}.json"
    #                             output_path = os.path.join(OUTPUT_DIR, output_filename)

    #                             # Gọi hàm xử lý chính với các tham số hiện tại
    #                             results = run_query_page_rank(
    #                                 working_dir_1=WORKING_DIR_1,
    #                                 query_path=QUERY_PATH,
    #                                 corpus_path=CORPUS_PATH,
    #                                 db_name=DB_NAME,
    #                                 model_path=MODEL_PATH,
    #                                 top_k_entities=top_k_entities,
    #                                 top_k_chunks_per_entity=top_k_chunks,
    #                                 # Sử dụng các giá trị từ vòng lặp
    #                                 subgraph_depth=depth,
    #                                 passage_node_weight=weight,
    #                                 options=opt,
    #                                 weight_method=weight_method,
    #                                 alpha=alpha
    #                             )

    #                             # Lưu kết quả của lần chạy này
    #                             print(f"💾 Saving results for (depth={depth}, weight={weight}) to {output_path}")
    #                             with open(output_path, "w", encoding="utf-8") as f:
    #                                 json.dump(results, f, ensure_ascii=False, indent=4)
    #                             print("✅ Done with this run.")

    # print("-" * 60)
    # print("🎉 Grid Search finished successfully!")

    WORKING_DIR_1 = "/home/hungpv/projects/TN/LIGHTRAG/new_popqa_en_without_embedding"
    QUERY_PATH = "/home/hungpv/projects/TN/data/data_popqa/dev_queries_vi.json"
    CORPUS_PATH = "/home/hungpv/projects/TN/data/data_popqa/filter_corpus_en.json"
    MODEL_PATH = "BAAI/bge-m3"
    DB_NAME = "bge-m3-cls"
    OUTPUT_DIR = "//home/hungpv/projects/ppr_weight/sub_graph_weight_assign_softmax_global" # Thư mục lưu kết quả

    TOP_K_ENTITIES = 10
    TOP_K_CHUNKS_PER_ENTITY = 10
    
    # --- PHẦN 2: ĐỊNH NGHĨA KHÔNG GIAN TÌM KIẾM (GRID SEARCH) ---
    
    top_k_chunk_to_search = [10]  # Có thể thêm các giá trị khác nếu muốn thử nghiệm

    top_k_entity_to_search = [10]  # Có thể thêm các giá trị khác nếu muốn thử nghiệm
    # Danh sách các giá trị `depth` cần thử nghiệm
    depths_to_search = [1] 
    
    # Danh sách các giá trị `weight` của passage cần thử nghiệm
    weights_to_search = [0, 0.05, 0.1]
    options = ['max', 'average']
    weights_options = ['global']
    alpha_to_search = [0.2,0.3,0.5,0.7]
    # --- PHẦN 3: THỰC THI VÒNG LẶP GRID SEARCH ---
    
    print("🚀 Starting Grid Search for Subgraph PPR...")
    for top_k_entities in top_k_entity_to_search:
    # Vòng lặp ngoài: duyệt qua các giá trị depth
        for depth in depths_to_search:
            # Vòng lặp trong: duyệt qua các giá trị weight
            for weight in weights_to_search:
                for top_k_chunks in top_k_chunk_to_search:
                    for opt in options:
                        for weight_method in weights_options:
                            for alpha in alpha_to_search:
                                print("-" * 60)
                                print(f"🧪 Running experiment with: SUBGRAPH_DEPTH = {depth}, PASSAGE_NODE_WEIGHT = {weight}")
                                
                                # Tạo tên file output động để không bị ghi đè
                                # output_filename = f"test_popqa_subgraph_top_entity_{top_k_entities}_depth_{depth}_weight{weight}.json"
                                output_filename = f"test_popqa_subgraph_{weight_method}_softmax_top_entity_{top_k_entities}_depth_{depth}_weight{weight}_top_chunk_per_entity_{top_k_chunks}_option_{opt}_wetght_of_entity_in_ec_{alpha}.json"
                                # output_filename = f"test_nq_subgraph_top_entity_{top_k_entities}_depth_{depth}_weight{weight}.json"
                                output_path = os.path.join(OUTPUT_DIR, output_filename)

                                # Gọi hàm xử lý chính với các tham số hiện tại
                                results = run_query_page_rank(
                                    working_dir_1=WORKING_DIR_1,
                                    query_path=QUERY_PATH,
                                    corpus_path=CORPUS_PATH,
                                    db_name=DB_NAME,
                                    model_path=MODEL_PATH,
                                    top_k_entities=top_k_entities,
                                    top_k_chunks_per_entity=top_k_chunks,
                                    # Sử dụng các giá trị từ vòng lặp
                                    subgraph_depth=depth,
                                    passage_node_weight=weight,
                                    options=opt,
                                    weight_method=weight_method,
                                    alpha=alpha
                                )

                                # Lưu kết quả của lần chạy này
                                print(f"💾 Saving results for (depth={depth}, weight={weight}) to {output_path}")
                                with open(output_path, "w", encoding="utf-8") as f:
                                    json.dump(results, f, ensure_ascii=False, indent=4)
                                print("✅ Done with this run.")

    print("-" * 60)
    print("🎉 Grid Search finished successfully!")


