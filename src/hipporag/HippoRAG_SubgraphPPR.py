import logging
from typing import List, Tuple, Dict, Optional, Set

import numpy as np
from igraph import Graph

from .HippoRAG import HippoRAG
from .utils.misc_utils import QuerySolution
from .prompts.linking import get_query_instruction
from .utils.misc_utils import min_max_normalize


logger = logging.getLogger(__name__)


class HippoRAGSubgraphPPR(HippoRAG):
    """
    Alternative retriever using a subgraph-based Personalized PageRank strategy inspired by
    PPR_ranking_on_subgraph.py. It reuses HippoRAG's existing embedding stores and graph data,
    but builds a light-weight per-query subgraph and runs PPR on it.

    Key differences vs default retrieve():
    - Selects top entities by query-to-fact similarity to form seeds
    - Gathers their connected chunks and filters to top-K per entity by dense scores
    - Assigns local-softmax edge weights per entity using a hybrid score of entity and chunk
    - Runs PPR on this subgraph with reset probabilities from entity and chunk scores
    """

    def retrieve(
        self,
        queries: List[str],
        num_to_retrieve: Optional[int] = None,
        gold_docs: Optional[List[List[str]]] = None,
        *,
        top_k_entities: int = 10,
        top_k_chunks_per_entity: int = 10,
        alpha: float = 0.5,
        local_softmax_temperature: float = 1.0,
        preselect_candidates: int = 200,
    ) -> List[QuerySolution] | Tuple[List[QuerySolution], Dict]:
        if num_to_retrieve is None:
            num_to_retrieve = self.global_config.retrieval_top_k

        if not self.ready_to_retrieve:
            self.prepare_retrieval_objects()

        self.get_query_embeddings(queries)

        results: List[QuerySolution] = []

        for query in queries:
            # 1) Entity scores by query-to-fact embedding
            q_triple_emb = self.query_to_embedding["triple"].get(query, None)
            if q_triple_emb is None:
                q_triple_emb = self.embedding_model.batch_encode(
                    query, instruction=get_query_instruction("query_to_fact"), norm=True
                )

            # Ensure entity embeddings exist (safety backfill)
            if len(self.entity_embeddings) == 0 or len(self.entity_node_keys) == 0:
                try:
                    all_openie_info, _ = self.load_existing_openie([])
                    unique_entities = set()
                    for doc in all_openie_info:
                        for t in doc.get("extracted_triples", []) or []:
                            if isinstance(t, (list, tuple)) and len(t) >= 3:
                                if isinstance(t[0], str) and t[0].strip():
                                    unique_entities.add(t[0])
                                if isinstance(t[2], str) and t[2].strip():
                                    unique_entities.add(t[2])
                        for e in doc.get("extracted_entities", []) or []:
                            if isinstance(e, str) and e.strip():
                                unique_entities.add(e)
                    if unique_entities:
                        self.entity_embedding_store.insert_strings(list(unique_entities))
                        self.entity_node_keys = list(self.entity_embedding_store.get_all_ids())
                        self.entity_embeddings = np.array(
                            self.entity_embedding_store.get_embeddings(self.entity_node_keys)
                        )
                except Exception as e:
                    logger.warning(f"Failed to backfill entity embeddings in retriever: {e}")

            if len(self.entity_embeddings) == 0:
                logger.warning("No entity embeddings available; falling back to DPR.")
                sorted_doc_ids, sorted_doc_scores = self.dense_passage_retrieval(query)
                docs = [
                    self.chunk_embedding_store.get_row(self.passage_node_keys[idx])["content"]
                    for idx in sorted_doc_ids[:num_to_retrieve]
                ]
                results.append(
                    QuerySolution(question=query, docs=docs, doc_scores=sorted_doc_scores[:num_to_retrieve])
                )
                continue

            entity_scores = np.dot(self.entity_embeddings, q_triple_emb.T)
            entity_scores = np.squeeze(entity_scores) if entity_scores.ndim == 2 else entity_scores
            entity_scores = min_max_normalize(entity_scores)

            # top entities
            ent_sorted_idx = np.argsort(entity_scores)[::-1].tolist()
            seed_ent_global_idx = ent_sorted_idx[: max(1, int(top_k_entities))]
            seed_ent_node_keys = [self.entity_node_keys[i] for i in seed_ent_global_idx]
            seed_ent_scores = {self.entity_node_keys[i]: float(entity_scores[i]) for i in seed_ent_global_idx}

            # 2) Dense chunk scores for query and preselect candidate chunks
            dpr_sorted_doc_ids, dpr_sorted_doc_scores = self.dense_passage_retrieval(query)
            dpr_sorted_doc_scores = np.squeeze(dpr_sorted_doc_scores)
            candidate_ids = dpr_sorted_doc_ids.tolist()[: max(1, int(preselect_candidates))]
            candidate_keys = [self.passage_node_keys[idx] for idx in candidate_ids]
            chunk_scores = {self.passage_node_keys[idx]: float(score) for idx, score in zip(dpr_sorted_doc_ids.tolist(), dpr_sorted_doc_scores.tolist())}

            # 3) Build per-query subgraph nodes: selected entities + their top chunks
            selected_chunk_keys: Set[str] = set()
            ent_to_chunks: Dict[str, List[str]] = {}
            for ent_key in seed_ent_node_keys:
                candidate_chunks_all = list(self.ent_node_to_chunk_ids.get(ent_key, set()))
                # restrict to DPR candidates only
                candidate_chunks = [ck for ck in candidate_chunks_all if ck in set(candidate_keys)]
                # rank candidate chunks by query dense score
                candidate_chunks.sort(key=lambda ck: chunk_scores.get(ck, 0.0), reverse=True)
                keep = candidate_chunks[: max(1, int(top_k_chunks_per_entity))]
                ent_to_chunks[ent_key] = keep
                selected_chunk_keys.update(keep)

            if not selected_chunk_keys:
                logger.info("No chunk neighbors selected; falling back to DPR.")
                sorted_doc_ids, sorted_doc_scores = self.dense_passage_retrieval(query)
                docs = [
                    self.chunk_embedding_store.get_row(self.passage_node_keys[idx])["content"]
                    for idx in sorted_doc_ids[:num_to_retrieve]
                ]
                results.append(
                    QuerySolution(question=query, docs=docs, doc_scores=sorted_doc_scores[:num_to_retrieve])
                )
                continue

            # 4) Construct igraph for this subgraph
            node_keys = seed_ent_node_keys + list(selected_chunk_keys)
            node_is_entity = {k: (k in seed_ent_node_keys) for k in node_keys}
            key_to_idx = {k: i for i, k in enumerate(node_keys)}
            g = Graph(directed=False)
            g.add_vertices(len(node_keys))

            # add edges: entity-entity if present in backbone stats; entity-chunk from ent_to_chunks
            edges = []
            ee_scores_local: Dict[int, List[float]] = {}
            ec_scores_local: Dict[int, List[float]] = {}
            adj_by_src: Dict[int, List[int]] = {}

            # entity-entity edges
            for i, u_key in enumerate(seed_ent_node_keys):
                for v_key in seed_ent_node_keys:
                    if u_key == v_key:
                        continue
                    # use existing co-occurrence/count if available
                    raw = float(self.node_to_node_stats.get((u_key, v_key), 0.0))
                    if raw <= 0:
                        continue
                    u = key_to_idx[u_key]
                    v = key_to_idx[v_key]
                    if u < v:
                        edges.append((u, v))
                    # store per-source score for local softmax (undirected treated as both ways)
                    ee_scores_local.setdefault(u, []).append(raw)
                    ee_scores_local.setdefault(v, []).append(raw)
                    adj_by_src.setdefault(u, []).append(v)
                    adj_by_src.setdefault(v, []).append(u)

            # entity-chunk edges with hybrid scores
            for ent_key, chunks in ent_to_chunks.items():
                u = key_to_idx[ent_key]
                for ch_key in chunks:
                    if ch_key not in key_to_idx:
                        continue
                    v = key_to_idx[ch_key]
                    if u < v:
                        edges.append((u, v))
                    # hybrid score per PPR_ranking idea: alpha*entity + (1-alpha)*chunk
                    s_ent = seed_ent_scores.get(ent_key, 0.0)
                    s_ch = chunk_scores.get(ch_key, 0.0)
                    hybrid = (alpha * s_ent) + ((1.0 - alpha) * s_ch)
                    ec_scores_local.setdefault(u, []).append(hybrid)
                    ec_scores_local.setdefault(v, []).append(hybrid)
                    adj_by_src.setdefault(u, []).append(v)
                    adj_by_src.setdefault(v, []).append(u)

            if edges:
                g.add_edges(edges)

            # 5) Assign local-softmax weights per entity node over its outgoing edges
            # initialize small epsilon for edges without weights
            weights = [1e-6] * g.ecount()
            eid_lookup = {(min(s, t), max(s, t)): eid for eid, (s, t) in enumerate(g.get_edgelist())}

            def set_edge_weight(src: int, dst: int, w: float):
                a, b = (src, dst) if src < dst else (dst, src)
                eid = eid_lookup.get((a, b))
                if eid is not None:
                    weights[eid] = float(w)

            temperature = max(1e-6, float(local_softmax_temperature))
            for src, neighbors in adj_by_src.items():
                raw_scores = []
                for dst in neighbors:
                    # choose appropriate raw score list placeholder
                    if node_is_entity.get(src, False) and node_is_entity.get(dst, False):
                        # entity-entity: pull corresponding value by index order
                        # We approximate with average of both sides if lengths differ
                        idx = adj_by_src[src].index(dst)
                        raw_array = ee_scores_local.get(src, [])
                        raw = raw_array[idx] if idx < len(raw_array) else (np.mean(raw_array) if raw_array else 0.0)
                    else:
                        idx = adj_by_src[src].index(dst)
                        raw_array = ec_scores_local.get(src, [])
                        raw = raw_array[idx] if idx < len(raw_array) else (np.mean(raw_array) if raw_array else 0.0)
                    raw_scores.append(raw)

                if not raw_scores:
                    continue
                logits = np.array(raw_scores) / temperature
                # softmax
                exps = np.exp(logits - np.max(logits))
                probs = exps / (np.sum(exps) + 1e-12)
                for dst, w in zip(neighbors, probs.tolist()):
                    set_edge_weight(src, dst, w)

            g.es["weight"] = weights

            # 6) Build reset vector: entity -> entity_score, chunk -> passage_node_weight * chunk_score
            reset = np.zeros(g.vcount(), dtype=float)
            for key, idx in key_to_idx.items():
                if node_is_entity.get(key, False):
                    reset[idx] = seed_ent_scores.get(key, 0.0)
                else:
                    reset[idx] = self.global_config.passage_node_weight * chunk_scores.get(key, 0.0)

            # 7) Run PPR on subgraph
            reset = np.where(np.isnan(reset) | (reset < 0), 0, reset)
            damping = self.global_config.damping if getattr(self.global_config, "damping", None) is not None else 0.85
            pr = g.personalized_pagerank(
                vertices=range(g.vcount()),
                damping=float(damping),
                directed=False,
                weights="weight",
                reset=reset,
                implementation="prpack",
            )

            # 8) Rank chunk nodes only
            chunk_keys_in_graph = [key for key in node_keys if not node_is_entity.get(key, False)]
            chunk_indices = [key_to_idx[k] for k in chunk_keys_in_graph]
            chunk_scores_pr = np.array([pr[idx] for idx in chunk_indices])
            order = np.argsort(chunk_scores_pr)[::-1]
            sorted_chunk_keys = [chunk_keys_in_graph[i] for i in order.tolist()]
            sorted_chunk_scores = chunk_scores_pr[order].tolist()

            docs = [self.chunk_embedding_store.get_row(k)["content"] for k in sorted_chunk_keys[:num_to_retrieve]]
            results.append(
                QuerySolution(
                    question=query,
                    docs=docs,
                    doc_scores=sorted_chunk_scores[:num_to_retrieve],
                )
            )

        if gold_docs is None:
            return results
        # If evaluation is needed, reuse parent evaluation utility (macro recall etc.)
        from .evaluation.retrieval_eval import RetrievalRecall

        retrieval_recall_evaluator = RetrievalRecall(global_config=self.global_config)
        k_list = [1, 2, 5, 10, 20, 30, 50, 100]
        overall_retrieval_result, _ = retrieval_recall_evaluator.calculate_metric_scores(
            gold_docs=gold_docs,
            retrieved_docs=[r.docs for r in results],
            k_list=k_list,
        )
        return results, overall_retrieval_result


    def retrieve_ppr_advanced(
        self,
        queries: List[str],
        num_to_retrieve: Optional[int] = None,
        gold_docs: Optional[List[List[str]]] = None,
        *,
        top_k_entities: int = 10,
        top_k_chunks_per_entity: int = 10,
        preselect_entity_candidates: int = 5000,
        preselect_chunk_candidates: int = 500,
        per_entity_pair_agg_top: int = 100,
        aggregation: str = "max",  # "max" or "average" over top pair scores
        weight_method: str = "local",  # "local" or "global"
        alpha: float = 0.5,
        local_softmax_temperature: float = 1.0,
    ) -> List[QuerySolution] | Tuple[List[QuerySolution], Dict]:
        """
        Advanced subgraph PPR retriever inspired by PPR_ranking_on_subgraph.py.

        Differences vs default Hippo graph_search:
        - Computes entity scores from top pair (entity, chunk) scores per entity
        - Builds subgraph of seed entities + their top chunks (restricted to DPR-preselected)
        - Assigns edge weights using either local softmax per entity or global hybrid scores
        - Runs PPR on this subgraph
        """
        if num_to_retrieve is None:
            num_to_retrieve = self.global_config.retrieval_top_k

        if not self.ready_to_retrieve:
            self.prepare_retrieval_objects()

        self.get_query_embeddings(queries)

        results: List[QuerySolution] = []

        # Convenience
        passage_node_weight = getattr(self.global_config, "passage_node_weight", 0.05)

        for query in queries:
            # Query embeddings
            q_triple_emb = self.query_to_embedding["triple"].get(query, None)
            if q_triple_emb is None:
                q_triple_emb = self.embedding_model.batch_encode(
                    query, instruction=get_query_instruction("query_to_fact"), norm=True
                )

            # Safety: ensure entity embeddings
            if len(self.entity_embeddings) == 0 or len(self.entity_node_keys) == 0:
                logger.warning("No entity embeddings available; falling back to DPR.")
                sorted_doc_ids, sorted_doc_scores = self.dense_passage_retrieval(query)
                docs = [
                    self.chunk_embedding_store.get_row(self.passage_node_keys[idx])["content"]
                    for idx in sorted_doc_ids[:num_to_retrieve]
                ]
                results.append(
                    QuerySolution(question=query, docs=docs, doc_scores=sorted_doc_scores[:num_to_retrieve])
                )
                continue

            # Base entity scores from query-to-entity embedding
            base_entity_scores = np.dot(self.entity_embeddings, q_triple_emb.T)
            base_entity_scores = (
                np.squeeze(base_entity_scores) if base_entity_scores.ndim == 2 else base_entity_scores
            )
            base_entity_scores = min_max_normalize(base_entity_scores)

            # Preselect candidate entities by base scores
            ent_sorted_idx = np.argsort(base_entity_scores)[::-1].tolist()
            ent_candidate_idx = ent_sorted_idx[: max(1, int(preselect_entity_candidates))]
            ent_candidate_keys = [self.entity_node_keys[i] for i in ent_candidate_idx]

            # DPR chunk scores and candidate chunks
            dpr_sorted_doc_ids, dpr_sorted_doc_scores = self.dense_passage_retrieval(query)
            dpr_sorted_doc_scores = np.squeeze(dpr_sorted_doc_scores)
            chunk_scores = {
                self.passage_node_keys[idx]: float(score)
                for idx, score in zip(dpr_sorted_doc_ids.tolist(), dpr_sorted_doc_scores.tolist())
            }
            candidate_chunk_keys = set(
                [self.passage_node_keys[idx] for idx in dpr_sorted_doc_ids.tolist()[: max(1, int(preselect_chunk_candidates))]]
            )

            # Build pair scores (entity, chunk) using DPR chunk score restricted by mapping
            entity_chunk_pair_scores: Dict[Tuple[str, str], float] = {}
            for ent_key in ent_candidate_keys:
                mapped_chunks = self.ent_node_to_chunk_ids.get(ent_key, set())
                if not mapped_chunks:
                    continue
                for ch_key in mapped_chunks:
                    if ch_key in candidate_chunk_keys:
                        pair_score = chunk_scores.get(ch_key, 0.0)
                        if pair_score > 0:
                            entity_chunk_pair_scores[(ent_key, ch_key)] = pair_score

            # Aggregate per-entity scores from top pair scores
            per_entity_raw: Dict[str, List[float]] = {}
            for (e_key, c_key), sc in entity_chunk_pair_scores.items():
                per_entity_raw.setdefault(e_key, []).append(sc)

            per_entity_agg: Dict[str, float] = {}
            for e_key, vals in per_entity_raw.items():
                vals.sort(reverse=True)
                top_vals = vals[: max(1, int(per_entity_pair_agg_top))]
                if aggregation == "max":
                    per_entity_agg[e_key] = float(max(top_vals)) if top_vals else 0.0
                else:
                    per_entity_agg[e_key] = float(sum(top_vals) / len(top_vals)) if top_vals else 0.0

            if not per_entity_agg:
                logger.info("No entity pair scores; falling back to DPR.")
                sorted_doc_ids, sorted_doc_scores = self.dense_passage_retrieval(query)
                docs = [
                    self.chunk_embedding_store.get_row(self.passage_node_keys[idx])["content"]
                    for idx in sorted_doc_ids[:num_to_retrieve]
                ]
                results.append(
                    QuerySolution(question=query, docs=docs, doc_scores=sorted_doc_scores[:num_to_retrieve])
                )
                continue

            # Normalize entity agg scores and select seeds
            agg_vals = np.array(list(per_entity_agg.values()), dtype=float)
            agg_norm = min_max_normalize(agg_vals) if len(agg_vals) > 0 else agg_vals
            per_entity_norm = {k: float(v) for k, v in zip(per_entity_agg.keys(), agg_norm.tolist())}
            seed_entities_sorted = sorted(per_entity_norm.items(), key=lambda x: x[1], reverse=True)
            seed_entities = dict(seed_entities_sorted[: max(1, int(top_k_entities))])
            seed_ent_keys = list(seed_entities.keys())

            if not seed_ent_keys:
                logger.info("No seed entities; falling back to DPR.")
                sorted_doc_ids, sorted_doc_scores = self.dense_passage_retrieval(query)
                docs = [
                    self.chunk_embedding_store.get_row(self.passage_node_keys[idx])["content"]
                    for idx in sorted_doc_ids[:num_to_retrieve]
                ]
                results.append(
                    QuerySolution(question=query, docs=docs, doc_scores=sorted_doc_scores[:num_to_retrieve])
                )
                continue

            # Select top chunks per seed entity by DPR chunk score
            ent_to_chunks: Dict[str, List[str]] = {}
            selected_chunk_keys: Set[str] = set()
            for e_key in seed_ent_keys:
                mapped_chunks = [ck for ck in self.ent_node_to_chunk_ids.get(e_key, set()) if ck in candidate_chunk_keys]
                mapped_chunks.sort(key=lambda ck: chunk_scores.get(ck, 0.0), reverse=True)
                keep = mapped_chunks[: max(1, int(top_k_chunks_per_entity))]
                ent_to_chunks[e_key] = keep
                selected_chunk_keys.update(keep)

            if not selected_chunk_keys:
                logger.info("No chunk neighbors selected; falling back to DPR.")
                sorted_doc_ids, sorted_doc_scores = self.dense_passage_retrieval(query)
                docs = [
                    self.chunk_embedding_store.get_row(self.passage_node_keys[idx])["content"]
                    for idx in sorted_doc_ids[:num_to_retrieve]
                ]
                results.append(
                    QuerySolution(question=query, docs=docs, doc_scores=sorted_doc_scores[:num_to_retrieve])
                )
                continue

            # Build subgraph
            node_keys = seed_ent_keys + list(selected_chunk_keys)
            node_is_entity = {k: (k in seed_ent_keys) for k in node_keys}
            key_to_idx = {k: i for i, k in enumerate(node_keys)}
            g = Graph(directed=False)
            g.add_vertices(len(node_keys))

            # Edges and adjacency
            edges = []
            adj_by_src: Dict[int, List[int]] = {}
            ee_relation_scores_local: Dict[int, List[float]] = {}
            ec_pair_scores_local: Dict[int, List[float]] = {}

            # E-E edges with relation scores from node_to_node_stats
            for i, u_key in enumerate(seed_ent_keys):
                for v_key in seed_ent_keys:
                    if u_key == v_key:
                        continue
                    raw_rel = float(self.node_to_node_stats.get((u_key, v_key), 0.0))
                    if raw_rel <= 0:
                        continue
                    u = key_to_idx[u_key]
                    v = key_to_idx[v_key]
                    if u < v:
                        edges.append((u, v))
                    ee_relation_scores_local.setdefault(u, []).append(raw_rel)
                    ee_relation_scores_local.setdefault(v, []).append(raw_rel)
                    adj_by_src.setdefault(u, []).append(v)
                    adj_by_src.setdefault(v, []).append(u)

            # E-C edges with hybrid pair scoring
            for e_key, chunks in ent_to_chunks.items():
                u = key_to_idx[e_key]
                for ch_key in chunks:
                    if ch_key not in key_to_idx:
                        continue
                    v = key_to_idx[ch_key]
                    if u < v:
                        edges.append((u, v))
                    pair = entity_chunk_pair_scores.get((e_key, ch_key), 0.0)
                    s_ch = chunk_scores.get(ch_key, 0.0)
                    hybrid = (alpha * pair) + ((1.0 - alpha) * s_ch)
                    ec_pair_scores_local.setdefault(u, []).append(hybrid)
                    ec_pair_scores_local.setdefault(v, []).append(hybrid)
                    adj_by_src.setdefault(u, []).append(v)
                    adj_by_src.setdefault(v, []).append(u)

            if edges:
                g.add_edges(edges)

            # Edge weights
            weights = [1e-6] * g.ecount()
            eid_lookup = {(min(s, t), max(s, t)): eid for eid, (s, t) in enumerate(g.get_edgelist())}

            def set_edge_weight(src: int, dst: int, w: float):
                a, b = (src, dst) if src < dst else (dst, src)
                eid = eid_lookup.get((a, b))
                if eid is not None:
                    weights[eid] = float(w)

            if weight_method == "local":
                temperature = max(1e-6, float(local_softmax_temperature))
                for src, neighbors in adj_by_src.items():
                    raw_scores = []
                    for dst in neighbors:
                        if node_is_entity.get(src, False) and node_is_entity.get(dst, False):
                            idx = adj_by_src[src].index(dst)
                            raw_array = ee_relation_scores_local.get(src, [])
                            raw = raw_array[idx] if idx < len(raw_array) else (np.mean(raw_array) if raw_array else 0.0)
                        else:
                            idx = adj_by_src[src].index(dst)
                            raw_array = ec_pair_scores_local.get(src, [])
                            raw = raw_array[idx] if idx < len(raw_array) else (np.mean(raw_array) if raw_array else 0.0)
                        raw_scores.append(raw)

                    if not raw_scores:
                        continue
                    logits = np.array(raw_scores) / temperature
                    exps = np.exp(logits - np.max(logits))
                    probs = exps / (np.sum(exps) + 1e-12)
                    for dst, w in zip(neighbors, probs.tolist()):
                        set_edge_weight(src, dst, w)
            else:
                # global: directly set hybrid scores proportionally (will be normalized by PR)
                for src, neighbors in adj_by_src.items():
                    for dst in neighbors:
                        if node_is_entity.get(src, False) and node_is_entity.get(dst, False):
                            idx = adj_by_src[src].index(dst)
                            raw_array = ee_relation_scores_local.get(src, [])
                            raw = raw_array[idx] if idx < len(raw_array) else (np.mean(raw_array) if raw_array else 0.0)
                            set_edge_weight(src, dst, raw)
                        else:
                            idx = adj_by_src[src].index(dst)
                            raw_array = ec_pair_scores_local.get(src, [])
                            raw = raw_array[idx] if idx < len(raw_array) else (np.mean(raw_array) if raw_array else 0.0)
                            set_edge_weight(src, dst, raw)

            g.es["weight"] = weights

            # Reset vector
            reset = np.zeros(g.vcount(), dtype=float)
            for key, idx in key_to_idx.items():
                if node_is_entity.get(key, False):
                    reset[idx] = seed_entities.get(key, 0.0)
                else:
                    reset[idx] = passage_node_weight * chunk_scores.get(key, 0.0)

            # PPR
            reset = np.where(np.isnan(reset) | (reset < 0), 0, reset)
            damping = self.global_config.damping if getattr(self.global_config, "damping", None) is not None else 0.85
            pr = g.personalized_pagerank(
                vertices=range(g.vcount()),
                damping=float(damping),
                directed=False,
                weights="weight",
                reset=reset,
                implementation="prpack",
            )

            # Rank chunk nodes
            chunk_keys_in_graph = [key for key in node_keys if not node_is_entity.get(key, False)]
            chunk_indices = [key_to_idx[k] for k in chunk_keys_in_graph]
            chunk_scores_pr = np.array([pr[idx] for idx in chunk_indices])
            order = np.argsort(chunk_scores_pr)[::-1]
            sorted_chunk_keys = [chunk_keys_in_graph[i] for i in order.tolist()]
            sorted_chunk_scores = chunk_scores_pr[order].tolist()

            docs = [self.chunk_embedding_store.get_row(k)["content"] for k in sorted_chunk_keys[:num_to_retrieve]]
            results.append(
                QuerySolution(
                    question=query,
                    docs=docs,
                    doc_scores=sorted_chunk_scores[:num_to_retrieve],
                )
            )

        if gold_docs is None:
            return results
        from .evaluation.retrieval_eval import RetrievalRecall
        retrieval_recall_evaluator = RetrievalRecall(global_config=self.global_config)
        k_list = [1, 2, 5, 10, 20, 30, 50, 100]
        overall_retrieval_result, _ = retrieval_recall_evaluator.calculate_metric_scores(
            gold_docs=gold_docs,
            retrieved_docs=[r.docs for r in results],
            k_list=k_list,
        )
        return results, overall_retrieval_result

