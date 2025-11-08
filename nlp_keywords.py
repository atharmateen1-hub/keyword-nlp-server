import math
from collections import Counter, defaultdict
from typing import List, Dict, Any
import sys
import json

import spacy
from sklearn.cluster import KMeans
import numpy as np

# Load spaCy model once
nlp = spacy.load("en_core_web_md")

SEO_USELESS_SINGLE = {
    "miss", "find", "click", "read", "see", "watch", "start", "go", "get",
    "know", "make", "take", "give", "use", "want", "need", "like",
    "new", "latest", "current", "next", "previous", "more", "other",
    "registration", "update", "announcement", "news", "info",
    "page", "post", "blog", "site", "website",
    "today", "now", "year", "month", "week", "day",
}

DOMAIN_WHITELIST_SINGLE = {
    "seo", "ielts", "english", "wordpress", "cambridge", "tefl"
}

NOISE_PHRASES = {
    "read more", "continue reading", "next", "previous", "older posts",
    "newer posts", "latest posts", "leave a comment", "no comments yet",
    "add a comment", "reply", "share this", "click here", "login",
    "log in", "register", "sign in", "sign up", "privacy policy",
    "terms of service", "terms and conditions", "cookie policy",
    "related posts", "related articles", "related content",
    "posted by", "posted in"
}

def normalize_phrase(text: str) -> str:
    doc = nlp(text)
    return " ".join(tok.text.lower() for tok in doc if tok.is_alpha and not tok.is_stop)

def is_noise_phrase(text: str) -> bool:
    t = text.lower().strip()
    if not t or len(t) < 3:
        return True
    if t in NOISE_PHRASES:
        return True
    return False

def add_candidate(candidates: Dict[str, Dict[str, Any]],
                  phrase: str, url: str,
                  title_text: str, h1_text: str, h2h3_text: str,
                  seen_in_doc: set):
    tokens = phrase.split()
    if len(tokens) == 1:
        t = tokens[0]
        if t in SEO_USELESS_SINGLE or t.isdigit():
            return
        if t not in DOMAIN_WHITELIST_SINGLE:
            return

    key = phrase
    if key not in candidates:
        candidates[key] = {
            "phrase": phrase,
            "total_count": 0,
            "doc_count": 0,
            "in_title": False,
            "in_h1": False,
            "in_h2_h3": False,
            "urls": set(),
        }

    candidates[key]["total_count"] += 1
    if key not in seen_in_doc:
        seen_in_doc.add(key)
        candidates[key]["doc_count"] += 1
        candidates[key]["urls"].add(url)

    if phrase in title_text:
        candidates[key]["in_title"] = True
    if phrase in h1_text:
        candidates[key]["in_h1"] = True
    if phrase in h2h3_text:
        candidates[key]["in_h2_h3"] = True

def extract_candidates(spacy_docs, focus_keyword: str):
    candidates = {}

    for entry in spacy_docs:
        url = entry["url"]
        doc = entry["doc"]
        title = entry["title"].lower()
        h1 = " ".join(entry.get("h1", [])).lower()
        h2h3 = " ".join(entry.get("h2", []) + entry.get("h3", [])).lower()

        seen_in_doc = set()

        for chunk in doc.noun_chunks:
            phrase = normalize_phrase(chunk.text)
            if not phrase or is_noise_phrase(phrase):
                continue
            add_candidate(candidates, phrase, url, title, h1, h2h3, seen_in_doc)

        for ent in doc.ents:
            if ent.label_ in {"ORG", "PRODUCT", "EVENT", "WORK_OF_ART", "GPE", "FAC"}:
                phrase = normalize_phrase(ent.text)
                if not phrase or is_noise_phrase(phrase):
                    continue
                add_candidate(candidates, phrase, url, title, h1, h2h3, seen_in_doc)

        tokens = [t for t in doc if t.is_alpha and not t.is_space]
        N = len(tokens)
        for n in range(3, 6):
            for i in range(0, N - n + 1):
                span = tokens[i:i+n]
                phrase = normalize_phrase(" ".join([t.text for t in span]))
                if not phrase or is_noise_phrase(phrase):
                    continue
                add_candidate(candidates, phrase, url, title, h1, h2h3, seen_in_doc)

    return candidates

def score_candidates(candidates, spacy_docs, focus_keyword: str):
    max_count = max(c["total_count"] for c in candidates.values()) or 1
    max_doc_count = max(c["doc_count"] for c in candidates.values()) or 1

    focus_doc = nlp(focus_keyword)
    focus_norm = normalize_phrase(focus_keyword)

    co_counts = Counter()
    sentences_with_focus = 0

    for entry in spacy_docs:
        doc = entry["doc"]
        for sent in doc.sents:
            sent_text = sent.text.lower()
            if focus_norm and focus_norm in sent_text:
                sentences_with_focus += 1
                for phrase in candidates.keys():
                    if phrase in sent_text:
                        co_counts[phrase] += 1

    scored = []
    for phrase, data in candidates.items():
        total = data["total_count"]
        doc_count = data["doc_count"]

        freq_score = total / max_count
        doc_score = doc_count / max_doc_count

        heading_score = 0.0
        if data["in_title"]:
            heading_score += 0.4
        if data["in_h1"]:
            heading_score += 0.3
        if data["in_h2_h3"]:
            heading_score += 0.2
        heading_score = min(1.0, heading_score)

        if sentences_with_focus > 0:
            co_score = co_counts[phrase] / sentences_with_focus
        else:
            co_score = 0.0

        sim_score = nlp(phrase).similarity(focus_doc)

        semantic_score = (
            0.3 * freq_score +
            0.2 * doc_score +
            0.2 * heading_score +
            0.2 * sim_score +
            0.1 * co_score
        )
        semantic_score = float(min(1.0, max(0.0, semantic_score)))

        is_semantic = (sim_score >= 0.7 and semantic_score >= 0.4)
        is_sli = (co_score >= 0.3 or doc_score >= 0.3)
        is_nlp = True

        scored.append({
            "phrase": phrase,
            "total_count": total,
            "doc_count": doc_count,
            "semantic_score": semantic_score,
            "sim_score": sim_score,
            "co_score": co_score,
            "doc_score": doc_score,
            "heading_score": heading_score,
            "is_semantic": is_semantic,
            "is_sli": is_sli,
            "is_nlp": is_nlp,
        })

    return scored

def filter_candidates(scored, min_score=0.3):
    result = []
    for kw in scored:
        phrase = kw["phrase"]
        tokens = phrase.split()

        if kw["semantic_score"] < min_score:
            continue

        if len(tokens) == 1:
            t = tokens[0]
            if t in SEO_USELESS_SINGLE or t.isdigit():
                continue
            if t not in DOMAIN_WHITELIST_SINGLE and kw["semantic_score"] < 0.6:
                continue

        result.append(kw)

    result.sort(key=lambda x: x["semantic_score"], reverse=True)
    return result

def cluster_keywords(filtered):
    phrases = [kw["phrase"] for kw in filtered]
    docs = [nlp(p) for p in phrases]
    vectors = np.vstack([d.vector for d in docs])

    n_phrases = len(phrases)
    n_clusters = max(2, min(10, n_phrases // 10))

    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    labels = kmeans.fit_predict(vectors)

    clusters = defaultdict(list)
    for kw, label in zip(filtered, labels):
        kw["cluster_id"] = int(label) + 1
        clusters[int(label) + 1].append(kw)

    result = []
    for cid, items in clusters.items():
        result.append({
            "cluster_id": cid,
            "keywords": items
        })
    return result

def build_keyword_map(clusters):
    keyword_map = []

    for cluster in clusters:
        cid = cluster["cluster_id"]
        items = cluster["keywords"]
        items_sorted = sorted(items, key=lambda x: x["semantic_score"], reverse=True)

        main = None
        body_keywords = []

        for kw in items_sorted:
            phrase = kw["phrase"]
            tokens = phrase.split()
            if len(tokens) >= 2:
                main = kw
                break

        if not main:
            cand = items_sorted[0]
            if len(cand["phrase"].split()) == 1 and cand["phrase"] in SEO_USELESS_SINGLE:
                continue
            main = cand

        main_phrase = main["phrase"]

        for kw in items_sorted:
            if kw is main:
                continue
            if kw["semantic_score"] < 0.25:
                continue
            body_keywords.append(kw["phrase"])

        body_keywords = sorted(set(body_keywords))

        if not body_keywords:
            continue

        keyword_map.append({
            "cluster_id": cid,
            "main_keyword": main_phrase,
            "body_keywords": body_keywords
        })

    keyword_map.sort(key=lambda x: x["cluster_id"])
    return keyword_map

def process_input(data):
    focus_keyword = data["focus_keyword"].lower().strip()
    docs = data["documents"]

    spacy_docs = []
    for d in docs:
        full_text = " ".join(d.get("paragraphs", []))
        if not full_text.strip():
            continue
        spacy_docs.append({
            "url": d.get("url"),
            "title": d.get("title", ""),
            "h1": d.get("h1", []),
            "h2": d.get("h2", []),
            "h3": d.get("h3", []),
            "doc": nlp(full_text)
        })

    if not spacy_docs:
        return {"keywords": [], "clusters": [], "keyword_map": []}

    candidates = extract_candidates(spacy_docs, focus_keyword)
    scored = score_candidates(candidates, spacy_docs, focus_keyword)
    filtered = filter_candidates(scored, min_score=0.3)

    if not filtered:
        return {"keywords": [], "clusters": [], "keyword_map": []}

    clusters = cluster_keywords(filtered)
    keyword_map = build_keyword_map(clusters)

    keywords_list = []
    for kw in filtered:
        keywords_list.append({
            "phrase": kw["phrase"],
            "total_count": kw["total_count"],
            "doc_count": kw["doc_count"],
            "semantic_score": kw["semantic_score"],
            "cluster_id": kw.get("cluster_id"),
            "is_semantic": kw["is_semantic"],
            "is_sli": kw["is_sli"],
            "is_nlp": kw["is_nlp"],
        })

    return {
        "keywords": keywords_list,
        "clusters": clusters,
        "keyword_map": keyword_map,
    }

if __name__ == "__main__":
    data = json.load(sys.stdin)
    out = process_input(data)
    json.dump(out, sys.stdout, ensure_ascii=False)
