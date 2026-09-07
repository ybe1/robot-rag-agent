import argparse
import json
import sys
from time import perf_counter
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

CASES_PATH = PROJECT_ROOT / "evaluation" / "retrieval_cases.jsonl"
RESULTS_DIRECTORY = PROJECT_ROOT / "evaluation" / "results"


def parse_args():
    parser = argparse.ArgumentParser(
        description="评测指定向量库配置的 Top-K 检索效果"
    )
    parser.add_argument(
        "--profile",
        choices=("baseline", "structured", "hybrid"),
        default="baseline",
        help="选择评测配置，默认使用 baseline",
    )
    return parser.parse_args()


def get_result_path(profile_name: str) -> Path:
    return RESULTS_DIRECTORY / f"{profile_name}_multievidence.json"


def load_cases() -> list[dict]:
    cases = []

    with CASES_PATH.open("r", encoding="utf-8") as file:
        for line_number, line in enumerate(file, start=1):
            line = line.strip()

            if not line:
                continue

            try:
                cases.append(json.loads(line))

            except json.JSONDecodeError as exc:
                raise ValueError(
                    f"评测文件第 {line_number} 行不是有效的 JSON"
                ) from exc

    for case in cases:
        if case["answerable"]:
            for evidence in get_evidence_groups(case):
                keywords = evidence["expected_keywords"]
                if (
                    not evidence["expected_source"]
                    or not isinstance(keywords, list)
                    or not keywords
                    or any(not isinstance(word, str) or not word.strip() for word in keywords)
                ):
                    raise ValueError(f"题目 {case['id']} 的证据来源或关键词无效")

    return cases


def get_evidence_groups(case: dict) -> list[dict]:
    """原有标注作为第一组，追加人工审核过的其他证据。"""
    if not case["answerable"]:
        return []
    return [{
        "expected_source": case["expected_source"],
        "expected_keywords": case["expected_keywords"],
        "evidence_quote": case.get("evidence_quote"),
    }, *case.get("alternative_evidence", [])]


def get_source_name(document) -> str:
    source = document.metadata.get("source", "")
    return str(source).replace("\\", "/").rsplit("/", 1)[-1]


def find_evidence_match(documents, case: dict) -> tuple[int | None, int | None]:
    """返回首个匹配片段的排名和证据组编号（均从1开始）。"""
    groups = get_evidence_groups(case)

    for rank, document in enumerate(documents, start=1):
        source_name = get_source_name(document)
        content = document.page_content

        for group_index, evidence in enumerate(groups, start=1):
            keywords = evidence["expected_keywords"]
            if (
                source_name == evidence["expected_source"]
                and keywords
                and all(keyword in content for keyword in keywords)
            ):
                return rank, group_index

    return None, None


def evaluate_case(retriever, case: dict) -> dict:
    documents = retriever.invoke(case["question"])

    if case["answerable"]:
        evidence_rank, matched_group = find_evidence_match(documents, case)
        passed = evidence_rank is not None
        needs_manual_review = False
    else:
        # 当前检索器固定返回Top-K，暂时无法自动判断拒答是否正确。
        evidence_rank = None
        matched_group = None
        passed = None
        needs_manual_review = True

    return {
        "id": case["id"],
        "question": case["question"],
        "category": case["category"],
        "answerable": case["answerable"],
        "passed": passed,
        "evidence_rank": evidence_rank,
        "matched_evidence_group": matched_group,
        "evidence_groups": get_evidence_groups(case),
        "needs_manual_review": needs_manual_review,
        "expected_source": case["expected_source"],
        "expected_keywords": case["expected_keywords"],
        "retrieved_documents": [
            {
                "rank": rank,
                "source": get_source_name(document),
                "content": document.page_content,
            }
            for rank, document in enumerate(documents, start=1)
        ],
    }


def save_results(summary: dict, result_path: Path):
    RESULTS_DIRECTORY.mkdir(parents=True, exist_ok=True)

    with result_path.open("w", encoding="utf-8") as file:
        json.dump(
            summary,
            file,
            ensure_ascii=False,
            indent=2,
        )


def main():
    args = parse_args()

    # 只在真正运行检索时加载模型和向量库；标注检查无需密钥或联网。
    from utils.config_handler import chroma_conf

    cases = load_cases()
    k = int(chroma_conf["k"])
    if args.profile == "hybrid":
        from rag.hybrid_retriever import HybridRetriever

        retriever = HybridRetriever.from_knowledge_base()
    else:
        from rag.vector_store import VectorStoreService

        vector_store = VectorStoreService(profile_name=args.profile)
        retriever = vector_store.get_retriever()
    result_path = get_result_path(args.profile)

    # print('='*30)
    # print('cases:',cases)
    # print('='*30)

    started_at = perf_counter()
    results = [
        evaluate_case(retriever, case)
        for case in cases
    ]
    elapsed_seconds = perf_counter() - started_at

    # print('='*30)
    # print('results:',results)
    # print('='*30)

    answerable_results = [
        result
        for result in results
        if result["answerable"]
    ]

    passed_count = sum(
        result["passed"]
        for result in answerable_results
    )

    total_count = len(answerable_results)
    hit_rate = passed_count / total_count if total_count else 0

    summary = {
        "profile": args.profile,
        "retrieval_config": {
            "result_k": k,
            "chunk_size": int(chroma_conf["chunk_size"]),
            "chunk_overlap": int(chroma_conf["chunk_overlap"]),
            **(
                {
                    "candidate_k": int(chroma_conf["hybrid_candidate_k"]),
                    "rrf_constant": int(chroma_conf["rrf_constant"]),
                    "vector_weight": float(chroma_conf["vector_weight"]),
                    "bm25_weight": float(chroma_conf["bm25_weight"]),
                }
                if args.profile == "hybrid"
                else {}
            ),
        },
        "metric": f"KeywordHit@{k}",
        "scoring_version": "multi_evidence_v1",
        "k": k,
        "total_cases": len(results),
        "answerable_cases": total_count,
        "passed_cases": passed_count,
        "hit_rate": hit_rate,
        "elapsed_seconds": elapsed_seconds,
        "average_latency_seconds": (
            elapsed_seconds / len(results) if results else 0
        ),
        "results": results,
    }

    save_results(summary, result_path)

    print(f"评测配置：{args.profile}")
    print(f"评测题总数：{len(results)}")
    print(f"有答案题数：{total_count}")
    print(f"证据关键词规则命中数：{passed_count}")
    print(f"KeywordHit@{k}：{hit_rate:.2%}")
    print(f"总耗时：{elapsed_seconds:.2f} 秒")
    print(
        "单题平均耗时："
        f"{elapsed_seconds / len(results) if results else 0:.3f} 秒"
    )
    print()

    for result in results:
        if result["needs_manual_review"]:
            status = "需要人工检查"
        elif result["passed"]:
            status = (
                f"规则通过，证据排名第 {result['evidence_rank']}，"
                f"匹配第 {result['matched_evidence_group']} 组依据"
            )
        else:
            status = "失败"

        print(f"{result['id']} | {status}")

    print()
    print(f"详细结果已保存到：{result_path}")


if __name__ == "__main__":
    main()
