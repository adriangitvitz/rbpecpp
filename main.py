from datasets import load_dataset

from vector_search import VectorSearchSystem


def load_leetcode_detailed_dataset():
    """Load the detailed LeetCode problem dataset"""
    dataset = load_dataset("kaysss/leetcode-problem-detailed", split="train")

    problems = []
    for item in dataset:
        tags = []
        if "topicTags" in item and item["topicTags"]:
            try:
                if isinstance(item["topicTags"], str):
                    import ast

                    parsed_tags = ast.literal_eval(item["topicTags"])
                    if isinstance(parsed_tags, list):
                        tags = [
                            tag.get("name", "")
                            for tag in parsed_tags
                            if isinstance(tag, dict)
                        ]
                    else:
                        tags = [item["topicTags"]]
                elif isinstance(item["topicTags"], list):
                    tags = [
                        tag.get("name", "")
                        for tag in item["topicTags"]
                        if isinstance(tag, dict)
                    ]
            except Exception as e:
                print(f"Error parsing topicTags: {e}")
                tags = [item["topicTags"]]

        problem = {
            "id": item["questionFrontendId"],
            "title": item["questionTitle"],
            "content": item["content"],
            "difficulty": item["difficulty"],
            "tags": tags,
            "slug": item["TitleSlug"],
            "url": f"https://leetcode.com/problems/{item['TitleSlug']}/description/",
        }
        problems.append(problem)

    return problems


def load_leetcode_basic_dataset():
    """Load the basic LeetCode problem dataset"""
    dataset = load_dataset("kaysss/leetcode-problem-set", split="train")

    problems = []
    for i, item in enumerate(dataset):
        tags = []
        if "topicTags" in item and item["topicTags"]:
            try:
                if isinstance(item["topicTags"], str):
                    if "," in item["topicTags"]:
                        tags = [tag.strip() for tag in item["topicTags"].split(",")]
                    else:
                        import json

                        try:
                            parsed_tags = json.loads(item["topicTags"])
                            if isinstance(parsed_tags, list):
                                tags = [
                                    tag.get("name", "")
                                    for tag in parsed_tags
                                    if isinstance(tag, dict)
                                ]
                            else:
                                tags = [item["topicTags"]]
                        except Exception:
                            tags = [item["topicTags"]]
                elif isinstance(item["topicTags"], list):
                    if all(isinstance(tag, dict) for tag in item["topicTags"]):
                        tags = [tag.get("name", "") for tag in item["topicTags"]]
                    else:
                        tags = item["topicTags"]
            except Exception as e:
                print(f"Error parsing topicTags: {e}")
                tags = []

        problem = {
            "id": item["frontendQuestionId"],
            "title": item["title"],
            "difficulty": item["difficulty"],
            "tags": tags,
            "url": f"https://leetcode.com/problems/{item['titleSlug']}/description/",
        }
        problems.append(problem)

    return problems


if __name__ == "__main__":
    tech_terms = [
        "time complexity",
        "space complexity",
        "dynamic programming",
        "binary search",
        "depth-first search",
        "breadth-first search",
        "backtracking",
        "greedy algorithm",
        "two pointers",
        "sliding window",
        "O(n log n)",
        "O(n²)",
        "O(2^n)",
        "linked list",
        "binary tree",
        "hash table",
    ]

    search_system = VectorSearchSystem(tech_terms=tech_terms)

    try:
        print("Loading detailed LeetCode dataset...")
        problems = load_leetcode_detailed_dataset()
    except Exception as e:
        print(f"Could not load detailed dataset: {e}")
        print("Falling back to basic dataset...")
        problems = load_leetcode_basic_dataset()

    print(f"Loaded {len(problems)} LeetCode problems")

    full_text = " ".join([f"{p['title']} {p.get('content', '')}" for p in problems])

    print("Training tokenizer...")
    search_system.train_tokenizer(full_text)

    print("Creating embeddings...")
    search_system.create_embeddings(problems)

    query = "Given an array of integers nums and an integer target"
    print(f"\nQuerying: {query}")
    results = search_system.query(query)

    print("\nTop matching problems:")
    for i, result in enumerate(results):
        tags = result.get("tags", [])
        tags_str = ", ".join(tags) if tags else "No tags"
        print(f"{i + 1}. {result['title']} ({result['difficulty']}) - {tags_str}")
        print(f"   URL: {result['url']}")

    sample_problem = """
    Given an array of integers nums and an integer target, return indices of the two numbers such that they add up to target.
    You may assume that each input would have exactly one solution, and you may not use the same element twice.
    You can return the answer in any order.
    """

    print("\nTesting tokenizer functions:")
    encoded = search_system.tokenizer.encode(sample_problem)
    encoded_with_dropout = search_system.tokenizer.encode_with_dropout(
        sample_problem, dropout_prob=0.1
    )
    decoded = search_system.tokenizer.decode(encoded)

    print(f"Original: {sample_problem[:100]}...")
    print(f"Encoded tokens: {len(encoded)} tokens")
    print(f"Encoded with dropout: {len(encoded_with_dropout)} tokens")
    print(f"Decoded: {decoded[:100]}...")

    chunks = search_system.tokenizer.chunk_with_overlap(
        sample_problem, chunk_size=100, overlap=20
    )
    print(f"\nChunked into {len(chunks)} overlapping segments")
    for i, chunk in enumerate(chunks):
        print(f"Chunk {i + 1}: {len(chunk)} tokens")
