from datetime import datetime, timezone
from typing import Generic, Literal, Optional, TypeAlias, TypeVar

from neo4j_extension import Graph, Neo4jConnection, Node, PythonType, Relationship

#
# (A) 위에서 제시한 #1 코드 (my_app.py) 의 내용 + 일부 확장
#
from pydantic import BaseModel

K = TypeVar("K", bound=str, covariant=True)


class Knowledge(BaseModel, Generic[K]):
    """
    메모리 요소의 최상위 클래스.
    - type: 지식의 종류
    """

    type: K


KT = TypeVar("KT", bound=Knowledge[str])


class KnowledgeCollection(BaseModel, Generic[KT]):
    """
    여러 개의 Knowledge를 보관하는 컨테이너
    """

    value: list[KT]


# 1) Semantic Memory
class Triplet(Knowledge[Literal["Triplet"]]):
    subject: str
    predicate: str
    object: str


class FileBlob(Knowledge[Literal["FileBlob"]]):
    filepath: str


class StringText(Knowledge[Literal["StringText"]]):
    text: str


SemanticKnowledgeType: TypeAlias = Triplet | FileBlob | StringText


class SemanticKnowledge(KnowledgeCollection[SemanticKnowledgeType]):
    pass


# 2) Episodic Memory
class ConversationSummary(Knowledge[Literal["ConversationSummary"]]):
    summary: str
    timestamp: Optional[datetime] = None


class FewShotExample(Knowledge[Literal["FewShotExample"]]):
    prompt: str
    response: str


EpisodicKnowledgeType: TypeAlias = ConversationSummary | FewShotExample


class EpisodicKnowledge(KnowledgeCollection[EpisodicKnowledgeType]):
    pass


# 3) Procedural Memory
class PersonalityTrait(Knowledge[Literal["PersonalityTrait"]]):
    trait: str
    description: Optional[str] = None


class ResponsePattern(Knowledge[Literal["ResponsePattern"]]):
    pattern: str
    example_responses: list[str]


class PromptRule(Knowledge[Literal["PromptRule"]]):
    rule: str


ProceduralKnowledgeType: TypeAlias = PersonalityTrait | ResponsePattern | PromptRule


class ProceduralKnowledge(KnowledgeCollection[ProceduralKnowledgeType]):
    pass


# (C) Knowledge -> Node 변환 함수
def knowledge_to_node(knowledge: Knowledge[str], global_id: str) -> Node:
    """
    단일 Knowledge를 Node로 변환.
    - Node의 label은 knowledge.type 사용
    - Node의 property에는 type을 제외한 나머지 필드를 모두 할당
    - globalId는 별도로 주어진 값 사용
    """
    label = knowledge.type  # 예: "Triplet", "ConversationSummary"
    props: dict[str, PythonType] = {}
    for field_name, field_value in knowledge.model_dump().items():
        if field_name == "type":
            continue
        props[field_name] = field_value

    # Node 생성
    node = Node(
        properties=props,
        labels={label},  # 라벨로 type을 설정
        globalId=global_id,  # 고유 ID 부여
    )
    return node


# (D) 1단계: 명시적 지식(모든 Knowledge)에 대해 unique ID를 할당하여 추출
def extract_knowledge(
    semantic: SemanticKnowledge,
    episodic: EpisodicKnowledge,
    procedural: ProceduralKnowledge,
) -> dict[str, Knowledge[str]]:
    """
    세 가지 KnowledgeCollection에서 Knowledge들을 꺼내어,
    각 지식마다 고유 ID를 부여한 뒤,
    { 고유ID: Knowledge객체 } 형태의 매핑을 구성한다.
    """
    knowledge_map: dict[str, Knowledge[str]] = {}
    index = 1

    # Semantic
    for item in semantic.value:
        k_id = f"K{index}"
        knowledge_map[k_id] = item
        index += 1

    # Episodic
    for item in episodic.value:
        k_id = f"K{index}"
        knowledge_map[k_id] = item
        index += 1

    # Procedural
    for item in procedural.value:
        k_id = f"K{index}"
        knowledge_map[k_id] = item
        index += 1

    return knowledge_map


# (E) 2단계: Dependency 규칙을 별도 함수로 분리하여 추출
def extract_dependencies(knowledge_map: dict[str, Knowledge[str]]) -> list[tuple[str, str, str]]:
    """
    전달받은 knowledge_map(고유ID -> Knowledge)에 대해,
    의존성(Dependency) 관계를 도출하여
    (start_id, end_id, rel_type) 형태의 튜플 리스트로 반환한다.

    예시 규칙:
      - ConversationSummary -> (DEPEND_ON) -> StringText
      - FewShotExample -> (DEPEND_ON) -> Triplet
    """
    dependencies: list[tuple[str, str, str]] = []

    # 간단한 예시: Episodic → Semantic 관계
    for k_id, knowledge in knowledge_map.items():
        if isinstance(knowledge, ConversationSummary):
            # 해당 Knowledge가 ConversationSummary인 경우,
            # 모든 StringText 지식을 찾아서 (StringText -> ConversationSummary) 관계를 만든다
            for other_id, other_knowledge in knowledge_map.items():
                if isinstance(other_knowledge, StringText):
                    dependencies.append((other_id, k_id, "DEPEND_ON"))

        elif isinstance(knowledge, FewShotExample):
            # 해당 Knowledge가 FewShotExample인 경우,
            # 모든 Triplet 지식을 찾아서 (Triplet -> FewShotExample) 관계를 만든다
            for other_id, other_knowledge in knowledge_map.items():
                if isinstance(other_knowledge, Triplet):
                    dependencies.append((other_id, k_id, "DEPEND_ON"))

    # ProceduralKnowledge에 대해서는 별도 규칙이 없다고 가정
    return dependencies


# (F) 3단계: 위에서 추출된 knowledge_map과 dependencies를 통해 최종 그래프 구성
def build_dependency_graph(
    knowledge_map: dict[str, Knowledge[str]],
    dependencies: list[tuple[str, str, str]],
) -> Graph:
    """
    knowledge_map과 dependencies( (start_id, end_id, rel_type) )를 사용해
    최종적인 Directed Graph를 구성한다.
    """
    graph = Graph()

    # 1) Node 생성
    node_map: dict[str, Node] = {}
    for k_id, knowledge_obj in knowledge_map.items():
        node = knowledge_to_node(knowledge_obj, global_id=k_id)
        graph.add_node(node)
        node_map[k_id] = node

    # 2) Relationship 생성
    for start_id, end_id, rel_type in dependencies:
        start_node = node_map[start_id]
        end_node = node_map[end_id]
        rel = Relationship(
            properties={},
            rel_type=rel_type,
            start_node=start_node,
            end_node=end_node,
        )
        graph.add_relationship(rel)

    return graph


# --------------------------------------------------------------------------
# (H) [신규] 비정형 문서로부터 Knowledge & Dependency를 추출하는 예시 함수들
# --------------------------------------------------------------------------
def extract_knowledge_from_unstructured_text(text: str) -> list[Knowledge[str]]:
    """
    비정형 텍스트에서 Triplet/StringText 등의 Knowledge를 추출하는 예시 함수.
    실제론 NLP/LLM 등을 사용해 파싱해야 함.
    여기서는 '주어 is 목적어' 형태를 찾는 아주 단순한 예시로 대체.
    """
    # 예: "Python is a powerful language. This text is an example."
    #     -> Triplet(type="Triplet", subject="Python", predicate="is", object="a powerful language")
    #     -> Triplet(type="Triplet", subject="This text", predicate="is", object="an example")
    # 실제론 훨씬 복잡한 전처리/토크나이징/NER/RE/LLM 호출 등이 필요함.

    knowledge_list: list[Knowledge[str]] = []
    sentences = text.split(".")  # 아주 단순화

    for sentence in sentences:
        sentence = sentence.strip()
        if not sentence:
            continue
        # "Python is a powerful language" 형태만 찾는다고 가정
        parts = sentence.split(" is ")
        if len(parts) == 2:
            subject = parts[0].strip()
            obj = parts[1].strip()
            triplet = Triplet(
                type="Triplet",
                subject=subject,
                predicate="is",
                object=obj,
            )
            knowledge_list.append(triplet)
        else:
            # 그 외 문장은 그냥 StringText 로 처리
            knowledge_list.append(StringText(type="StringText", text=sentence))

    return knowledge_list


def extract_dependencies_from_unstructured_text(knowledge_map: dict[str, Knowledge[str]]) -> list[tuple[str, str, str]]:
    """
    문서에서 새로 추출된 Knowledge들을 대상으로 추가 Dependency를 도출.
    여기서는 간단히 Triplet이 존재하면 그에 대응되는 StringText가
    DEPEND_ON 관계를 갖는다고 가정(예시).
    """
    deps: list[tuple[str, str, str]] = []
    # 임의 규칙: Triplet -> (DEPEND_ON) -> StringText
    for k_id, knowledge in knowledge_map.items():
        if isinstance(knowledge, Triplet):
            # Triplet과 동일한 문장에서 나온 StringText를 찾아 의존성 부여
            # (단순히 "문자열 길이 비교" 같은 임의 논리로 시연)
            for other_id, other_knowledge in knowledge_map.items():
                if isinstance(other_knowledge, StringText):
                    if len(other_knowledge.text) < len(knowledge.object):
                        # 임의의 조건으로 Triplet이 StringText에 의존한다고 처리
                        deps.append((k_id, other_id, "DEPEND_ON"))

    return deps


def integrate_document_knowledge(
    docs: list[str],
    existing_knowledge_map: dict[str, Knowledge[str]],
    existing_dependencies: list[tuple[str, str, str]],
    start_index: int = 1000,
) -> tuple[dict[str, Knowledge[str]], list[tuple[str, str, str]]]:
    """
    다수의 비정형 문서를 순회하며 Knowledge, Dependency를 추출하고,
    기존 knowledge_map, dependency 리스트에 통합한다.
    - start_index: 새로 부여할 Knowledge ID 시작 번호
    """
    knowledge_map = dict(existing_knowledge_map)
    dependencies = list(existing_dependencies)
    current_index = start_index

    for doc_text in docs:
        # 1) 문서에서 Knowledge 추출
        extracted_list = extract_knowledge_from_unstructured_text(doc_text)

        # 2) 고유 ID 부여 후 knowledge_map에 추가
        doc_ids: list[str] = []
        for item in extracted_list:
            k_id = f"K{current_index}"
            knowledge_map[k_id] = item
            doc_ids.append(k_id)
            current_index += 1

        # 3) 문서 내에서의 Dependency 추출
        #    → extract_dependencies_from_unstructured_text() 같은 함수를 사용
        #    단, 함수가 knowledge_map 전체를 보고 판단하기 때문에, 새로 추가된 것 포함
        doc_deps = extract_dependencies_from_unstructured_text(knowledge_map)
        dependencies.extend(doc_deps)

    return knowledge_map, dependencies


# --------------------------------------------------------------------------
# (I) 최종 실행 예시
# --------------------------------------------------------------------------
if __name__ == "__main__":
    # 1) 예시 KnowledgeCollection 준비
    semantic = SemanticKnowledge(
        value=[
            Triplet(type="Triplet", subject="Python", predicate="is", object="programming_language"),
            FileBlob(type="FileBlob", filepath="/path/to/some_file.pdf"),
            StringText(type="StringText", text="This is some reference text."),
        ]
    )

    episodic = EpisodicKnowledge(
        value=[
            ConversationSummary(
                type="ConversationSummary",
                summary="User asked about Python decorators.",
                timestamp=datetime.now(timezone.utc),
            ),
            FewShotExample(
                type="FewShotExample",
                prompt="Explain Python decorators with an example.",
                response="A decorator is a function that wraps another function...",
            ),
        ]
    )

    procedural = ProceduralKnowledge(
        value=[
            PersonalityTrait(type="PersonalityTrait", trait="friendly", description="warm manner"),
            ResponsePattern(type="ResponsePattern", pattern="system_instructions", example_responses=[]),
            PromptRule(type="PromptRule", rule="Always respond politely."),
        ]
    )

    # 2) 기존의 Knowledge / Dependencies 추출
    knowledge_map = extract_knowledge(semantic, episodic, procedural)
    dependencies = extract_dependencies(knowledge_map)

    # 3) 비정형 문서 예시
    unstructured_docs = [
        "Python is a versatile language. This text is from a user manual.",
        "GPT is an advanced model. It is developed by OpenAI.",
    ]

    # 4) 문서에서 Knowledge, Dependency를 추가 추출하여 통합
    knowledge_map, dependencies = integrate_document_knowledge(
        unstructured_docs,
        existing_knowledge_map=knowledge_map,
        existing_dependencies=dependencies,
        start_index=1000,  # 기존 K1, K2... 다음부터는 K1000 등으로 부여
    )

    # 5) 최종 그래프 구성
    graph = build_dependency_graph(knowledge_map, dependencies)

    # 결과 확인
    print("=== In-Memory Graph ===")
    print(graph)

    print("\n=== Generated Cypher Query ===")
    print(graph.to_cypher())

    print("\n=== Graph Schema ===")
    print(graph.get_formatted_graph_schema())

    # Neo4j에 업서트
    with Neo4jConnection() as conn:
        conn.clear_all()
        conn.upsert_graph(graph)

    print("\nAll done!")
