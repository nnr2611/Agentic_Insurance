from typing import List, TypedDict

class ClaimGraphState(TypedDict):
    """
    Represents the state of our claim approval prediction graph.

    Attributes:
        prediction: predicted approved benefit amount.
        general_fields: the structured input fields about the claim.
        extra_documents: any extra docs retrieved for more context.
        has_extra_docs: whether retrieval was successful and useful.
    """
    prediction: float
    general_fields: dict

    question: str
    original_docs: str
    chunk_docs: str
    doc_path: str
    has_extra_docs: bool