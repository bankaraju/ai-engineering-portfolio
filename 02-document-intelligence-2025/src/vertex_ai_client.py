"""Stand-in for the optional Vertex AI embedding client.

The original client (a thin wrapper around a Vertex AI endpoint serving BGE)
is not included. `multi_index_retrieval.py` only imports it when run with
`--use-vertex`; the default path uses a local HuggingFace BGE model.
"""


class VertexBGEClient:
    def __init__(self, endpoint_id: str):
        self.endpoint_id = endpoint_id

    def __getattr__(self, name):
        raise NotImplementedError(
            "VertexBGEClient is not included in this archive; run without --use-vertex"
        )
