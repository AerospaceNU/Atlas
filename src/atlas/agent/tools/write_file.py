from typing import Any

from pydantic import BaseModel, Field

from atlas.agent.artifacts import LocalArtifactStore
from atlas.agent.contracts import Tool, ToolResult


# pydantic model for the inputs to the tool
class WriteFileInput(BaseModel):
    path: str = Field(description="Path relative to the local artifact workspace.")
    content: str = Field(
        description="The content that should be written to the file path. Input should be a raw string, and it will be written to the file as UTF-8 text."
    )


"""Tool for writing content to a file in the local artifact workspace."""


class WriteTool(Tool):
    name = "write_file"
    description = "Write to a UTF-8 text file."
    trust = "default"
    max_bytes = 1_000_000  # max file size of 1MB

    @property
    def input_model(self) -> type[BaseModel]:
        return WriteFileInput

    def run(self, arguments: dict[str, Any], store: LocalArtifactStore) -> ToolResult:
        """Run the write file tool, which writes content to a specified file path in the local artifact workspace."""
        request = WriteFileInput.model_validate(arguments)  # validate the input
        path = store.resolve(request.path)

        content_size = len(request.content.encode("utf-8"))
        if content_size > self.max_bytes:
            raise ValueError(f"Content exceeds the {self.max_bytes} byte write limit")

        path.parent.mkdir(
            parents=True, exist_ok=True
        )  # create the parent directories if they don't exist
        path.write_text(request.content, encoding="utf-8")  # write the content to the file
        return ToolResult(
            text=f"File {request.path} created and written successfully (bytes: {content_size}).",
            artifacts=[
                store.relative(path)
            ],  # return the relative path of the written file as an artifact
        )
