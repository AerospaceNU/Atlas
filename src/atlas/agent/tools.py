from __future__ import annotations

import ast
import math
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any

from PIL import Image, ImageDraw, ImageOps
from pydantic import BaseModel, Field

from atlas.agent.artifacts import ImageInfo, LocalArtifactStore
from atlas.agent.contracts import ToolDefinition


class ToolResult(BaseModel):
    """Structured tool output returned to the model and retained in the run log."""

    text: str
    artifacts: list[str] = Field(default_factory=list)


class Tool(ABC):
    """A local capability that can be safely advertised to an agent."""

    name: str
    description: str

    @property
    @abstractmethod
    def input_model(self) -> type[BaseModel]:
        """Pydantic input model used for both validation and JSON schema."""
        raise NotImplementedError

    def definition(self) -> ToolDefinition:
        return ToolDefinition(
            name=self.name,
            description=self.description,
            parameters=self.input_model.model_json_schema(),
        )

    @abstractmethod
    def run(self, arguments: dict[str, Any], store: LocalArtifactStore) -> ToolResult:
        """Validate and execute a tool call within ``store``."""
        raise NotImplementedError


class ToolRegistry:
    """Explicit allow-list of tools available to an agent session."""

    def __init__(self) -> None:
        self._tools: dict[str, Tool] = {}

    def register(self, tool: Tool) -> None:
        if tool.name in self._tools:
            raise ValueError(f"Tool already registered: {tool.name}")
        self._tools[tool.name] = tool

    @property
    def definitions(self) -> list[ToolDefinition]:
        return [tool.definition() for tool in self._tools.values()]

    def execute(
        self, name: str, arguments: dict[str, Any], store: LocalArtifactStore
    ) -> ToolResult:
        try:
            tool = self._tools[name]
        except KeyError as exc:
            raise KeyError(f"Unknown tool: {name}") from exc
        return tool.run(arguments, store)


class ListImagesInput(BaseModel):
    pass


class ListImagesTool(Tool):
    name = "list_images"
    description = "List the images available in the local artifact workspace."

    @property
    def input_model(self) -> type[BaseModel]:
        return ListImagesInput

    def run(self, arguments: dict[str, Any], store: LocalArtifactStore) -> ToolResult:
        ListImagesInput.model_validate(arguments)
        images = store.list_images()
        if not images:
            return ToolResult(text="No images are available in the local artifact workspace.")
        summary = "\n".join(_image_line(image) for image in images)
        return ToolResult(text=f"Found {len(images)} image(s):\n{summary}")


class InspectImageInput(BaseModel):
    path: str = Field(description="Path relative to the local artifact workspace")


class InspectImageTool(Tool):
    name = "inspect_image"
    description = "Read dimensions, mode, and format of one local image."

    @property
    def input_model(self) -> type[BaseModel]:
        return InspectImageInput

    def run(self, arguments: dict[str, Any], store: LocalArtifactStore) -> ToolResult:
        request = InspectImageInput.model_validate(arguments)
        image = store.inspect(request.path)
        return ToolResult(text=_image_line(image))


class ContactSheetInput(BaseModel):
    paths: list[str] = Field(
        default_factory=list,
        description="Optional local image paths. Omit to include every image in the workspace.",
    )
    output_path: str = Field(
        default="artifacts/contact-sheet.png",
        description="PNG path, relative to the local artifact workspace.",
    )
    columns: int = Field(default=3, ge=1, le=10)
    thumbnail_size: int = Field(default=256, ge=32, le=2048)


class ContactSheetTool(Tool):
    name = "create_contact_sheet"
    description = "Create a labeled PNG contact sheet from local workspace images."

    @property
    def input_model(self) -> type[BaseModel]:
        return ContactSheetInput

    def run(self, arguments: dict[str, Any], store: LocalArtifactStore) -> ToolResult:
        request = ContactSheetInput.model_validate(arguments)
        paths = request.paths or [image.path for image in store.list_images()]
        if not paths:
            raise ValueError("Cannot create a contact sheet without local images")
        output = store.resolve(request.output_path)
        if output.suffix.lower() != ".png":
            raise ValueError("Contact-sheet output_path must end in .png")
        if store.relative(output) in paths:
            raise ValueError("Contact-sheet output cannot overwrite an input image")

        thumbnails: list[tuple[str, Image.Image]] = []
        for relative_path in paths:
            source = store.resolve(relative_path)
            with Image.open(source) as image:
                thumbnail = ImageOps.contain(image.convert("RGB"), (request.thumbnail_size,) * 2)
                thumbnails.append((relative_path, thumbnail))

        label_height = 28
        padding = 12
        cell_size = request.thumbnail_size + (padding * 2)
        rows = math.ceil(len(thumbnails) / request.columns)
        sheet = Image.new(
            "RGB",
            (request.columns * cell_size, rows * (cell_size + label_height)),
            color="white",
        )
        draw = ImageDraw.Draw(sheet)
        for index, (relative_path, thumbnail) in enumerate(thumbnails):
            col = index % request.columns
            row = index // request.columns
            origin_x = col * cell_size
            origin_y = row * (cell_size + label_height)
            x = origin_x + (cell_size - thumbnail.width) // 2
            y = origin_y + (cell_size - thumbnail.height) // 2
            sheet.paste(thumbnail, (x, y))
            draw.text((origin_x + padding, origin_y + cell_size), relative_path, fill="black")

        output.parent.mkdir(parents=True, exist_ok=True)
        sheet.save(output, format="PNG")
        relative_output = store.relative(output)
        return ToolResult(
            text=f"Created a {len(thumbnails)}-image contact sheet at {relative_output}.",
            artifacts=[relative_output],
        )


class StageScriptInput(BaseModel):
    name: str = Field(pattern=r"^[a-z][a-z0-9_]{2,63}$")
    source: str = Field(
        min_length=1, description="Python source saved for later review; never executed."
    )
    description: str = Field(min_length=1)


class StageScriptTool(Tool):
    """Allow the agent to propose a temporary script without granting execution."""

    name = "stage_script_proposal"
    description = (
        "Save a proposed Python helper script for human review. The script is not executed and is "
        "not automatically added to the trusted tool registry."
    )

    @property
    def input_model(self) -> type[BaseModel]:
        return StageScriptInput

    def run(self, arguments: dict[str, Any], store: LocalArtifactStore) -> ToolResult:
        request = StageScriptInput.model_validate(arguments)
        filename = f"proposals/{request.name}.py"
        path = store.resolve(filename)
        if path.exists():
            raise ValueError(f"A proposal already exists for {request.name}")
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(request.source, encoding="utf-8")
        manifest = store.resolve(f"proposals/{request.name}.md")
        manifest.write_text(f"# {request.name}\n\n{request.description}\n", encoding="utf-8")
        return ToolResult(
            text=(
                f"Staged proposal {request.name} at {filename}. It is not executable or registered; "
                "review and register it explicitly in application code."
            ),
            artifacts=[filename, store.relative(manifest)],
        )


class ReviewScriptProposalsInput(BaseModel):
    output_path: str = Field(
        default="proposals/review.json",
        description="JSON report path, relative to the local artifact workspace.",
    )


class ProposalReview(BaseModel):
    name: str
    syntax_valid: bool
    findings: list[str] = Field(default_factory=list)


class ProposalReviewReport(BaseModel):
    """Static assessment only; a clean report is not an approval to execute."""

    proposals: list[ProposalReview] = Field(default_factory=list)


class ReviewScriptProposalsTool(Tool):
    """Produce a deterministic, local review record for staged agent scripts."""

    name = "review_script_proposals"
    description = (
        "Statically review staged Python script proposals and write a local JSON report. "
        "This does not execute, approve, or register any proposal."
    )

    @property
    def input_model(self) -> type[BaseModel]:
        return ReviewScriptProposalsInput

    def run(self, arguments: dict[str, Any], store: LocalArtifactStore) -> ToolResult:
        request = ReviewScriptProposalsInput.model_validate(arguments)
        output = store.resolve(request.output_path)
        if output.suffix.lower() != ".json":
            raise ValueError("Proposal-review output_path must end in .json")

        proposal_dir = store.resolve("proposals")
        reports: list[ProposalReview] = []
        if proposal_dir.is_dir():
            for path in sorted(proposal_dir.glob("*.py")):
                reports.append(_review_proposal(path))
        report = ProposalReviewReport(proposals=reports)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(report.model_dump_json(indent=2), encoding="utf-8")
        relative_output = store.relative(output)
        return ToolResult(
            text=(
                f"Reviewed {len(reports)} staged proposal(s) at {relative_output}. "
                "The report is static analysis only; no proposal was executed or registered."
            ),
            artifacts=[relative_output],
        )


def default_registry(*, allow_script_proposals: bool = False) -> ToolRegistry:
    """Build the normal local-image tool allow-list for one agent session."""
    registry = ToolRegistry()
    registry.register(ListImagesTool())
    registry.register(InspectImageTool())
    registry.register(ContactSheetTool())
    if allow_script_proposals:
        registry.register(StageScriptTool())
        registry.register(ReviewScriptProposalsTool())
    from atlas.models.registry import register_model_tools

    register_model_tools(registry)
    return registry


def _image_line(image: ImageInfo) -> str:
    format_name = image.format or "unknown"
    return f"- {image.path}: {image.width}x{image.height}, {image.mode}, {format_name}"


def _review_proposal(path: Path) -> ProposalReview:
    source = path.read_text(encoding="utf-8")
    findings: list[str] = []
    try:
        tree = ast.parse(source, filename=path.name)
    except SyntaxError as exc:
        return ProposalReview(
            name=path.stem,
            syntax_valid=False,
            findings=[f"SyntaxError at line {exc.lineno}: {exc.msg}"],
        )

    risky_imports = {"httpx", "os", "requests", "socket", "subprocess"}
    risky_calls = {"__import__", "compile", "eval", "exec", "open"}
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                if alias.name.split(".")[0] in risky_imports:
                    findings.append(f"Imports {alias.name}; review its host or network effects.")
        elif isinstance(node, ast.ImportFrom) and node.module is not None:
            if node.module.split(".")[0] in risky_imports:
                findings.append(f"Imports from {node.module}; review its host or network effects.")
        elif (
            isinstance(node, ast.Call)
            and isinstance(node.func, ast.Name)
            and node.func.id in risky_calls
        ):
            findings.append(f"Calls {node.func.id}(); review before any promotion.")
    return ProposalReview(name=path.stem, syntax_valid=True, findings=sorted(set(findings)))
