#!/usr/bin/env python3
"""
Model Versioning and Registry for Local LLMs.

Features:
- Model versioning
- Model metadata storage
- Model comparison
- Model deployment tracking
- Model lifecycle management

Usage:
    from deployment.registry import ModelRegistry

    registry = ModelRegistry()
    registry.register_model("phi3-mini-v1", "./phi3-mini-q4_k_m.gguf", metadata={})
"""

import argparse
import json
import os
import shutil
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional


@dataclass
class ModelVersion:
    """Model version information."""
    version: str
    model_path: str
    created_at: float
    metadata: dict = field(default_factory=dict)
    tags: list[str] = field(default_factory=list)
    status: str = "active"  # active, deprecated, archived
    size_bytes: int = 0
    checksum: str = ""


@dataclass
class ModelInfo:
    """Model information."""
    name: str
    versions: list[ModelVersion] = field(default_factory=list)
    description: str = ""
    created_at: float = 0
    updated_at: float = 0


class ModelRegistry:
    """
    Model versioning and registry.

    Features:
    - Register new model versions
    - List and search models
    - Compare model versions
    - Deploy/rollback models
    - Model metadata management
    """

    def __init__(self, registry_dir: str = "./model_registry"):
        """
        Initialize registry.

        Args:
            registry_dir: Registry directory
        """
        self.registry_dir = Path(registry_dir)
        self.registry_dir.mkdir(parents=True, exist_ok=True)
        self.registry_file = self.registry_dir / "registry.json"
        self.models: dict[str, ModelInfo] = {}
        self._load_registry()

    def _load_registry(self):
        """Load registry from file."""
        if self.registry_file.exists():
            with open(self.registry_file) as f:
                data = json.load(f)
                for name, info in data.items():
                    self.models[name] = ModelInfo(
                        name=name,
                        versions=[
                            ModelVersion(**v) for v in info.get("versions", [])
                        ],
                        description=info.get("description", ""),
                        created_at=info.get("created_at", 0),
                        updated_at=info.get("updated_at", 0),
                    )

    def _save_registry(self):
        """Save registry to file."""
        data = {}
        for name, info in self.models.items():
            data[name] = {
                "name": info.name,
                "versions": [
                    {
                        "version": v.version,
                        "model_path": v.model_path,
                        "created_at": v.created_at,
                        "metadata": v.metadata,
                        "tags": v.tags,
                        "status": v.status,
                        "size_bytes": v.size_bytes,
                        "checksum": v.checksum,
                    }
                    for v in info.versions
                ],
                "description": info.description,
                "created_at": info.created_at,
                "updated_at": info.updated_at,
            }

        with open(self.registry_file, "w") as f:
            json.dump(data, f, indent=2)

    def register_model(
        self,
        name: str,
        model_path: str,
        version: Optional[str] = None,
        description: str = "",
        metadata: Optional[dict] = None,
        tags: Optional[list[str]] = None,
    ) -> ModelVersion:
        """
        Register a new model version.

        Args:
            name: Model name
            model_path: Path to model file
            version: Version string (auto-generated if None)
            description: Model description
            metadata: Additional metadata
            tags: Model tags

        Returns:
            ModelVersion
        """
        # Auto-generate version if not provided
        if version is None:
            existing = self.models.get(name)
            if existing and existing.versions:
                last_version = existing.versions[-1].version
                parts = last_version.split(".")
                parts[-1] = str(int(parts[-1]) + 1)
                version = ".".join(parts)
            else:
                version = "1.0.0"

        # Calculate model size
        model_file = Path(model_path)
        size_bytes = model_file.stat().st_size if model_file.exists() else 0

        # Create version
        model_version = ModelVersion(
            version=version,
            model_path=str(model_file.absolute()),
            created_at=time.time(),
            metadata=metadata or {},
            tags=tags or [],
            status="active",
            size_bytes=size_bytes,
        )

        # Add to registry
        if name not in self.models:
            self.models[name] = ModelInfo(
                name=name,
                description=description,
                created_at=time.time(),
            )

        self.models[name].versions.append(model_version)
        self.models[name].updated_at = time.time()
        self.models[name].description = description or self.models[name].description

        self._save_registry()

        print(f"Registered model: {name} v{version}")
        print(f"  Path: {model_path}")
        print(f"  Size: {size_bytes / 1024 / 1024:.1f} MB")

        return model_version

    def get_model(self, name: str) -> Optional[ModelInfo]:
        """Get model information."""
        return self.models.get(name)

    def get_version(self, name: str, version: str) -> Optional[ModelVersion]:
        """Get specific model version."""
        model = self.models.get(name)
        if model:
            for v in model.versions:
                if v.version == version:
                    return v
        return None

    def get_latest_version(self, name: str) -> Optional[ModelVersion]:
        """Get latest model version."""
        model = self.models.get(name)
        if model and model.versions:
            return model.versions[-1]
        return None

    def list_models(self, status: Optional[str] = None) -> list[dict]:
        """
        List all models.

        Args:
            status: Filter by status

        Returns:
            List of model info dicts
        """
        result = []
        for name, info in self.models.items():
            versions = info.versions
            if status:
                versions = [v for v in versions if v.status == status]

            if versions:
                latest = versions[-1]
                result.append({
                    "name": name,
                    "description": info.description,
                    "latest_version": latest.version,
                    "total_versions": len(versions),
                    "status": latest.status,
                    "size_mb": latest.size_bytes / 1024 / 1024,
                    "created_at": info.created_at,
                    "updated_at": info.updated_at,
                })

        return sorted(result, key=lambda x: x["updated_at"], reverse=True)

    def search_models(self, query: str) -> list[dict]:
        """
        Search models by name, description, or tags.

        Args:
            query: Search query

        Returns:
            List of matching models
        """
        query_lower = query.lower()
        results = []

        for name, info in self.models.items():
            # Check name
            if query_lower in name.lower():
                results.append({"name": name, "match": "name"})
                continue

            # Check description
            if query_lower in info.description.lower():
                results.append({"name": name, "match": "description"})
                continue

            # Check tags
            for version in info.versions:
                if any(query_lower in tag.lower() for tag in version.tags):
                    results.append({"name": name, "match": "tag"})
                    break

        return results

    def deprecate_version(self, name: str, version: str):
        """Deprecate a model version."""
        model_version = self.get_version(name, version)
        if model_version:
            model_version.status = "deprecated"
            self._save_registry()
            print(f"Deprecated: {name} v{version}")

    def archive_version(self, name: str, version: str):
        """Archive a model version."""
        model_version = self.get_version(name, version)
        if model_version:
            model_version.status = "archived"
            self._save_registry()
            print(f"Archived: {name} v{version}")

    def delete_version(self, name: str, version: str):
        """Delete a model version."""
        model = self.models.get(name)
        if model:
            model.versions = [v for v in model.versions if v.version != version]
            if not model.versions:
                del self.models[name]
            self._save_registry()
            print(f"Deleted: {name} v{version}")

    def compare_versions(self, name: str, version1: str, version2: str) -> dict:
        """
        Compare two model versions.

        Args:
            name: Model name
            version1: First version
            version2: Second version

        Returns:
            Comparison dict
        """
        v1 = self.get_version(name, version1)
        v2 = self.get_version(name, version2)

        if not v1 or not v2:
            return {"error": "Version not found"}

        return {
            "model": name,
            "version1": {
                "version": v1.version,
                "size_mb": v1.size_bytes / 1024 / 1024,
                "created_at": v1.created_at,
                "status": v1.status,
                "metadata": v1.metadata,
            },
            "version2": {
                "version": v2.version,
                "size_mb": v2.size_bytes / 1024 / 1024,
                "created_at": v2.created_at,
                "status": v2.status,
                "metadata": v2.metadata,
            },
        }

    def get_model_history(self, name: str) -> list[dict]:
        """Get model version history."""
        model = self.models.get(name)
        if not model:
            return []

        return [
            {
                "version": v.version,
                "created_at": v.created_at,
                "status": v.status,
                "size_mb": v.size_bytes / 1024 / 1024,
                "tags": v.tags,
            }
            for v in model.versions
        ]

    def export_registry(self, format: str = "json") -> str:
        """
        Export registry.

        Args:
            format: Export format

        Returns:
            Exported data
        """
        if format == "json":
            return json.dumps(
                {
                    name: {
                        "description": info.description,
                        "versions": [
                            {
                                "version": v.version,
                                "status": v.status,
                                "created_at": v.created_at,
                            }
                            for v in info.versions
                        ],
                    }
                    for name, info in self.models.items()
                },
                indent=2,
            )
        else:
            raise ValueError(f"Unsupported format: {format}")


def main(argv=None):
    """Model versioning and registry operations from the command line."""
    parser = argparse.ArgumentParser(
        description="Model versioning and registry"
    )
    parser.add_argument("--registry-dir", default="./model_registry")
    sub = parser.add_subparsers(dest="command", required=True)

    p_register = sub.add_parser("register", help="Register a new model version")
    p_register.add_argument("--name", required=True)
    p_register.add_argument("--model-path", required=True)
    p_register.add_argument("--version")
    p_register.add_argument("--description", default="")
    p_register.add_argument("--tags", nargs="*", default=[])

    p_list = sub.add_parser("list", help="List registered models")
    p_list.add_argument("--status", help="Filter by status")

    p_search = sub.add_parser("search", help="Search models by query")
    p_search.add_argument("--query", required=True)

    p_show = sub.add_parser("show", help="Show latest version of a model")
    p_show.add_argument("--name", required=True)

    p_history = sub.add_parser("history", help="Show version history of a model")
    p_history.add_argument("--name", required=True)

    p_compare = sub.add_parser("compare", help="Compare two versions of a model")
    p_compare.add_argument("--name", required=True)
    p_compare.add_argument("--v1", required=True)
    p_compare.add_argument("--v2", required=True)

    p_deprecate = sub.add_parser("deprecate", help="Deprecate a model version")
    p_deprecate.add_argument("--name", required=True)
    p_deprecate.add_argument("--version", required=True)

    p_archive = sub.add_parser("archive", help="Archive a model version")
    p_archive.add_argument("--name", required=True)
    p_archive.add_argument("--version", required=True)

    p_delete = sub.add_parser("delete", help="Delete a model version")
    p_delete.add_argument("--name", required=True)
    p_delete.add_argument("--version", required=True)

    p_export = sub.add_parser("export", help="Export registry as JSON/CSV")
    p_export.add_argument("--format", default="json", choices=["json"])
    p_export.add_argument("--output", help="Write export to this path")

    args = parser.parse_args(argv)

    try:
        registry = ModelRegistry(registry_dir=args.registry_dir)

        if args.command == "register":
            version = registry.register_model(
                args.name,
                args.model_path,
                version=args.version,
                description=args.description,
                tags=args.tags,
            )
            print(json.dumps({
                "name": args.name,
                "version": version.version,
                "model_path": version.model_path,
                "status": version.status,
            }, indent=2, default=str))

        elif args.command == "list":
            print(json.dumps(registry.list_models(status=args.status), indent=2, default=str))

        elif args.command == "search":
            print(json.dumps(registry.search_models(args.query), indent=2, default=str))

        elif args.command == "show":
            latest = registry.get_latest_version(args.name)
            if not latest:
                print(f"Error: model '{args.name}' not found", file=sys.stderr)
                return 1
            print(json.dumps({
                "name": args.name,
                "version": latest.version,
                "model_path": latest.model_path,
                "status": latest.status,
                "size_mb": latest.size_bytes / 1024 / 1024,
                "tags": latest.tags,
            }, indent=2, default=str))

        elif args.command == "history":
            print(json.dumps(registry.get_model_history(args.name), indent=2, default=str))

        elif args.command == "compare":
            result = registry.compare_versions(args.name, args.v1, args.v2)
            print(json.dumps(result, indent=2, default=str))

        elif args.command in ("deprecate", "archive", "delete"):
            method = {
                "deprecate": registry.deprecate_version,
                "archive": registry.archive_version,
                "delete": registry.delete_version,
            }[args.command]
            method(args.name, args.version)

        elif args.command == "export":
            data = registry.export_registry(args.format)
            if args.output:
                Path(args.output).write_text(data)
                print(f"Exported registry to {args.output}")
            else:
                print(data)

        return 0
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    import sys
    sys.exit(main())
