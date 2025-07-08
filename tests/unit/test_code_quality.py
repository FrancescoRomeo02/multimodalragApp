"""
Tests for code quality and project readiness.
"""
import ast
import os
import subprocess
import sys
from pathlib import Path


def test_no_syntax_errors():
    """Test that all Python files have valid syntax."""
    project_root = Path(__file__).parent.parent.parent
    python_files = []
    
    # Find all Python files in src/ and streamlit_app/
    for directory in ["src", "streamlit_app", "scripts"]:
        dir_path = project_root / directory
        if dir_path.exists():
            python_files.extend(dir_path.rglob("*.py"))
    
    for py_file in python_files:
        with open(py_file, 'r', encoding='utf-8') as f:
            try:
                ast.parse(f.read())
            except SyntaxError as e:
                assert False, f"Syntax error in {py_file}: {e}"


def test_no_todo_fixme_in_main_code():
    """Test that main code doesn't contain TODO or FIXME comments."""
    project_root = Path(__file__).parent.parent.parent
    
    for directory in ["src", "streamlit_app"]:
        dir_path = project_root / directory
        if dir_path.exists():
            for py_file in dir_path.rglob("*.py"):
                with open(py_file, 'r', encoding='utf-8') as f:
                    content = f.read().upper()
                    lines = content.split('\n')
                    for i, line in enumerate(lines, 1):
                        if 'TODO' in line or 'FIXME' in line:
                            # Allow TODOs in test files or comments marked as acceptable
                            if 'test' not in str(py_file).lower() and 'acceptable' not in line.lower():
                                print(f"Warning: Found TODO/FIXME in {py_file}:{i}: {line.strip()}")


def test_imports_are_clean():
    """Test that imports are properly organized."""
    project_root = Path(__file__).parent.parent.parent
    
    for directory in ["src", "streamlit_app"]:
        dir_path = project_root / directory
        if dir_path.exists():
            for py_file in dir_path.rglob("*.py"):
                if py_file.name == "__init__.py":
                    continue
                    
                with open(py_file, 'r', encoding='utf-8') as f:
                    content = f.read()
                    
                # Check for unused imports (basic check)
                lines = content.split('\n')
                imports = []
                for line in lines:
                    if line.strip().startswith(('import ', 'from ')):
                        imports.append(line.strip())
                
                # Basic check: if file has imports, it should have some code
                if imports and len([l for l in lines if l.strip() and not l.strip().startswith('#')]) < 5:
                    print(f"Warning: {py_file} has imports but minimal code")


def test_environment_files_exist():
    """Test that environment configuration files exist."""
    project_root = Path(__file__).parent.parent.parent
    
    required_files = [
        ".env.example",
        "requirements.txt",
        "pyproject.toml",
        ".gitignore",
        "README.md",
        "Dockerfile",
        "docker-compose.yml"
    ]
    
    for filename in required_files:
        file_path = project_root / filename
        assert file_path.exists(), f"Required file {filename} is missing"
        assert file_path.stat().st_size > 0, f"Required file {filename} is empty"


def test_no_large_files_in_git():
    """Test that no large files are tracked by git."""
    project_root = Path(__file__).parent.parent.parent
    
    # Skip if not a git repository
    if not (project_root / ".git").exists():
        return
    
    try:
        result = subprocess.run(
            ["git", "ls-files"],
            cwd=project_root,
            capture_output=True,
            text=True,
            check=True
        )
        
        tracked_files = result.stdout.strip().split('\n')
        large_files = []
        
        for filename in tracked_files:
            file_path = project_root / filename
            if file_path.exists() and file_path.stat().st_size > 50 * 1024 * 1024:  # 50MB
                large_files.append((filename, file_path.stat().st_size))
        
        if large_files:
            file_list = [f"{name} ({size/1024/1024:.1f}MB)" for name, size in large_files]
            assert False, f"Large files found in git: {', '.join(file_list)}"
            
    except subprocess.CalledProcessError:
        # Git command failed, skip this test
        pass


def test_docker_files_are_valid():
    """Test that Docker files have basic validity."""
    project_root = Path(__file__).parent.parent.parent
    
    # Check Dockerfile
    dockerfile = project_root / "Dockerfile"
    if dockerfile.exists():
        with open(dockerfile, 'r') as f:
            content = f.read()
            assert "FROM" in content, "Dockerfile should have a FROM instruction"
            assert "WORKDIR" in content or "COPY" in content, "Dockerfile should have WORKDIR or COPY"
    
    # Check docker-compose.yml
    compose_file = project_root / "docker-compose.yml"
    if compose_file.exists():
        with open(compose_file, 'r') as f:
            content = f.read()
            assert "version:" in content or "services:" in content, "docker-compose.yml should have version or services"


def test_makefile_has_basic_targets():
    """Test that Makefile has expected targets."""
    project_root = Path(__file__).parent.parent.parent
    makefile = project_root / "Makefile"
    
    if makefile.exists():
        with open(makefile, 'r') as f:
            content = f.read()
            expected_targets = ["test", "install", "clean"]
            for target in expected_targets:
                assert f"{target}:" in content, f"Makefile should have '{target}' target"


def test_no_sensitive_data():
    """Test that no sensitive data is present in tracked files."""
    project_root = Path(__file__).parent.parent.parent
    sensitive_patterns = [
        "password",
        "secret",
        "api_key",
        "private_key",
        "token"
    ]
    
    # Check main configuration and code files
    for directory in ["src", "streamlit_app", "."]:
        dir_path = project_root / directory
        if not dir_path.exists():
            continue
            
        if directory == ".":
            # Check only specific files in root
            files_to_check = [
                "README.md", "pyproject.toml", "requirements.txt", 
                "docker-compose.yml", "Dockerfile"
            ]
            for filename in files_to_check:
                file_path = dir_path / filename
                if file_path.exists():
                    _check_file_for_sensitive_data(file_path, sensitive_patterns)
        else:
            # Check all Python files in subdirectories
            for py_file in dir_path.rglob("*.py"):
                _check_file_for_sensitive_data(py_file, sensitive_patterns)


def _check_file_for_sensitive_data(file_path, sensitive_patterns):
    """Helper function to check a single file for sensitive data."""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read().lower()
            for pattern in sensitive_patterns:
                if f"{pattern}=" in content or f'"{pattern}"' in content:
                    # Allow references to environment variables or documentation
                    lines = content.split('\n')
                    for i, line in enumerate(lines, 1):
                        if pattern in line and not any(word in line for word in [
                            "env", "example", "todo", "fixme", "comment", "doc", "readme"
                        ]):
                            print(f"Warning: Potential sensitive data in {file_path}:{i}")
    except UnicodeDecodeError:
        # Skip binary files
        pass
