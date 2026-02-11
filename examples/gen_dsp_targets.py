"""Generate, build, and validate gen-dsp projects for each example.

Usage:
    python examples/gen_dsp_targets.py              # generate adapter files only
    python examples/gen_dsp_targets.py --build      # generate full projects and build
    python examples/gen_dsp_targets.py --validate   # build and validate binaries

Requires gen_dsp to be installed.
"""

from __future__ import annotations

import importlib
import json
import os
import platform as plat_mod
import subprocess
import sys
from pathlib import Path

from dsp_graph.compile import compile_graph
from dsp_graph.gen_dsp_adapter import (
    compile_for_gen_dsp,
    generate_adapter_cpp,
    generate_manifest,
)

IS_MACOS = plat_mod.system() == "Darwin"
IS_LINUX = plat_mod.system() == "Linux"

# Map example module -> gen-dsp platform (one of each).
# These platforms build out-of-the-box with cmake + make (no external SDKs).
# PD/Max require additional SDKs; vcvrack/daisy/circle need special toolchains.
# AU is macOS-only; on Linux we skip it (5 targets instead of 6).
_ALL_TARGETS: list[tuple[str, str]] = [
    ("stereo_gain", "chuck"),
    ("onepole", "clap"),
    ("fbdelay", "au"),
    ("wavetable", "lv2"),
    ("smooth_gain", "sc"),
    ("multirate_synth", "vst3"),
]

TARGETS = [(m, p) for m, p in _ALL_TARGETS if not (p == "au" and IS_LINUX)]

# Expected entry-point symbol (substring) exported by each platform binary.
# macOS nm prepends underscore to C symbols; Linux does not.
_ENTRY_SYMBOLS: dict[str, str] = {
    "chuck": "ck_query",
    "clap": "clap_entry",
    "au": "AUGenFactory",
    "lv2": "lv2_descriptor",
    "sc": "T _load" if IS_MACOS else "T load",
    "vst3": "GetPluginFactory",
}

BUILD_DIR = Path("build/gen_dsp")


def _generate_only() -> int:
    """Generate adapter files (no build system, no build)."""
    examples_dir = str(Path(__file__).resolve().parent)
    if examples_dir not in sys.path:
        sys.path.insert(0, examples_dir)

    BUILD_DIR.mkdir(parents=True, exist_ok=True)
    errors = 0

    for module_name, platform in TARGETS:
        mod = importlib.import_module(module_name)
        graph = mod.graph
        out_dir = BUILD_DIR / f"{graph.name}_{platform}"

        try:
            compile_for_gen_dsp(graph, out_dir, platform)
        except Exception as exc:
            print(f"  FAIL {module_name} -> {platform}: {exc}")
            errors += 1
            continue

        expected = [
            f"{graph.name}.cpp",
            f"_ext_{platform}.cpp",
            "manifest.json",
        ]
        missing = [f for f in expected if not (out_dir / f).is_file()]
        if missing:
            print(f"  FAIL {module_name} -> {platform}: missing {missing}")
            errors += 1
        else:
            print(f"  OK   {module_name} -> {platform}  ({out_dir})")

    if errors:
        print(f"\n{errors} target(s) failed.")
        return 1
    print(f"\nAll {len(TARGETS)} gen-dsp targets generated successfully.")
    return 0


def _assemble_buildable_project(
    graph: object,
    output_dir: Path,
    platform: str,
) -> Path:
    """Assemble a fully buildable gen-dsp project.

    Uses gen-dsp's Platform.generate_project() for build files,
    then overlays dsp-graph adapter code and creates genlib stubs.
    """
    from gen_dsp.core.manifest import Manifest  # type: ignore[import-untyped]
    from gen_dsp.platforms import get_platform  # type: ignore[import-untyped]

    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    # 1. Build Manifest from graph
    manifest_json = generate_manifest(graph)  # type: ignore[arg-type]
    manifest_dict = json.loads(manifest_json)
    manifest = Manifest.from_dict(manifest_dict)
    lib_name = manifest.gen_name

    # 2. Generate full project skeleton (build files + platform templates)
    plat = get_platform(platform)
    plat.generate_project(manifest, out, lib_name)

    # 3. Create stub genlib files (build system references these but
    #    dsp-graph adapter doesn't need them -- empty stubs compile to no-ops)
    gen_dsp_dir = out / "gen" / "gen_dsp"
    gen_dsp_dir.mkdir(parents=True, exist_ok=True)
    (gen_dsp_dir / "genlib.cpp").write_text("// stub -- dsp-graph adapter\n")
    (gen_dsp_dir / "json.c").write_text("// stub\n")
    (gen_dsp_dir / "json_builder.c").write_text("// stub\n")

    # 4. Create stub gen~ export files (referenced by GEN_EXPORTED_HEADER/CPP
    #    macros, but the dsp-graph adapter doesn't use them)
    gen_dir = out / "gen"
    (gen_dir / f"{lib_name}.cpp").write_text("// stub -- replaced by dsp-graph\n")
    (gen_dir / f"{lib_name}.h").write_text("// stub -- replaced by dsp-graph\n")

    # 5. Write dsp-graph compiled code
    code = compile_graph(graph)  # type: ignore[arg-type]
    (out / f"{lib_name}.cpp").write_text(code)

    # 6. Overwrite _ext_{platform}.cpp with dsp-graph adapter
    adapter = generate_adapter_cpp(graph, platform)  # type: ignore[arg-type]
    (out / f"_ext_{platform}.cpp").write_text(adapter)

    # 7. Overwrite manifest.json
    (out / "manifest.json").write_text(manifest_json)

    return out


def _find_binary(output_file: Path) -> Path | None:
    """Resolve the actual shared-library binary from a build output path.

    Handles bundles (.component, .vst3, .lv2) by searching inside.
    """
    if output_file.is_file():
        return output_file

    if not output_file.is_dir():
        return None

    # macOS bundle: Contents/MacOS/<name>
    macos_dir = output_file / "Contents" / "MacOS"
    if macos_dir.is_dir():
        for f in macos_dir.iterdir():
            if f.is_file():
                return f

    # LV2/generic bundle: <name>.dylib or <name>.so
    for f in output_file.iterdir():
        if f.suffix in (".dylib", ".so") and f.is_file():
            return f

    return None


def _validate_binary(binary: Path, platform: str) -> list[str]:
    """Validate a built binary. Returns list of error strings (empty = OK)."""
    errors: list[str] = []

    # 1. Exists and non-empty
    if not binary.is_file():
        return [f"binary not found: {binary}"]
    size = binary.stat().st_size
    if size < 1024:
        errors.append(f"suspiciously small ({size} bytes)")

    # 2. Valid shared library (Mach-O on macOS, ELF on Linux)
    try:
        result = subprocess.run(
            ["file", str(binary)],
            capture_output=True, text=True, timeout=5,
        )
        file_type = result.stdout.strip()
        if IS_MACOS:
            if "Mach-O" not in file_type:
                errors.append(f"not a Mach-O binary: {file_type}")
            elif ("dynamically linked shared library" not in file_type
                  and "bundle" not in file_type):
                errors.append(f"not a shared library/bundle: {file_type}")
        elif IS_LINUX:
            if "ELF" not in file_type:
                errors.append(f"not an ELF binary: {file_type}")
            elif "shared object" not in file_type and "pie executable" not in file_type:
                errors.append(f"not a shared object: {file_type}")
        else:
            # Best-effort on other platforms
            if "executable" not in file_type.lower() and "library" not in file_type.lower():
                errors.append(f"unexpected file type: {file_type}")
    except Exception as exc:
        errors.append(f"file command failed: {exc}")

    # 3. Expected entry-point symbol
    expected_sym = _ENTRY_SYMBOLS.get(platform)
    if expected_sym:
        try:
            nm_flags = ["-g", "--defined-only"] if IS_LINUX else ["-g"]
            result = subprocess.run(
                ["nm", *nm_flags, str(binary)],
                capture_output=True, text=True, timeout=10,
            )
            if expected_sym not in result.stdout:
                errors.append(f"missing entry symbol '{expected_sym}'")
        except Exception as exc:
            errors.append(f"nm command failed: {exc}")

    return errors


def _validate_vst3(output_file: Path, project_dir: Path) -> tuple[list[str], list[str]]:
    """Run VST3 SDK moduleinfotool and validator on a .vst3 bundle.

    Returns (errors, info) where errors are fatal and info are advisory lines.
    """
    import re as _re

    errors: list[str] = []
    info: list[str] = []

    bin_dir = project_dir / "build" / "bin" / "Release"
    moduleinfotool = bin_dir / "moduleinfotool"
    validator = bin_dir / "validator"

    # moduleinfotool -create: extract and verify module metadata
    if moduleinfotool.is_file():
        try:
            result = subprocess.run(
                [str(moduleinfotool), "-create", "-version", "1.0.0",
                 "-path", str(output_file)],
                capture_output=True, text=True, timeout=15,
            )
            if result.returncode != 0:
                errors.append(f"moduleinfotool failed (rc={result.returncode})")
            else:
                # VST3 SDK outputs JSON5 with trailing commas -- strip them
                cleaned = _re.sub(r",(\s*[}\]])", r"\1", result.stdout)
                module_info = json.loads(cleaned)
                classes = module_info.get("Classes", [])
                if not classes:
                    errors.append("moduleinfotool: no VST3 classes found")
                else:
                    cls = classes[0]
                    info.append(f"moduleinfo: {cls.get('Name')} "
                                f"[{', '.join(cls.get('Sub Categories', []))}] "
                                f"SDK {cls.get('SDKVersion')}")
        except json.JSONDecodeError:
            errors.append("moduleinfotool: invalid JSON output")
        except Exception as exc:
            errors.append(f"moduleinfotool: {exc}")

    # validator: run the VST3 SDK test suite
    if validator.is_file():
        try:
            result = subprocess.run(
                [str(validator), str(output_file)],
                capture_output=True, text=True, timeout=30,
            )
            output = result.stdout + result.stderr
            # Parse "Result: N tests passed, M tests failed"
            for line in output.splitlines():
                if "Result:" in line and "passed" in line:
                    info.append(f"validator: {line.strip()}")
                    m = _re.search(r"(\d+) tests passed.*?(\d+) tests failed", line)
                    if m:
                        passed, failed = int(m.group(1)), int(m.group(2))
                        if passed == 0:
                            errors.append("validator: zero tests passed")
                    break
        except Exception as exc:
            errors.append(f"validator: {exc}")

    return errors, info


def _generate_and_build(validate: bool = False) -> int:
    """Generate full buildable projects, build, and optionally validate."""
    from gen_dsp.platforms import get_platform  # type: ignore[import-untyped]

    examples_dir = str(Path(__file__).resolve().parent)
    if examples_dir not in sys.path:
        sys.path.insert(0, examples_dir)

    BUILD_DIR.mkdir(parents=True, exist_ok=True)

    # Shared FetchContent cache to avoid re-downloading SDKs
    cache_dir = BUILD_DIR / "_fetchcontent_cache"
    cache_dir.mkdir(exist_ok=True)
    os.environ["GEN_DSP_CACHE_DIR"] = str(cache_dir.resolve())
    # Prevent git from hanging on credential prompts
    os.environ["GIT_TERMINAL_PROMPT"] = "0"

    errors = 0
    built = 0

    for module_name, platform in TARGETS:
        mod = importlib.import_module(module_name)
        graph = mod.graph
        out_dir = BUILD_DIR / f"{graph.name}_{platform}"

        # Assemble
        print(f"  GEN  {module_name} -> {platform} ...", flush=True)
        try:
            _assemble_buildable_project(graph, out_dir, platform)
        except Exception as exc:
            print(f"  FAIL {module_name} -> {platform} (generate): {exc}")
            errors += 1
            continue

        # Build
        print(f"  BLD  {module_name} -> {platform} ...", flush=True)
        try:
            plat = get_platform(platform)
            result = plat.build(out_dir, clean=False, verbose=False)
            if not result.success:
                stderr_tail = (result.stderr or "")[-300:]
                print(f"  FAIL {module_name} -> {platform} (build):")
                if stderr_tail:
                    print(f"         {stderr_tail}")
                errors += 1
                continue
        except Exception as exc:
            print(f"  FAIL {module_name} -> {platform} (build): {exc}")
            errors += 1
            continue

        # Validate
        if validate and result.output_file:
            binary = _find_binary(result.output_file)
            if binary is None:
                print(f"  FAIL {module_name} -> {platform} (validate): "
                      f"cannot locate binary in {result.output_file}")
                errors += 1
                continue

            val_errors = _validate_binary(binary, platform)
            if val_errors:
                print(f"  FAIL {module_name} -> {platform} (validate):")
                for e in val_errors:
                    print(f"         {e}")
                errors += 1
                continue

            # VST3-specific: run SDK moduleinfotool + validator
            vst3_info: list[str] = []
            if platform == "vst3":
                vst3_errors, vst3_info = _validate_vst3(
                    result.output_file, out_dir)
                if vst3_errors:
                    print(f"  FAIL {module_name} -> {platform} (vst3 validate):")
                    for e in vst3_errors:
                        print(f"         {e}")
                    errors += 1
                    continue

            size_str = f"{binary.stat().st_size:,} bytes"
            extra = f"  [{'; '.join(vst3_info)}]" if vst3_info else ""
            print(f"  OK   {module_name} -> {platform}  {binary.name} "
                  f"({size_str}){extra}")
        else:
            output = result.output_file or "unknown"
            print(f"  OK   {module_name} -> {platform}  ({output})")

        built += 1

    print()
    label = "built and validated" if validate else "built"
    if errors:
        print(f"{built}/{len(TARGETS)} {label}, {errors} failed.")
        return 1
    print(f"All {len(TARGETS)} gen-dsp targets {label} successfully.")
    return 0


def main() -> int:
    if "--validate" in sys.argv:
        return _generate_and_build(validate=True)
    if "--build" in sys.argv:
        return _generate_and_build(validate=False)
    return _generate_only()


if __name__ == "__main__":
    sys.exit(main())
