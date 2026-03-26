from __future__ import annotations

import argparse
import base64
import os
import pathlib
import shutil
import subprocess
import sys
import tempfile

from typing import Optional

try:
    from PIL import Image
except ImportError as e:
    print(
        "Pillow is required to run this script. Install it with `pip install pillow`."
    )
    raise e


def _ensure_dir(path: pathlib.Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def _export_raster(
    img: Image.Image,
    tiff_path: pathlib.Path,
    pdf_path: Optional[pathlib.Path],
    dpi: int,
    max_width: int,
    max_height: int,
    upscale: float = 1.0,
) -> None:
    w, h = img.size
    scale = min(1.0, max_width / w, max_height / h) * upscale
    if scale != 1.0:
        new_size = (int(round(w * scale)), int(round(h * scale)))
        img = img.resize(new_size, Image.LANCZOS)
        w, h = img.size

    if img.mode not in ("RGBA", "RGB"):
        img = img.convert("RGBA")

    img.save(tiff_path, format="TIFF", dpi=(600, 600), compression="tiff_lzw")

    if pdf_path:
        if img.mode == "RGBA":
            bg = Image.new("RGB", img.size, (255, 255, 255))
            bg.paste(img, mask=img.split()[3])
            bg.save(pdf_path, format="PDF", resolution=600)
        else:
            img.save(pdf_path, format="PDF", resolution=600)


def _pdf_to_tiff_cairosvg(
    pdf_path: pathlib.Path, output_path: pathlib.Path, dpi: int
) -> bool:
    try:
        import cairosvg

        with tempfile.TemporaryDirectory() as td:
            png_path = pathlib.Path(td) / "out.png"
            cairosvg.png_from_pdf(
                file_obj=str(pdf_path),
                write_to=str(png_path),
                dpi=600,
            )
            img = Image.open(png_path)
            _export_raster(
                img,
                output_path,
                None,
                dpi=600,
                max_width=img.width,
                max_height=img.height,
            )
        return True
    except ImportError:
        return False
    except Exception as e:
        print(f"[warning] cairosvg failed to convert {pdf_path.name}: {e}")
        return False


def _pdf_to_tiff_pdftoppm(
    pdf_path: pathlib.Path, output_path: pathlib.Path, dpi: int
) -> bool:
    pdftoppm = shutil.which("pdftoppm")
    if not pdftoppm:
        return False

    with tempfile.TemporaryDirectory() as td:
        png_path = pathlib.Path(td) / "out.png"
        cmd = [
            pdftoppm,
            "-singlefile",
            "-png",
            f"-r",
            "600",
            str(pdf_path),
            str(png_path.with_suffix("").as_posix()),
        ]
        try:
            subprocess.run(
                cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL
            )
        except subprocess.CalledProcessError:
            return False

        try:
            img = Image.open(png_path)
            _export_raster(
                img,
                output_path,
                None,
                dpi=600,
                max_width=img.width,
                max_height=img.height,
            )
            return True
        except Exception:
            return False


def _pdf_to_tiff_ghostscript(
    pdf_path: pathlib.Path, output_path: pathlib.Path, dpi: int
) -> bool:
    gs = shutil.which("gs")
    if not gs:
        return False

    with tempfile.TemporaryDirectory() as td:
        png_path = pathlib.Path(td) / "out.png"
        cmd = [
            gs,
            "-dSAFER",
            "-dBATCH",
            "-dNOPAUSE",
            "-dFirstPage=1",
            "-dLastPage=1",
            "-sDEVICE=pngalpha",
            "-r600",
            f"-sOutputFile={png_path}",
            str(pdf_path),
        ]
        try:
            subprocess.run(
                cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL
            )
        except subprocess.CalledProcessError:
            return False

        try:
            img = Image.open(png_path)
            _export_raster(
                img,
                output_path,
                None,
                dpi=600,
                max_width=img.width,
                max_height=img.height,
            )
            return True
        except Exception:
            return False


def _process_file(
    file_path: pathlib.Path,
    tiff_dir: pathlib.Path,
    pdf_dir: pathlib.Path,
    dpi: int,
    max_width: int,
    max_height: int,
    upscale: float,
) -> None:
    tiff_path = tiff_dir / (file_path.stem + ".tiff")
    pdf_path = pdf_dir / (file_path.stem + ".pdf")

    if file_path.suffix.lower() in {".png", ".jpg", ".jpeg", ".tif", ".tiff"}:
        img = Image.open(file_path)
        _export_raster(
            img,
            tiff_path,
            pdf_path,
            dpi=600,
            max_width=max_width,
            max_height=max_height,
            upscale=upscale,
        )
        return

    if file_path.suffix.lower() == ".pdf":
        shutil.copy2(file_path, pdf_path)

        if _pdf_to_tiff_cairosvg(file_path, tiff_path, dpi=600):
            return

        if _pdf_to_tiff_pdftoppm(file_path, tiff_path, dpi=600):
            return

        if _pdf_to_tiff_ghostscript(file_path, tiff_path, dpi=600):
            return

        raise RuntimeError(
            "Unable to convert PDF to TIFF: install cairosvg, pdftoppm (poppler), or gs (ghostscript) to enable conversion."
        )

    raise ValueError(f"Unsupported file type: {file_path}")


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Standardize and export figures into PDF and TIFF folders."
    )
    parser.add_argument(
        "--input-dir",
        default="Output",
        help="Directory containing figure files (default: Output)",
    )
    parser.add_argument(
        "--output-dir",
        default="Figures",
        help="Directory to write standardized files to (default: Output/standardized_figures)",
    )
    parser.add_argument(
        "--dpi",
        type=int,
        default=600,
        help="Target DPI for raster upscaling (default: 600)",
    )
    parser.add_argument(
        "--max-width",
        type=int,
        default=7200,
        help="Maximum allowed width in pixels (default: 7200)",
    )
    parser.add_argument(
        "--max-height",
        type=int,
        default=7200,
        help="Maximum allowed height in pixels (default: 7200)",
    )
    parser.add_argument(
        "--upscale",
        type=float,
        default=1.0,
        help="Additional upscale factor to apply after fitting to max dimensions (default: 1.0)",
    )

    args, _ = parser.parse_known_args()

    input_dir = pathlib.Path(args.input_dir)
    output_dir = pathlib.Path(args.output_dir)

    if not input_dir.exists():
        print(f"Input directory does not exist: {input_dir}")
        return 1

    tiff_dir = output_dir / "tiff"
    pdf_dir = output_dir

    _ensure_dir(tiff_dir)
    _ensure_dir(pdf_dir)

    supported_exts = {".png", ".jpg", ".jpeg", ".tif", ".tiff", ".pdf"}

    files = sorted(
        [
            p
            for p in input_dir.iterdir()
            if p.is_file() and p.suffix.lower() in supported_exts
        ]
    )

    if not files:
        print(f"No supported figure files found in {input_dir}")
        return 0

    print(
        f"Processing {len(files)} figure(s) from {input_dir} -> {output_dir}/pdf & {output_dir}/tiff (forced dpi=600)"
    )

    for f in files:
        try:
            _process_file(
                f,
                tiff_dir,
                pdf_dir,
                dpi=600,
                max_width=args.max_width,
                max_height=args.max_height,
                upscale=args.upscale,
            )
            print(f"  ✅ {f.name} -> tiff/ & pdf/")
        except Exception as e:
            print(f"  ⚠️  Failed to process {f.name}: {e}")

    print("Done.")
    return 0


if __name__ == "__main__":
    main()
