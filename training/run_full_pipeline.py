#!/usr/bin/env python3
"""
Full Training & Deployment Pipeline

Runs the entire end-to-end workflow:
  1. Export training data from each ChromaDB collection
  2. Merge training manifests into one combined dataset
  3. Fine-tune DINOv2 on the combined dataset
  4. Re-embed each collection with the fine-tuned model
  5. Upload fine-tuned weights to RunPod via S3
  6. Sync ChromaDB to RunPod via S3
  7. Clean up old/stale ChromaDB collections

Usage:
    python training/run_full_pipeline.py
    python training/run_full_pipeline.py --chromadb "\\\\192.168.1.14\\data\\scraper\\chromadb"
    python training/run_full_pipeline.py --skip-training    # re-embed + deploy only
    python training/run_full_pipeline.py --skip-export      # already exported, just train
    python training/run_full_pipeline.py --step 4           # resume from step 4
    python training/run_full_pipeline.py --epochs 10 --batch-size 16
"""
import argparse
import json
import os
import subprocess
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.prompt import Confirm, Prompt

console = Console()

# ── Default collection config ────────────────────────────────────────────
# Each entry: (collection_name, export_dir, finetuned_collection_name)
DEFAULT_COLLECTIONS = [
    {
        "name": "pokemon_embeddings_dinov2",
        "export_dir": "./training_data_pokemon",
        "finetuned_name": "pokemon_embeddings_dinov2_finetuned",
    },
    {
        "name": "card_embeddings_dinov2",
        "export_dir": "./training_data_sports",
        "finetuned_name": "card_embeddings_dinov2_finetuned",
    },
]

# Collections to delete during cleanup (old/stale)
OLD_COLLECTIONS = [
    "card_images",
    "card_images_dinov2",
]

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_DIR = os.path.dirname(SCRIPT_DIR)


def run_command(cmd: list[str], description: str) -> bool:
    """Run a subprocess command with live output. Returns True on success."""
    console.print(f"\n[dim]$ {' '.join(cmd)}[/dim]\n")
    try:
        result = subprocess.run(
            cmd,
            cwd=PROJECT_DIR,
            check=True,
        )
        return True
    except subprocess.CalledProcessError as e:
        console.print(f"[red]Command failed (exit code {e.returncode})[/red]")
        return False
    except KeyboardInterrupt:
        console.print("\n[yellow]Interrupted by user[/yellow]")
        return False


def get_chromadb_path(args_chromadb: str | None) -> str:
    """Resolve ChromaDB path from args or config."""
    if args_chromadb:
        return args_chromadb
    try:
        import config
        return config.CHROMA_DIR
    except ImportError:
        console.print("[red]Cannot import config.py — specify --chromadb path[/red]")
        sys.exit(1)


def show_plan(collections: list[dict], chromadb_path: str, args):
    """Display what the pipeline will do."""
    console.print(Panel.fit(
        "[bold cyan]DINOv2 Fine-Tuning & Deployment Pipeline[/bold cyan]",
        border_style="cyan",
    ))

    table = Table(title="Collections to process", show_header=True)
    table.add_column("Source Collection", style="cyan")
    table.add_column("Export Dir", style="dim")
    table.add_column("Fine-tuned Collection", style="green")

    for col in collections:
        table.add_row(col["name"], col["export_dir"], col["finetuned_name"])

    console.print(table)
    console.print(f"\n  ChromaDB:     [cyan]{chromadb_path}[/cyan]")
    console.print(f"  Epochs:       [cyan]{args.epochs}[/cyan]")
    console.print(f"  Batch size:   [cyan]{args.batch_size}[/cyan]")
    console.print(f"  Checkpoint:   [cyan]./checkpoints/dinov2_finetuned_best.pt[/cyan]")

    steps = []
    if args.step <= 1 and not args.skip_export:
        steps.append("1. Export training data from each collection")
    if args.step <= 2 and not args.skip_export:
        steps.append("2. Merge training manifests")
    if args.step <= 3 and not args.skip_training:
        steps.append("3. Fine-tune DINOv2 on combined dataset")
    if args.step <= 4:
        steps.append("4. Re-embed all collections with fine-tuned model")
    if args.step <= 5:
        steps.append("5. Upload fine-tuned weights to RunPod (S3)")
    if args.step <= 6:
        steps.append("6. Sync ChromaDB to RunPod (S3)")
    if args.step <= 7:
        steps.append("7. Clean up old collections")
    if args.step <= 8 and not args.skip_variant_classifier:
        steps.append("8. Export variant-classifier training data (Part B)")
    if args.step <= 9 and not args.skip_variant_classifier:
        steps.append(f"9. Train variant classifier ({args.variant_arch}) (Part B)")

    console.print(f"\n[bold]Steps to run:[/bold]")
    for s in steps:
        console.print(f"  {s}")
    console.print()


# ══════════════════════════════════════════════════════════════════════════
# Step 1: Export training data
# ══════════════════════════════════════════════════════════════════════════

def step1_export(collections: list[dict], chromadb_path: str) -> bool:
    console.print(Panel("[bold]Step 1: Export Training Data[/bold]", border_style="cyan"))

    for col in collections:
        console.print(f"\n[cyan]Exporting {col['name']}...[/cyan]")

        # Check if already exported
        manifest_path = os.path.join(PROJECT_DIR, col["export_dir"], "manifest.json")
        if os.path.exists(manifest_path):
            with open(manifest_path) as f:
                existing = json.load(f)
            console.print(f"  [yellow]Already exported ({len(existing):,} cards). Re-export?[/yellow]")
            if not Confirm.ask("  Re-export?", default=False):
                continue

        cmd = [
            sys.executable, os.path.join(SCRIPT_DIR, "01_export_training_data.py"),
            "--chromadb", chromadb_path,
            "--collection", col["name"],
            "--output", col["export_dir"],
        ]
        if not run_command(cmd, f"Export {col['name']}"):
            return False

    return True


# ══════════════════════════════════════════════════════════════════════════
# Step 2: Merge training manifests
# ══════════════════════════════════════════════════════════════════════════

def step2_merge(collections: list[dict]) -> bool:
    console.print(Panel("[bold]Step 2: Merge Training Manifests[/bold]", border_style="cyan"))

    input_dirs = [col["export_dir"] for col in collections]

    # Verify all exports exist
    for d in input_dirs:
        manifest = os.path.join(PROJECT_DIR, d, "manifest.json")
        if not os.path.exists(manifest):
            console.print(f"[red]Missing {manifest} — run Step 1 first[/red]")
            return False

    cmd = [
        sys.executable, os.path.join(SCRIPT_DIR, "merge_training_data.py"),
        "--inputs", *input_dirs,
        "--output", "./training_data_combined",
    ]
    return run_command(cmd, "Merge manifests")


# ══════════════════════════════════════════════════════════════════════════
# Step 3: Fine-tune DINOv2
# ══════════════════════════════════════════════════════════════════════════

def step3_train(epochs: int, batch_size: int, lr: float) -> bool:
    console.print(Panel("[bold]Step 3: Fine-Tune DINOv2[/bold]", border_style="cyan"))

    manifest = "./training_data_combined/manifest.json"
    hard_neg = "./training_data_combined/hard_negatives.json"

    if not os.path.exists(os.path.join(PROJECT_DIR, manifest)):
        console.print(f"[red]Missing {manifest} — run Step 2 first[/red]")
        return False

    # Check for existing checkpoint
    best_ckpt = os.path.join(PROJECT_DIR, "checkpoints", "dinov2_finetuned_best.pt")
    if os.path.exists(best_ckpt):
        size_mb = os.path.getsize(best_ckpt) / (1024 ** 2)
        console.print(f"  [yellow]Existing checkpoint found: {best_ckpt} ({size_mb:.0f} MB)[/yellow]")
        console.print(f"  [yellow]Training will overwrite it.[/yellow]")
        if not Confirm.ask("  Continue with training?", default=True):
            console.print("  [dim]Skipping training, using existing checkpoint[/dim]")
            return True

    cmd = [
        sys.executable, os.path.join(SCRIPT_DIR, "02_finetune_dinov2.py"),
        "--manifest", manifest,
        "--hard-negatives", hard_neg,
        "--epochs", str(epochs),
        "--batch-size", str(batch_size),
        "--lr", str(lr),
    ]
    return run_command(cmd, "Fine-tune DINOv2")


# ══════════════════════════════════════════════════════════════════════════
# Step 4: Re-embed all collections
# ══════════════════════════════════════════════════════════════════════════

def step4_reembed(collections: list[dict], chromadb_path: str, batch_size: int) -> bool:
    console.print(Panel("[bold]Step 4: Re-Embed Collections[/bold]", border_style="cyan"))

    # Find the best checkpoint
    best_ckpt = os.path.join(PROJECT_DIR, "checkpoints", "dinov2_finetuned_best.pt")
    backbone_ckpt = os.path.join(PROJECT_DIR, "checkpoints", "dinov2_finetuned_backbone.pt")

    checkpoint = None
    if os.path.exists(best_ckpt):
        checkpoint = best_ckpt
    elif os.path.exists(backbone_ckpt):
        checkpoint = backbone_ckpt
    else:
        console.print("[red]No checkpoint found in ./checkpoints/[/red]")
        console.print("[yellow]Run Step 3 first, or provide a checkpoint.[/yellow]")
        return False

    console.print(f"  Using checkpoint: [cyan]{checkpoint}[/cyan]\n")

    # Re-embed batch size for inference (can be higher than training)
    reembed_batch = min(batch_size * 4, 128)

    for col in collections:
        console.print(f"\n[cyan]Re-embedding → {col['finetuned_name']}...[/cyan]")

        manifest = os.path.join(PROJECT_DIR, col["export_dir"], "manifest.json")
        if not os.path.exists(manifest):
            console.print(f"[red]Missing {manifest} — run Step 1 first[/red]")
            return False

        cmd = [
            sys.executable, os.path.join(SCRIPT_DIR, "03_reembed_collection.py"),
            "--manifest", manifest,
            "--checkpoint", checkpoint,
            "--chromadb", chromadb_path,
            "--collection", col["finetuned_name"],
            "--batch", str(reembed_batch),
        ]
        if not run_command(cmd, f"Re-embed {col['finetuned_name']}"):
            return False

    return True


# ══════════════════════════════════════════════════════════════════════════
# Step 5: Upload fine-tuned weights to RunPod via S3
# ══════════════════════════════════════════════════════════════════════════

def step5_upload_weights() -> bool:
    console.print(Panel("[bold]Step 5: Upload Weights to RunPod[/bold]", border_style="cyan"))

    try:
        import config
        import boto3
    except ImportError as e:
        console.print(f"[red]Missing dependency: {e}[/red]")
        console.print("[yellow]pip install boto3[/yellow]")
        return False

    if not config.RUNPOD_S3_ACCESS_KEY or not config.RUNPOD_S3_SECRET_KEY:
        console.print("[red]RunPod S3 credentials not set.[/red]")
        console.print('  Set: $env:RUNPOD_S3_ACCESS_KEY = "..."')
        console.print('  Set: $env:RUNPOD_S3_SECRET_KEY = "..."')
        console.print("[yellow]Skipping upload — you can do this manually later.[/yellow]")
        return True  # Non-fatal, user can upload manually

    # Find checkpoint to upload
    best_ckpt = os.path.join(PROJECT_DIR, "checkpoints", "dinov2_finetuned_best.pt")
    backbone_ckpt = os.path.join(PROJECT_DIR, "checkpoints", "dinov2_finetuned_backbone.pt")

    checkpoint = best_ckpt if os.path.exists(best_ckpt) else backbone_ckpt
    if not os.path.exists(checkpoint):
        console.print("[red]No checkpoint found to upload.[/red]")
        return False

    size_mb = os.path.getsize(checkpoint) / (1024 ** 2)
    s3_key = "dinov2_finetuned_backbone.pt"

    console.print(f"  Local:  [cyan]{checkpoint}[/cyan] ({size_mb:.0f} MB)")
    console.print(f"  Remote: [cyan]s3://{config.RUNPOD_S3_BUCKET}/{s3_key}[/cyan]")

    s3 = boto3.client(
        "s3",
        endpoint_url=config.RUNPOD_S3_ENDPOINT,
        aws_access_key_id=config.RUNPOD_S3_ACCESS_KEY,
        aws_secret_access_key=config.RUNPOD_S3_SECRET_KEY,
        region_name="us-nc-1",
    )

    console.print("\n  [cyan]Uploading weights...[/cyan]")
    start = time.time()
    try:
        s3.upload_file(
            checkpoint,
            config.RUNPOD_S3_BUCKET,
            s3_key,
            Callback=_S3ProgressCallback(size_mb),
        )
        elapsed = time.time() - start
        console.print(f"\n  [green]Uploaded in {elapsed:.0f}s[/green]")
    except Exception as e:
        console.print(f"\n  [red]Upload failed: {e}[/red]")
        return False

    # Also upload variant classifier checkpoint if it exists (Part B).
    for fname in ("variant_classifier_linear.pt", "variant_classifier_mlp.pt"):
        v_ckpt = os.path.join(PROJECT_DIR, "checkpoints", fname)
        if not os.path.exists(v_ckpt):
            continue
        v_size = os.path.getsize(v_ckpt) / (1024 ** 2)
        console.print(f"\n  [cyan]Uploading variant classifier {fname} ({v_size:.1f} MB)...[/cyan]")
        try:
            s3.upload_file(v_ckpt, config.RUNPOD_S3_BUCKET, fname,
                           Callback=_S3ProgressCallback(v_size))
            console.print(f"\n  [green]Uploaded {fname}[/green]")
        except Exception as e:
            console.print(f"\n  [yellow]Variant classifier upload failed (non-fatal): {e}[/yellow]")

    return True


class _S3ProgressCallback:
    """Simple progress callback for S3 uploads."""

    def __init__(self, total_mb: float):
        self.uploaded = 0
        self.total_mb = total_mb

    def __call__(self, bytes_transferred):
        self.uploaded += bytes_transferred
        mb = self.uploaded / (1024 ** 2)
        pct = mb / self.total_mb * 100 if self.total_mb > 0 else 0
        console.print(f"  {mb:.0f}/{self.total_mb:.0f} MB ({pct:.0f}%)", end="\r")


# ══════════════════════════════════════════════════════════════════════════
# Step 6: Sync ChromaDB to RunPod
# ══════════════════════════════════════════════════════════════════════════

def step6_sync_chromadb() -> bool:
    console.print(Panel("[bold]Step 6: Sync ChromaDB to RunPod[/bold]", border_style="cyan"))

    cmd = [
        sys.executable, os.path.join(PROJECT_DIR, "sync_to_runpod.py"),
    ]
    return run_command(cmd, "Sync ChromaDB → RunPod S3")


# ══════════════════════════════════════════════════════════════════════════
# Step 7: Clean up old collections
# ══════════════════════════════════════════════════════════════════════════

def step7_cleanup(collections: list[dict], chromadb_path: str) -> bool:
    console.print(Panel("[bold]Step 7: Clean Up Old Collections[/bold]", border_style="cyan"))

    try:
        import chromadb
    except ImportError:
        console.print("[red]chromadb not installed[/red]")
        return False

    client = chromadb.PersistentClient(path=chromadb_path)

    # Get all existing collections
    all_collections = {}
    for col in client.list_collections():
        col_name = col.name if hasattr(col, 'name') else str(col)
        c = client.get_collection(col_name)
        all_collections[col_name] = c.count()

    # Determine which to keep
    keep_names = set()
    for col in collections:
        keep_names.add(col["finetuned_name"])  # Keep the new finetuned ones

    # Show current state
    table = Table(title="All ChromaDB Collections", show_header=True)
    table.add_column("Collection", style="cyan")
    table.add_column("Count", justify="right")
    table.add_column("Action", style="white")

    to_delete = []
    for name, count in sorted(all_collections.items()):
        if name in keep_names:
            table.add_row(name, f"{count:,}", "[green]KEEP (finetuned)[/green]")
        elif name in OLD_COLLECTIONS:
            table.add_row(name, f"{count:,}", "[red]DELETE (old/stale)[/red]")
            to_delete.append(name)
        else:
            # Collections not in keep or explicit delete — ask
            table.add_row(name, f"{count:,}", "[yellow]DELETE (replaced by finetuned)[/yellow]")
            to_delete.append(name)

    console.print(table)

    if not to_delete:
        console.print("\n[green]No collections to delete.[/green]")
        return True

    console.print(f"\n[yellow]Will delete {len(to_delete)} collection(s):[/yellow]")
    for name in to_delete:
        console.print(f"  - {name} ({all_collections[name]:,} embeddings)")

    if not Confirm.ask("\nProceed with deletion?", default=False):
        console.print("[dim]Skipped cleanup[/dim]")
        return True

    for name in to_delete:
        try:
            client.delete_collection(name)
            console.print(f"  [red]Deleted {name}[/red]")
        except Exception as e:
            console.print(f"  [red]Failed to delete {name}: {e}[/red]")

    # Show remaining
    console.print(f"\n[green]Remaining collections:[/green]")
    for col in client.list_collections():
        col_name = col.name if hasattr(col, 'name') else str(col)
        c = client.get_collection(col_name)
        console.print(f"  {col_name}: {c.count():,}")

    return True


# ══════════════════════════════════════════════════════════════════════════
# Step 8 (Part B): Export variant-classifier training data
# ══════════════════════════════════════════════════════════════════════════

def step8_export_variants(min_samples: int) -> bool:
    console.print(Panel("[bold]Step 8: Export Variant-Classifier Training Data[/bold]", border_style="cyan"))
    cmd = [
        sys.executable, os.path.join(SCRIPT_DIR, "04_export_variant_training_data.py"),
        "--min-samples", str(min_samples),
    ]
    return run_command(cmd, "Export variant-classifier data")


# ══════════════════════════════════════════════════════════════════════════
# Step 9 (Part B): Train variant classifier
# ══════════════════════════════════════════════════════════════════════════

def step9_train_variant_classifier(arch: str, epochs: int, batch: int, lr: float, resume: bool) -> bool:
    console.print(Panel(f"[bold]Step 9: Train Variant Classifier ({arch})[/bold]", border_style="cyan"))

    # Same fallback order as step 4: prefer _best, then _backbone.
    best_ckpt     = os.path.join(PROJECT_DIR, "checkpoints", "dinov2_finetuned_best.pt")
    backbone_ckpt = os.path.join(PROJECT_DIR, "checkpoints", "dinov2_finetuned_backbone.pt")
    finetuned_backbone = best_ckpt if os.path.exists(best_ckpt) else backbone_ckpt

    cmd = [
        sys.executable, os.path.join(SCRIPT_DIR, "05_train_variant_classifier.py"),
        "--arch", arch,
        "--epochs", str(epochs),
        "--batch", str(batch),
        "--lr", str(lr),
    ]
    if os.path.exists(finetuned_backbone):
        console.print(f"  Using fine-tuned backbone: [cyan]{finetuned_backbone}[/cyan]")
        cmd += ["--finetuned-backbone", finetuned_backbone]
    else:
        console.print(f"  [yellow]No fine-tuned backbone found — training on base DINOv2 features.[/yellow]")
    if resume:
        cmd += ["--resume"]  # auto-discovers the last checkpoint
    return run_command(cmd, "Train variant classifier")


# ══════════════════════════════════════════════════════════════════════════
# Main
# ══════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(
        description="Full DINOv2 fine-tuning & deployment pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Full pipeline from scratch
  python training/run_full_pipeline.py --chromadb "\\\\192.168.1.14\\data\\scraper\\chromadb"

  # Resume from re-embedding (training already done)
  python training/run_full_pipeline.py --step 4

  # Skip training, just re-embed and deploy with existing checkpoint
  python training/run_full_pipeline.py --skip-training

  # Add a new collection (e.g., baseball)
  python training/run_full_pipeline.py --collections pokemon_embeddings_dinov2 card_embeddings_dinov2 baseball_embeddings_dinov2
        """,
    )

    parser.add_argument("--chromadb", type=str, default=None,
                        help="ChromaDB path (default: auto-detect from config.py)")
    parser.add_argument("--collections", nargs="+", default=None,
                        help="Collection names to include (default: pokemon + sports)")
    parser.add_argument("--epochs", type=int, default=10,
                        help="Training epochs (default: 10, good for large combined datasets)")
    parser.add_argument("--batch-size", type=int, default=16,
                        help="Training batch size (default: 16)")
    parser.add_argument("--lr", type=float, default=1e-5,
                        help="Learning rate (default: 1e-5)")
    parser.add_argument("--step", type=int, default=1,
                        help="Resume from this step number (1-9)")
    parser.add_argument("--skip-training", action="store_true",
                        help="Skip training, use existing checkpoint for re-embedding + deploy")
    parser.add_argument("--skip-export", action="store_true",
                        help="Skip export, use existing training data")
    parser.add_argument("--skip-variant-classifier", action="store_true",
                        help="Skip Part B steps 8-9 (variant-classifier export + train)")
    parser.add_argument("--variant-arch", choices=["linear", "mlp"], default="linear",
                        help="Variant classifier head architecture (default: linear)")
    parser.add_argument("--variant-min-samples", type=int, default=10,
                        help="Min samples per class to keep as its own label (default: 10)")
    parser.add_argument("--variant-epochs", type=int, default=10)
    parser.add_argument("--variant-batch", type=int, default=128)
    parser.add_argument("--variant-lr", type=float, default=1e-3)
    parser.add_argument("--variant-resume", action="store_true",
                        help="Resume variant-classifier training from the last saved epoch")
    parser.add_argument("--dry-run", action="store_true",
                        help="Show plan without executing")

    args = parser.parse_args()

    # Resolve ChromaDB path
    chromadb_path = get_chromadb_path(args.chromadb)

    # Build collection list
    if args.collections:
        collections = []
        for name in args.collections:
            # Auto-generate export dir and finetuned name
            short = name.replace("_embeddings_dinov2", "").replace("_embeddings", "")
            collections.append({
                "name": name,
                "export_dir": f"./training_data_{short}",
                "finetuned_name": f"{name}_finetuned",
            })
    else:
        collections = DEFAULT_COLLECTIONS

    # Show plan
    show_plan(collections, chromadb_path, args)

    if args.dry_run:
        console.print("[dim]Dry run — exiting without making changes.[/dim]")
        return

    if not Confirm.ask("Start pipeline?", default=True):
        return

    start_time = time.time()

    # ── Step 1: Export ────────────────────────────────────────────────
    if args.step <= 1 and not args.skip_export:
        if not step1_export(collections, chromadb_path):
            console.print("[red]Step 1 failed. Fix errors and re-run with --step 1[/red]")
            return

    # ── Step 2: Merge ─────────────────────────────────────────────────
    if args.step <= 2 and not args.skip_export:
        if not step2_merge(collections):
            console.print("[red]Step 2 failed. Fix errors and re-run with --step 2[/red]")
            return

    # ── Step 3: Train ─────────────────────────────────────────────────
    if args.step <= 3 and not args.skip_training:
        if not step3_train(args.epochs, args.batch_size, args.lr):
            console.print("[red]Step 3 failed. Fix errors and re-run with --step 3[/red]")
            return

    # ── Step 4: Re-embed ──────────────────────────────────────────────
    if args.step <= 4:
        if not step4_reembed(collections, chromadb_path, args.batch_size):
            console.print("[red]Step 4 failed. Fix errors and re-run with --step 4[/red]")
            return

    # ── Step 5: Upload weights ────────────────────────────────────────
    if args.step <= 5:
        if not step5_upload_weights():
            console.print("[yellow]Step 5 had issues but continuing...[/yellow]")

    # ── Step 6: Sync ChromaDB ─────────────────────────────────────────
    if args.step <= 6:
        if not step6_sync_chromadb():
            console.print("[yellow]Step 6 had issues but continuing...[/yellow]")

    # ── Step 7: Cleanup ───────────────────────────────────────────────
    if args.step <= 7:
        step7_cleanup(collections, chromadb_path)

    # ── Step 8 (Part B): Export variant-classifier training data ──────
    if args.step <= 8 and not args.skip_variant_classifier:
        if not step8_export_variants(args.variant_min_samples):
            console.print("[yellow]Step 8 failed — skipping Part B[/yellow]")
            args.skip_variant_classifier = True

    # ── Step 9 (Part B): Train variant classifier ────────────────────
    if args.step <= 9 and not args.skip_variant_classifier:
        if not step9_train_variant_classifier(
            args.variant_arch, args.variant_epochs, args.variant_batch, args.variant_lr,
            args.variant_resume,
        ):
            console.print("[yellow]Step 9 failed — continuing[/yellow]")

    # ── Done ──────────────────────────────────────────────────────────
    total_time = time.time() - start_time
    hours = int(total_time // 3600)
    minutes = int((total_time % 3600) // 60)

    console.print(Panel.fit(
        f"[bold green]Pipeline complete![/bold green]\n\n"
        f"  Total time: {hours}h {minutes}m\n\n"
        f"  [bold]What happened:[/bold]\n"
        f"  - Fine-tuned model trained on all card types\n"
        f"  - All collections re-embedded with fine-tuned model\n"
        f"  - Weights + ChromaDB synced to RunPod\n"
        f"  - Old collections cleaned up\n\n"
        f"  [bold]RunPod env vars to set:[/bold]\n"
        f"  FINETUNED_WEIGHTS=/runpod-volume/dinov2_finetuned_backbone.pt\n"
        f"  COLLECTION_NAME={collections[0]['finetuned_name']}\n\n"
        f"  [bold]To verify:[/bold]\n"
        f"  python sync_to_runpod.py --verify\n\n"
        f"  [dim]Restart RunPod workers to pick up the new model.[/dim]",
        border_style="green",
        title="Done",
    ))


if __name__ == "__main__":
    main()
