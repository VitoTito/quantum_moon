from pathlib import Path
from glob import glob
import argparse
import shutil
from ultralytics import YOLO


class YOLOvRunner:
    """
    Unified runner for training, inference, and evaluation using YOLOv10/11 via Ultralytics.

    This class supports command-line interaction to train models, run inference,
    and evaluate trained weights on a dataset defined in YOLO format.

    Methods
    -------
    train(args) :
        Trains a YOLO model using the provided dataset and configuration.
    infer(args) :
        Runs inference on a given image or directory of images.
    evaluate(args) :
        Computes performance metrics (mAP, precision, recall) on the validation set.
    """

    @staticmethod
    def train(args: argparse.Namespace) -> None:
        model = YOLO(args.model)
        results = model.train(
            data=args.data,
            epochs=args.epochs,
            imgsz=args.imgsz,
            batch=args.batch,
            device=args.device,
            name=args.name,
        )

        # Trouver le bon dossier runs/detect/<name> (le plus récent)
        run_dirs = sorted(
            Path("runs/detect").glob(f"{args.name}*"),
            key=lambda p: p.stat().st_mtime,
            reverse=True
        )
        latest_run = run_dirs[0]
        best_model_path = latest_run / "weights" / "best.pt"

        if best_model_path.exists():
            dest_path = Path("models") / f"{args.name}.pt"
            dest_path.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy(best_model_path, dest_path)
            print(f"✅ Best model saved to: {dest_path}")
        else:
            print(f"⚠️ best.pt not found in: {latest_run / 'weights'}")

    @staticmethod
    def infer(args: argparse.Namespace) -> None:
        """
        Run inference using a trained YOLO model on images or a directory.

        Parameters
        ----------
        args : argparse.Namespace
            Parsed arguments including model weights and input source.
        """
        model = YOLO(args.weights)
        results = model(args.source, save=True, imgsz=args.imgsz, conf=args.conf)
        print("✅ Inference completed. Results saved in: runs/detect/predict")

    @staticmethod
    def evaluate(args: argparse.Namespace) -> None:
        """
        Evaluate a YOLO model on the validation set and display key metrics.

        Parameters
        ----------
        args : argparse.Namespace
            Parsed arguments specifying the model weights for evaluation.
        """
        model = YOLO(args.weights)
        metrics = model.val()

        print("\n📊 Evaluation Metrics:")
        print(f"mAP@0.5      : {metrics.box.map50:.4f}")
        print(f"mAP@0.5:0.95 : {metrics.box.map:.4f}")
        print(f"Precision    : {metrics.box.precision:.4f}")
        print(f"Recall       : {metrics.box.recall:.4f}")


def parse_arguments() -> argparse.Namespace:
    """
    Parses command-line arguments for training, inference, and evaluation.

    Returns
    -------
    argparse.Namespace
        Object containing all parsed CLI arguments.
    """
    parser = argparse.ArgumentParser(
        description="YOLOv10/v11 Training, Inference, and Evaluation CLI"
    )
    subparsers = parser.add_subparsers(dest="command", help="Sub-commands")

    # --- Train ---
    train_parser = subparsers.add_parser("train", help="Train the YOLO model")
    train_parser.add_argument("--model", type=str, default="yolo11s.pt", help="YOLO model name or path")
    train_parser.add_argument("--data", type=str, default="data/dataset.yaml", help="Path to dataset.yaml")
    train_parser.add_argument("--epochs", type=int, default=100, help="Number of training epochs")
    train_parser.add_argument("--imgsz", type=int, default=640, help="Image size")
    train_parser.add_argument("--batch", type=int, default=16, help="Batch size")
    train_parser.add_argument("--device", type=int, default=0, help="GPU device ID (or 'cpu')")
    train_parser.add_argument("--name", type=str, default="yolo11-quantummoon", help="Run name")

    # --- Inference ---
    infer_parser = subparsers.add_parser("infer", help="Run inference on images")
    infer_parser.add_argument("--weights", type=str, default="models/yolo11-quantummoon.pt", help="Trained model path")
    infer_parser.add_argument("--source", type=str, default="data/images/val", help="Image file or directory")
    infer_parser.add_argument("--imgsz", type=int, default=640, help="Image size for inference")
    infer_parser.add_argument("--conf", type=float, default=0.25, help="Confidence threshold")

    # --- Evaluate ---
    eval_parser = subparsers.add_parser("eval", help="Evaluate model on validation data")
    eval_parser.add_argument("--weights", type=str, default="models/yolo11-quantummoon.pt", help="Model weights path")

    return parser.parse_args()


if __name__ == "__main__":
    args = parse_arguments()
    runner = YOLOvRunner()

    if args.command == "train":
        runner.train(args)
    elif args.command == "infer":
        runner.infer(args)
    elif args.command == "eval":
        runner.evaluate(args)
    else:
        print("❌ Please specify a command: `train`, `infer`, or `eval`.")