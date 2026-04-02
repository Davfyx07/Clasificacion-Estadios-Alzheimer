import argparse
import os
import random
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torchvision import transforms

from v2.train_engine import CLASSES, DEVICE, GradCAM, MODEL_CONFIG


DATASET_ROOT = "/home/davfy/Escritorio/Vision/dataset_balanceado2"
RESULTS_ROOT = "/home/davfy/Escritorio/Vision/v2/resultados"
OUTPUT_ROOT = "/home/davfy/Escritorio/Vision/v2/resultados/gradcam_manual"
MEAN = [0.485, 0.456, 0.406]
STD = [0.229, 0.224, 0.225]
DEFAULT_TEST_IMAGES = [
	"dataset_balanceado2/test/Non_Demented/non_67.jpg",
	"dataset_balanceado2/test/Very_Mild_Demented/verymild_56.jpg",
]


transform_eval = transforms.Compose(
	[
		transforms.Resize((224, 224)),
		transforms.ToTensor(),
		transforms.Normalize(mean=MEAN, std=STD),
	]
)


def pick_test_image(dataset_root: str, image_path: str = "") -> str:
	if image_path:
		if not os.path.isfile(image_path):
			raise FileNotFoundError(f"No existe la imagen: {image_path}")
		return image_path

	test_dir = os.path.join(dataset_root, "test")
	if not os.path.isdir(test_dir):
		raise FileNotFoundError(f"No existe carpeta test: {test_dir}")

	candidates: List[str] = []
	for class_name in CLASSES:
		class_dir = os.path.join(test_dir, class_name)
		if not os.path.isdir(class_dir):
			continue
		for name in os.listdir(class_dir):
			low = name.lower()
			if low.endswith((".png", ".jpg", ".jpeg", ".bmp", ".webp")):
				candidates.append(os.path.join(class_dir, name))

	if not candidates:
		raise RuntimeError(f"No se encontraron imágenes en: {test_dir}")

	return random.choice(candidates)


def load_image_tensor(img_path: str) -> Tuple[torch.Tensor, Image.Image]:
	pil_img = Image.open(img_path).convert("RGB")
	tensor = transform_eval(pil_img).unsqueeze(0).to(DEVICE)
	return tensor, pil_img


def infer_true_class(img_path: str) -> str:
	parent = os.path.basename(os.path.dirname(img_path))
	if parent in CLASSES:
		return parent
	return "Desconocida"


def model_configs_by_saved_name() -> Dict[str, dict]:
	return {cfg["model_name"]: cfg for cfg in MODEL_CONFIG.values()}


def discover_checkpoints(results_root: str) -> List[Tuple[str, str, dict]]:
	cfg_by_name = model_configs_by_saved_name()
	found: List[Tuple[str, str, dict]] = []

	for folder in sorted(os.listdir(results_root)):
		model_dir = os.path.join(results_root, folder)
		ckpt = os.path.join(model_dir, "best_model.pth")
		if not os.path.isdir(model_dir) or not os.path.isfile(ckpt):
			continue
		if folder not in cfg_by_name:
			continue
		found.append((folder, ckpt, cfg_by_name[folder]))

	if not found:
		raise RuntimeError(f"No se encontraron checkpoints válidos en: {results_root}")
	return found


def unnormalize(tensor_img: torch.Tensor) -> np.ndarray:
	img = tensor_img[0].detach().cpu().permute(1, 2, 0).numpy()
	img = np.clip(img * np.array(STD) + np.array(MEAN), 0, 1)
	return img


def refine_cam(cam: np.ndarray, out_h: int = 224, out_w: int = 224) -> np.ndarray:
	"""Reescala y suaviza CAM para evitar visualización en bloques."""
	cam_t = torch.tensor(cam, dtype=torch.float32).unsqueeze(0).unsqueeze(0)
	cam_t = F.interpolate(cam_t, size=(out_h, out_w), mode="bicubic", align_corners=False)
	cam_t = F.avg_pool2d(cam_t, kernel_size=7, stride=1, padding=3)
	cam_np = cam_t.squeeze().detach().cpu().numpy()
	cam_np = np.maximum(cam_np, 0)
	cam_np = (cam_np - cam_np.min()) / (cam_np.max() - cam_np.min() + 1e-8)
	return cam_np


def save_gradcam_figure(
	image_tensor: torch.Tensor,
	cam: np.ndarray,
	output_path: str,
	model_name: str,
	pred_label: str,
	pred_prob: float,
	true_label: str,
) -> None:
	img = unnormalize(image_tensor)
	cam = refine_cam(cam, out_h=img.shape[0], out_w=img.shape[1])
	os.makedirs(os.path.dirname(output_path), exist_ok=True)

	fig, axes = plt.subplots(1, 2, figsize=(12, 5))
	axes[0].imshow(img)
	axes[0].set_title("Imagen original")
	axes[0].axis("off")

	axes[1].imshow(img)
	axes[1].imshow(cam, cmap="jet", alpha=0.45)
	axes[1].set_title(
		f"{model_name}\nReal: {true_label} | Pred: {pred_label} ({pred_prob:.2%})"
	)
	axes[1].axis("off")

	fig.tight_layout()
	fig.savefig(output_path, dpi=150)
	plt.close(fig)


def run_manual_test(img_path: str, output_root: str, show: bool = False) -> None:
	image_tensor, _ = load_image_tensor(img_path)
	true_label = infer_true_class(img_path)
	checkpoints = discover_checkpoints(RESULTS_ROOT)
	img_stem = os.path.splitext(os.path.basename(img_path))[0]

	print("=" * 88)
	print(f"Imagen de prueba: {img_path}")
	print(f"Clase real: {true_label}")
	print("=" * 88)

	for model_name, ckpt_path, cfg in checkpoints:
		model = cfg["builder"](num_classes=len(CLASSES)).to(DEVICE)
		state = torch.load(ckpt_path, map_location=DEVICE)

		# Soporta checkpoints guardados como state_dict o diccionario con llaves conocidas.
		if isinstance(state, dict) and "state_dict" in state:
			state = state["state_dict"]
		if isinstance(state, dict) and "model_state_dict" in state:
			state = state["model_state_dict"]

		model.load_state_dict(state, strict=True)
		model.eval()

		with torch.no_grad():
			logits = model(image_tensor)
			probs = torch.softmax(logits, dim=1)
			pred_idx = int(torch.argmax(probs, dim=1).item())
			pred_prob = float(probs[0, pred_idx].item())

		gradcam = GradCAM(model, cfg["target_layer"](model))
		cam = gradcam.generate(image_tensor, pred_idx)
		gradcam.close()

		pred_label = CLASSES[pred_idx]
		ok = "OK" if pred_label == true_label else "FALLO"
		model_out_dir = os.path.join(output_root, model_name)
		out_path = os.path.join(model_out_dir, f"{img_stem}_gradcam.png")

		save_gradcam_figure(
			image_tensor=image_tensor,
			cam=cam,
			output_path=out_path,
			model_name=model_name,
			pred_label=pred_label,
			pred_prob=pred_prob,
			true_label=true_label,
		)

		print(f"[{model_name}] {ok} | real={true_label} | pred={pred_label} | p={pred_prob:.4f}")
		print(f"  Grad-CAM guardado en: {out_path}")

		if show:
			img = unnormalize(image_tensor)
			plt.figure(figsize=(6, 5))
			plt.imshow(img)
			plt.imshow(cam, cmap="jet", alpha=0.45)
			plt.title(f"{model_name} | Real={true_label} | Pred={pred_label} ({pred_prob:.2%})")
			plt.axis("off")
			plt.tight_layout()
			plt.show()

	print("=" * 88)
	print(f"Finalizado. Resultados en: {output_root}")


def parse_args() -> argparse.Namespace:
	parser = argparse.ArgumentParser(
		description="Prueba manual multi-modelo con predicción + Grad-CAM sobre una imagen de test."
	)
	parser.add_argument(
		"--image",
		type=str,
		default="",
		help="Ruta de imagen de test. Si no se pasa, elige una aleatoria en dataset/test.",
	)
	parser.add_argument(
		"--dataset-root",
		type=str,
		default=DATASET_ROOT,
		help="Ruta al dataset con subcarpetas train/val/test.",
	)
	parser.add_argument(
		"--output-root",
		type=str,
		default=OUTPUT_ROOT,
		help="Carpeta donde se guardan los mapas Grad-CAM.",
	)
	parser.add_argument(
		"--show",
		action="store_true",
		help="Muestra las figuras en pantalla además de guardarlas.",
	)
	return parser.parse_args()


if __name__ == "__main__":
	args = parse_args()
	if args.image:
		selected_image = pick_test_image(args.dataset_root, args.image)
		run_manual_test(selected_image, args.output_root, show=args.show)
	else:
		for img_path in DEFAULT_TEST_IMAGES:
			run_manual_test(img_path, args.output_root, show=args.show)
