"""Build on the existing TF Hub stylization model instead of replacing it.

This script keeps the arbitrary image stylization model as the project's
baseline and adds simple controls for experiments that strengthen style
preservation while still keeping recognizable content.
"""

from __future__ import annotations

import argparse
import os
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import PIL.Image
import tensorflow as tf

# Load model from tensorflow_hub.
os.environ["TFHUB_MODEL_LOAD_FORMAT"] = "COMPRESSED"
import tensorflow_hub as hub


DEFAULT_MODEL_URL = "https://tfhub.dev/google/magenta/arbitrary-image-stylization-v1-256/2"


@dataclass
class StyleTransferConfig:
    content_path: Path
    style_path: Path
    output_dir: Path = Path("results")
    image_size: int = 512
    style_passes: int = 1
    content_preservation: float = 0.0
    style_strength: float = 1.0
    generate_comparison_set: bool = False


def load_image(image_path: Path, image_size: int = 512) -> tf.Tensor:
    """Load, resize, and normalize an image for the TF Hub model."""
    image = tf.io.read_file(str(image_path))
    image = tf.io.decode_image(image, channels=3, expand_animations=False)
    image = tf.image.convert_image_dtype(image, tf.float32)

    shape = tf.cast(tf.shape(image)[:-1], tf.float32)
    longest_dim = tf.reduce_max(shape)
    scale = image_size / longest_dim
    new_shape = tf.cast(shape * scale, tf.int32)

    image = tf.image.resize(image, new_shape)
    image = image[tf.newaxis, :]
    return image


def tensor_to_image(image_tensor: tf.Tensor) -> PIL.Image.Image:
    """Convert a model output tensor to a PIL image."""
    image_tensor = tf.squeeze(image_tensor, axis=0)
    image_tensor = tf.clip_by_value(image_tensor, 0.0, 1.0)
    image_array = np.array(image_tensor * 255, dtype=np.uint8)
    return PIL.Image.fromarray(image_array)


def save_image(image_tensor: tf.Tensor, output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    tensor_to_image(image_tensor).save(output_path)


def blend_with_content(
    stylized_image: tf.Tensor,
    original_content: tf.Tensor,
    content_preservation: float,
) -> tf.Tensor:
    """Blend the stylized image with the original content image.

    Lower values keep more of the stylized output.
    Higher values preserve more structure from the original content image.
    """
    content_preservation = float(np.clip(content_preservation, 0.0, 1.0))
    return (
        stylized_image * (1.0 - content_preservation)
        + original_content * content_preservation
    )


def apply_style_strength(
    stylized_image: tf.Tensor,
    original_content: tf.Tensor,
    style_strength: float,
) -> tf.Tensor:
    """Interpolate between the content image and stylized output.

    Values above 1.0 push farther toward the stylized output and are clipped.
    """
    style_strength = max(style_strength, 0.0)
    adjusted = original_content + style_strength * (stylized_image - original_content)
    return tf.clip_by_value(adjusted, 0.0, 1.0)


def match_content_shape(
    image_tensor: tf.Tensor,
    reference_tensor: tf.Tensor,
) -> tf.Tensor:
    """Resize a tensor to the spatial shape of the reference content image."""
    target_shape = tf.shape(reference_tensor)[1:3]
    return tf.image.resize(image_tensor, target_shape)


def stylize_image(
    hub_model,
    content_image: tf.Tensor,
    style_image: tf.Tensor,
    style_passes: int = 1,
    content_preservation: float = 0.0,
    style_strength: float = 1.0,
) -> tf.Tensor:
    """Run the TF Hub model and optionally restylize the output.

    Repeated passes provide a simple way to test whether style features become
    more visible while content preservation keeps the result from drifting too
    far away from the source image.
    """
    current_image = content_image
    original_content = content_image

    for _ in range(max(style_passes, 1)):
        stylized_image = hub_model(
            tf.constant(current_image),
            tf.constant(style_image),
        )[0]
        stylized_image = match_content_shape(
            image_tensor=stylized_image,
            reference_tensor=original_content,
        )
        stylized_image = apply_style_strength(
            stylized_image=stylized_image,
            original_content=original_content,
            style_strength=style_strength,
        )
        current_image = blend_with_content(
            stylized_image=stylized_image,
            original_content=original_content,
            content_preservation=content_preservation,
        )

    return current_image


def build_output_name(config: StyleTransferConfig) -> str:
    content_name = config.content_path.stem
    style_name = config.style_path.stem
    return (
        f"{content_name}__{style_name}"
        f"__passes-{config.style_passes}"
        f"__preserve-{config.content_preservation:.2f}"
        f"__strength-{config.style_strength:.2f}.png"
    )


def generate_comparison_set(
    hub_model,
    content_image: tf.Tensor,
    style_image: tf.Tensor,
    config: StyleTransferConfig,
) -> None:
    """Save a small grid of experiments for report-ready comparisons."""
    experiment_settings = [
        {"style_passes": 1, "content_preservation": 0.0, "style_strength": 1.0},
        {"style_passes": 2, "content_preservation": 0.0, "style_strength": 1.0},
        {"style_passes": 2, "content_preservation": 0.15, "style_strength": 1.1},
        {"style_passes": 3, "content_preservation": 0.2, "style_strength": 1.15},
    ]

    for setting in experiment_settings:
        experiment_config = StyleTransferConfig(
            content_path=config.content_path,
            style_path=config.style_path,
            output_dir=config.output_dir / "comparisons",
            image_size=config.image_size,
            style_passes=setting["style_passes"],
            content_preservation=setting["content_preservation"],
            style_strength=setting["style_strength"],
        )
        result = stylize_image(
            hub_model=hub_model,
            content_image=content_image,
            style_image=style_image,
            style_passes=experiment_config.style_passes,
            content_preservation=experiment_config.content_preservation,
            style_strength=experiment_config.style_strength,
        )
        save_image(result, experiment_config.output_dir / build_output_name(experiment_config))


def run_pipeline(config: StyleTransferConfig) -> Path:
    hub_model = hub.load(DEFAULT_MODEL_URL)
    content_image = load_image(config.content_path, image_size=config.image_size)
    style_image = load_image(config.style_path, image_size=config.image_size)

    result = stylize_image(
        hub_model=hub_model,
        content_image=content_image,
        style_image=style_image,
        style_passes=config.style_passes,
        content_preservation=config.content_preservation,
        style_strength=config.style_strength,
    )

    output_path = config.output_dir / build_output_name(config)
    save_image(result, output_path)

    if config.generate_comparison_set:
        generate_comparison_set(
            hub_model=hub_model,
            content_image=content_image,
            style_image=style_image,
            config=config,
        )

    return output_path


def parse_args() -> StyleTransferConfig:
    parser = argparse.ArgumentParser(
        description=(
            "Run style transfer experiments that build on the existing "
            "TensorFlow Hub model."
        )
    )
    parser.add_argument("--content", required=True, help="Path to the content image.")
    parser.add_argument("--style", required=True, help="Path to the style image.")
    parser.add_argument(
        "--output-dir",
        default="results",
        help="Directory where output images will be saved.",
    )
    parser.add_argument(
        "--image-size",
        type=int,
        default=512,
        help="Maximum image dimension used before stylization.",
    )
    parser.add_argument(
        "--style-passes",
        type=int,
        default=1,
        help="Number of times to reapply the style transfer model.",
    )
    parser.add_argument(
        "--content-preservation",
        type=float,
        default=0.0,
        help="Blend amount from the original content image back into the output.",
    )
    parser.add_argument(
        "--style-strength",
        type=float,
        default=1.0,
        help="How strongly to push the output away from the content image.",
    )
    parser.add_argument(
        "--generate-comparison-set",
        action="store_true",
        help="Save several experiment variants for side-by-side comparison.",
    )
    args = parser.parse_args()

    return StyleTransferConfig(
        content_path=Path(args.content),
        style_path=Path(args.style),
        output_dir=Path(args.output_dir),
        image_size=args.image_size,
        style_passes=args.style_passes,
        content_preservation=args.content_preservation,
        style_strength=args.style_strength,
        generate_comparison_set=args.generate_comparison_set,
    )


def main() -> None:
    config = parse_args()
    output_path = run_pipeline(config)
    print(f"Saved stylized image to: {output_path}")


if __name__ == "__main__":
    main()
